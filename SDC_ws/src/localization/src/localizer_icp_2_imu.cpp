#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <string.h>
#include <time.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include<Eigen/Dense>

#include<pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/PCLHeader.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>

class Localizer{
private:

  float mapLeafSize = 1., scanLeafSize = 1.;
  std::vector<float> d_max_list, n_iter_list;
  float voxel_scan_range[3]={0.15f, 0.15f, 0.15f}, voxel_map_range[3]={0.15f, 0.15f, 0.15f};
  float pass_scan_map[3]={10.0, 10.0, 10.0}, pass_map_range[3]={100.0, 100.0, 100.0};

  ros::NodeHandle _nh;
  ros::Subscriber sub_points, sub_gps, sub_imu, sub_odom;
  ros::Publisher pub_points, pub_filter_p, pub_filter_m, pub_pose, pub_map;
  tf::TransformBroadcaster br;

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_points;
  pcl::PointXYZ gps_point;
  double frame_count;
  double imu_r = 0.0, imu_p = 0.0, imu_y = 0.0;
  bool gps_ready = false, map_ready = false, imu_ready = false, initialied = false;
  double init_yaw;
  double lidar_freq, odom_freq, freq_ratio;
  double diff_x, diff_y, diff_z;
  double odom_x, odom_y, odom_z;
  double previous_score, fix_rate;
  double last_time;
  Eigen::Matrix4f init_guess;
  int cnt = 0;
  
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  pcl::RadiusOutlierRemoval<pcl::PointXYZI> ro_filter;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> so_filter;
  pcl::PassThrough<pcl::PointXYZI> pass_filter;
  pcl::ExtractIndices<pcl::PointXYZI> cliper;


  std::string result_save_path;
  std::ofstream outfile;
  std::string map_path;
  geometry_msgs::Transform car2Lidar;
  std::string mapFrame, lidarFrame;

public:
  Localizer(ros::NodeHandle nh): map_points(new pcl::PointCloud<pcl::PointXYZI>) {
    std::vector<float> trans, rot;

    _nh = nh;

    _nh.param<std::vector<float>>("baselink2lidar_trans", trans, std::vector<float>());
    _nh.param<std::vector<float>>("baselink2lidar_rot", rot, std::vector<float>());
    _nh.param<std::string>("result_save_path", result_save_path, "result.csv");
    _nh.param<std::string>("map_path", map_path, "nuscenes_map.pcd");
    _nh.param<float>("scanLeafSize", scanLeafSize, 1.0);
    _nh.param<float>("mapLeafSize", mapLeafSize, 1.0);
    _nh.param<double>("init_yaw", init_yaw, 1.57);
    _nh.param<std::string>("mapFrame", mapFrame, "world");
    _nh.param<std::string>("lidarFrame", lidarFrame, "nuscenes_lidar");

    // set some initail value
    frame_count = 0;
    lidar_freq = 20.0;
    odom_freq = 13;
    freq_ratio = lidar_freq/odom_freq;

    diff_x = 0.0;
    diff_y = 0.0;
    diff_z = 0.0;
    odom_x = 0.0;
    odom_y = 0.0;
    odom_z = 0.0;

    previous_score = 0;
    fix_rate = 1.1;


    ROS_INFO("saving results to %s", result_save_path.c_str());
    outfile.open(result_save_path);
    outfile << "id,x,y,z,yaw,pitch,roll" << std::endl;

    if(trans.size() != 3 | rot.size() != 4){
      ROS_ERROR("transform not set properly");
    }

    car2Lidar.translation.x = trans.at(0);
    car2Lidar.translation.y = trans.at(1);
    car2Lidar.translation.z = trans.at(2);
    car2Lidar.rotation.x = rot.at(0);
    car2Lidar.rotation.y = rot.at(1);
    car2Lidar.rotation.z = rot.at(2);
    car2Lidar.rotation.w = rot.at(3);

    sub_points = _nh.subscribe("/lidar_points", 400000, &Localizer::pc_callback, this);
    sub_gps = _nh.subscribe("/gps", 400000, &Localizer::gps_callback, this);
    sub_imu = _nh.subscribe("/imu/data", 400000, &Localizer::imu_callback, this);
    sub_odom = nh.subscribe("/wheel_odometry", 4000000, &Localizer::odom_callback, this);

    pub_points = _nh.advertise<sensor_msgs::PointCloud2>("/transformed_points", 1);
    pub_filter_p = _nh.advertise<sensor_msgs::PointCloud2>("/filter_points", 1);
    pub_filter_m = _nh.advertise<sensor_msgs::PointCloud2>("/filter_map", 1);
    pub_pose = _nh.advertise<geometry_msgs::PoseStamped>("/lidar_pose", 1);
    pub_map = nh.advertise<sensor_msgs::PointCloud2>("/map", 1);
    init_guess.setIdentity();

    // read map and pub map
    map_points = (new pcl::PointCloud<pcl::PointXYZI>)->makeShared();
	  if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_path, *map_points) == -1){                                                                           
	  	PCL_ERROR("Couldn't read that pcd file\n");                         
	  	exit(0);
	  }
    else{
      ROS_INFO("Got map ~~");
      map_ready = true;
      sensor_msgs::PointCloud2::Ptr map_cloud(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*map_points, *map_cloud);
      map_cloud->header.frame_id = "world";
      pub_map.publish(*map_cloud);
      ROS_INFO("Pub Map");
    }
    ROS_INFO("%s initialized", ros::this_node::getName().c_str());
  }

  // Gentaly end the node
  ~Localizer(){
    if(outfile.is_open()) outfile.close();
  }
  
  void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    ROS_INFO("Got lidar message");
    std::cout << "Msg count frame : " << ++frame_count << std::endl;
    std::cout << "Msg time : " << msg->header.stamp << std::endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f result;

    while(!(gps_ready & map_ready)){
      ROS_WARN("waiting for map and gps data ...");
      ros::Duration(0.05).sleep();
      ros::spinOnce();
    }

    pcl::fromROSMsg(*msg, *scan_ptr);
    ROS_INFO("point size: %d", scan_ptr->width);
    result = align_map(scan_ptr, msg); // map -> lidar

    // publish transformed points
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    pcl_ros::transformPointCloud(result, *msg, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_points.publish(out_msg);

    // broadcast transforms
    tf::Matrix3x3 rot;
    rot.setValue(
      static_cast<double>(result(0, 0)), static_cast<double>(result(0, 1)), static_cast<double>(result(0, 2)), 
      static_cast<double>(result(1, 0)), static_cast<double>(result(1, 1)), static_cast<double>(result(1, 2)),
      static_cast<double>(result(2, 0)), static_cast<double>(result(2, 1)), static_cast<double>(result(2, 2))
    );
    tf::Vector3 trans(result(0, 3), result(1, 3), result(2, 3));
    tf::Transform transform(rot, trans);
    br.sendTransform(tf::StampedTransform(transform.inverse(), msg->header.stamp, lidarFrame, mapFrame));

    // publish lidar pose
    geometry_msgs::PoseStamped pose;
    pose.header = msg->header;
    pose.header.frame_id = mapFrame;
    pose.pose.position.x = trans.getX();
    pose.pose.position.y = trans.getY();
    pose.pose.position.z = trans.getZ();
    pose.pose.orientation.x = transform.getRotation().getX();
    pose.pose.orientation.y = transform.getRotation().getY();
    pose.pose.orientation.z = transform.getRotation().getZ();
    pose.pose.orientation.w = transform.getRotation().getW();
    pub_pose.publish(pose);

    Eigen::Affine3d transform_c2l, transform_m2l;
    transform_m2l.matrix() = result.cast<double>();
    transform_c2l = (tf2::transformToEigen(car2Lidar));
    Eigen::Affine3d tf_p = transform_m2l * transform_c2l.inverse();
    geometry_msgs::TransformStamped transform_m2c = tf2::eigenToTransform(tf_p);

    tf::Quaternion q(transform_m2c.transform.rotation.x, transform_m2c.transform.rotation.y, transform_m2c.transform.rotation.z, transform_m2c.transform.rotation.w);
    tfScalar yaw, pitch, roll;
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
    // std::cout << "Input orientation : " << yaw << ", " << pitch << ", " << roll << std::endl;

    outfile << ++cnt << "," << tf_p.translation().x() << "," << tf_p.translation().y() << "," << 0.0 << "," << yaw << "," << pitch << "," << roll << std::endl;

    if(cnt == 396)
      ROS_INFO("Bag data over, ICP finish calculation !");
    else
      std::cout << "current frame : " << cnt << std::endl;
    
    ROS_INFO("---------------------");

  }

  void gps_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
    ROS_INFO("Got GPS message");
    gps_point.x = msg->point.x;
    gps_point.y = msg->point.y;
    gps_point.z = msg->point.z;

    if(!initialied){
      std::cout << "GPS pose : " << gps_point.x << ", " << gps_point.y << std::endl;
    // if(true){
      geometry_msgs::PoseStamped pose;
      pose.header = msg->header;
      pose.pose.position = msg->point;
      pub_pose.publish(pose);
      // ROS_INFO("pub pose");

      tf::Matrix3x3 rot;
      rot.setIdentity();
      tf::Vector3 trans(msg->point.x, msg->point.y, msg->point.z);
      tf::Transform transform(rot, trans);
      br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "world", "nuscenes_lidar"));

      // init_guess(0, 3) = gps_point.x;
      // init_guess(1, 3) = gps_point.y;
      // init_guess(2, 3) = gps_point.z;
    }

    gps_ready = true;
    return;
  }

  void imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    ROS_INFO("IMU_get");
    double cur_time = msg->header.stamp.toSec();
    double ax = msg->linear_acceleration.x;
    double ay = msg->linear_acceleration.y;
    double duration = cur_time-last_time;
    init_guess(0, 3) += ax*duration*duration ;
    init_guess(1, 3) += ay*duration*duration ;
    last_time = cur_time;
    // tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    // tf::Matrix3x3 temp(q);
    // temp.getRPY(imu_r, imu_p, imu_y);
    // ROS_INFO("Imu data : %f", imu_y);
    imu_ready = true;
    return;
  }

  void odom_callback(const nav_msgs::Odometry::ConstPtr &msg){

		// diff_x = msg->pose.pose.position.x - odom_x;
		// diff_y = msg->pose.pose.position.y - odom_y;
		// diff_z = msg->pose.pose.position.z - odom_z;
		// odom_x = msg->pose.pose.position.x;
		// odom_y = msg->pose.pose.position.y;
		// odom_z = msg->pose.pose.position.z;

    // init_guess(0, 3) += diff_x ;
    // init_guess(1, 3) += diff_y ;
    // init_guess(2, 3) += diff_z ;

	}

  Eigen::Matrix4f align_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points, const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f result;

    // Downsampling scan_points
    ROS_INFO("Input scan point : %ld", scan_points->points.size());
    // downsampling_filter("passthrough_z", voxel_scan_range, scan_points, filtered_scan_ptr);
    filtered_scan_ptr = downsampling_filter("voxel", voxel_scan_range, scan_points);
    // downsampling_filter("nofilter", voxel_scan_range, scan_points, filtered_scan_ptr);
    ROS_INFO("filter output scan point : %ld", filtered_scan_ptr->points.size());

    // Downsampling map_points
    ROS_INFO("Input map point : %ld", map_points->points.size());
    if(initialied)
      filtered_map_ptr = downsampling_filter("passthrough_xy", pass_map_range, map_points);
    else
      filtered_map_ptr = downsampling_filter("nofilter", pass_map_range, map_points);
    ROS_INFO("filter output map point : %ld", filtered_map_ptr->points.size());


    /* Find the initial orientation for fist scan */
    if(!initialied){
      initial_guess_icp(filtered_scan_ptr, filtered_map_ptr, transformed_scan_ptr);
    }
	
    /* ICP */
    ROS_INFO("Start ICP");
    icp.setInputSource(filtered_scan_ptr);
    icp.setInputTarget(filtered_map_ptr);
    icp.setMaxCorrespondenceDistance(0.75);
    icp.setMaximumIterations(1000);
    icp.setTransformationEpsilon(1e-12);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setRANSACOutlierRejectionThreshold(0.05);
    icp.align(*transformed_scan_ptr, init_guess);
    result = icp.getFinalTransformation();

    ROS_INFO("Current score: %f", icp.getFitnessScore());

    /* Publish filter points */
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    sensor_msgs::PointCloud2::Ptr trans_msgs(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_scan_ptr, *trans_msgs);
    pcl_ros::transformPointCloud(result, *trans_msgs, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_filter_p.publish(out_msg);

    /* Publish filter map */
    sensor_msgs::PointCloud2::Ptr map_out(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_map_ptr, *map_out);
    map_out->header.frame_id = "world";
    pub_filter_m.publish(*map_out);
      
    /* Use result as next initial guess */
    init_guess = result;
    // std::cout << "Guess : " << init_guess(0, 3) << ", " << init_guess(1, 3) << ", " << init_guess(2, 3) << std::endl;
    // std::cout << "Diff : " << diff_x << ", " << diff_y << ", " << diff_z << std::endl;
    // init_guess(0, 3) += diff_x / freq_ratio;
    // init_guess(1, 3) += diff_y / freq_ratio;
    // init_guess(2, 3) += diff_z / freq_ratio;

    // if (icp.getFitnessScore() > previous_score || !icp.hasConverged())
		// 	freq_ratio * fix_rate;
		// else
		// 	freq_ratio / fix_rate;
		// previous_score = icp.getFitnessScore();

    return result;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampling_filter(std::string type, float range[], const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_points){
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>());

    if(type == "voxel"){
      voxel_filter.setInputCloud(input_points);
      voxel_filter.setLeafSize(range[0], range[1], range[2]);
      voxel_filter.filter(*filtered_ptr);
    }
    else if(type == "passthrough_z"){
      pass_filter.setInputCloud(input_points);
      pass_filter.setFilterFieldName("z");
      pass_filter.setFilterLimits(-2, 7);
      pass_filter.filter (*filtered_ptr);
    }
    else if(type == "passthrough_xy"){
      ROS_INFO("Inside pass through xy");
      pass_filter.setInputCloud(input_points);
      pass_filter.setFilterFieldName("x");
      pass_filter.setFilterLimits(init_guess(0,3) - range[0], init_guess(0,3) + range[0]);
      pass_filter.filter (*filtered_ptr);

      pass_filter.setInputCloud(filtered_ptr);
      pass_filter.setFilterFieldName("y");
      pass_filter.setFilterLimits(init_guess(1,3) - range[1], init_guess(1,3) + range[1]);
      pass_filter.filter (*filtered_ptr);
    }
    else{
      filtered_ptr = input_points;
    }

    return filtered_ptr;
  }

  void initial_guess_icp(pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_scan_ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_map_ptr,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_scan_ptr){

    // =============== Initial guess =============== //
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> first_icp;
    float yaw, min_yaw, min_score = std::numeric_limits<float>::max();
    Eigen::Matrix4f min_pose(Eigen::Matrix4f::Identity());

    // Rotate to find best initial guess
    ROS_INFO("Start initial guess !");
    // for(yaw = 1.0; yaw <= 3.0; yaw += 0.1){ //2*M_PI
    //   // Set initial guess of transformation
    //   // target = R*source + t , R is rotation, t is translation
    //   Eigen::AngleAxisf init_rotation(yaw, Eigen::Vector3f::UnitZ()); // we only consider yaw rotate
    //   Eigen::Translation3f init_translation(gps_point.x, gps_point.y, gps_point.z);
    //   init_guess = (init_translation * init_rotation).matrix();
    //   ROS_INFO("Testing yaw : %f", yaw);

    //   // Set the input source and target
    //   first_icp.setInputSource(filtered_scan_ptr);
    //   first_icp.setInputTarget(filtered_map_ptr);
    //   first_icp.setMaxCorrespondenceDistance(2); // Set the max correspondence distance to ?m (e.g., correspondences with higher distances will be ignored)
    //   first_icp.setMaximumIterations(1000);      // Set the maximum number of iterations (criterion 1)
    //   first_icp.setTransformationEpsilon(1e-12);  // Set the transformation epsilon (criterion 2)
    //   first_icp.setEuclideanFitnessEpsilon(1e-8);// Set the euclidean distance difference epsilon (criterion 3)
    //   first_icp.align(*transformed_scan_ptr, init_guess); // Perform the alignment
    //   // Check Converged
    //   bool converged = first_icp.hasConverged();
    //   ROS_INFO("has converged: %s", converged ? "true" : "false");
    //   // Get the score
    //   double c_score = first_icp.getFitnessScore();
    //   ROS_INFO("min score: %f, score: %f", min_score, c_score);
      
    //   if( c_score < min_score && converged){
    //     min_score = c_score;
    //     // Obtain the transformation that aligned cloud_source to cloud_source_registered
    //     min_yaw = yaw;
    //     min_pose = first_icp.getFinalTransformation();
    //     ROS_INFO("Update !!!");
    //   }
    //   ROS_INFO("---------------------");
    // }

		// init_guess << 	cos(init_yaw), -sin(init_yaw), 	0, 	gps_point.x,
		// 						    sin(init_yaw),  cos(init_yaw), 	0, 	gps_point.y,
		// 							  0            ,  0            , 	1, 	gps_point.z,
		// 							  0            ,  0            , 	0,  1;
    init_guess << 	cos(init_yaw), -sin(init_yaw), 	0, 	1774.59,
						        sin(init_yaw),  cos(init_yaw), 	0, 	866.342,
							      0            ,  0            , 	1, 	0.015302,
							      0            ,  0            , 	0,  1;
    last_time = ros::Time::now().toSec();
    ROS_INFO("Initial Set over");
    // ROS_INFO("Initial Yaw : %f", min_yaw);
    ROS_INFO("Initial Yaw : %f", init_yaw);
    ROS_INFO("---------------------");

    // set initial guess
    // init_guess = min_pose;
    initialied = true;
  }

  void showMinMax(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points){
    double max[3] = {0, 0, 0};
    double min[3] = {0, 0, 0};
    for(int i = 0; i < scan_points->points.size(); i++){
      if(scan_points->points[i].x > max[0])
        max[0] = scan_points->points[i].x;
      if(scan_points->points[i].x < min[0])
        min[0] = scan_points->points[i].x;

      if(scan_points->points[i].y > max[1])
        max[1] = scan_points->points[i].y;
      if(scan_points->points[i].x < min[1])
        min[1] = scan_points->points[i].y;
      
      if(scan_points->points[i].z > max[2])
        max[2] = scan_points->points[i].z;
      if(scan_points->points[i].z < min[2])
        min[2] = scan_points->points[i].z;
      // ROS_INFO("point : %f %f %f \r\n", scan_points->points[i].x, scan_points->points[i].y, scan_points->points[i].z);
    }
    ROS_INFO("max_p: %f %f %f \r\n", max[0], max[1], max[1]);
    ROS_INFO("min_p : %f %f %f \r\n", min[0], min[1], min[1]);
  }
};


int main(int argc, char* argv[]){
  ros::init(argc, argv, "localizer");
  ros::NodeHandle n("~");
  Localizer localizer(n);
  ros::spin();
  return 0;
}
