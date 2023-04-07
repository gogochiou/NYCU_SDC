#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <string.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>


#include<Eigen/Dense>

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

  // float mapLeafSize = 1., scanLeafSize = 1.;
  float mapLeafSize = 0.15f, scanLeafSize = 0.15f;
  std::vector<float> d_max_list, n_iter_list;

  ros::NodeHandle _nh;
  ros::Subscriber sub_map, sub_points, sub_gps;
  ros::Publisher pub_points, pub_filter_p, pub_pose;
  tf::TransformBroadcaster br;

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_points;
  pcl::PointXYZ gps_point;
  bool gps_ready = false, map_ready = false, initialied = false;
  Eigen::Matrix4f init_guess;
  int cnt = 0;
  float scan_range[3]={0.15, 0.15, 0.15}, map_range[3]={1.0, 1.0, 1.0}, map_range_pass[3]={50.0, 50.0, 1.0};
  
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  pcl::RadiusOutlierRemoval<pcl::PointXYZI> ro_filter;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> so_filter;
  pcl::PassThrough<pcl::PointXYZI> pass_filter;
  pcl::ExtractIndices<pcl::PointXYZI> cliper;


  std::string result_save_path;
  std::ofstream outfile;
  geometry_msgs::Transform car2Lidar;
  Eigen::Matrix4f c2l_transform;
  std::string mapFrame, lidarFrame, carFrame;

public:
  Localizer(ros::NodeHandle nh): map_points(new pcl::PointCloud<pcl::PointXYZI>) {
    std::vector<float> trans, rot;

    _nh = nh;

    _nh.param<std::vector<float>>("baselink2lidar_trans", trans, std::vector<float>());
    _nh.param<std::vector<float>>("baselink2lidar_rot", rot, std::vector<float>());
    _nh.param<std::string>("result_save_path", result_save_path, "result.csv");
    _nh.param<float>("scanLeafSize", scanLeafSize, 1.0);
    _nh.param<float>("mapLeafSize", mapLeafSize, 1.0);
    _nh.param<std::string>("mapFrame", mapFrame, "world");
    _nh.param<std::string>("lidarFrame", lidarFrame, "nuscenes_lidar");
    _nh.param<std::string>("carFrame", carFrame, "My car");


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

    /* setting my eigen transform */
    Eigen::Quaternionf link_quaternion(rot.at(3), rot.at(0), rot.at(1), rot.at(2));
		Eigen::Matrix3f link_rotation = link_quaternion.toRotationMatrix();
    c2l_transform << link_rotation(0, 0), link_rotation(0, 1), link_rotation(0, 2), trans.at(0),
									         link_rotation(1, 0), link_rotation(1, 1), link_rotation(1, 2), trans.at(1),
									         link_rotation(2, 0), link_rotation(2, 1), link_rotation(2, 2), trans.at(2),
													 0, 		   		        0,                   0,                   1;		

    sub_map = _nh.subscribe("/map", 1, &Localizer::map_callback, this);
    sub_points = _nh.subscribe("/lidar_points", 400000, &Localizer::pc_callback, this);
    sub_gps = _nh.subscribe("/gps", 1, &Localizer::gps_callback, this);
    pub_points = _nh.advertise<sensor_msgs::PointCloud2>("/transformed_points", 1);
    pub_filter_p = _nh.advertise<sensor_msgs::PointCloud2>("/filter_points", 1);
    pub_pose = _nh.advertise<geometry_msgs::PoseStamped>("/lidar_pose", 1);
    init_guess.setIdentity();
    ROS_INFO("%s initialized", ros::this_node::getName().c_str());
  }

  // Gentaly end the node
  ~Localizer(){
    if(outfile.is_open()) outfile.close();
  }

  void map_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    ROS_INFO("Got map message");
    pcl::fromROSMsg(*msg, *map_points);
    map_ready = true;
  }
  
  void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    ROS_INFO("Got lidar message");
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f result;

    while(!(gps_ready & map_ready)){
      ROS_WARN("waiting for map and gps data ...");
      ros::Duration(0.05).sleep();
      ros::spinOnce();
    }

    pcl::fromROSMsg(*msg, *scan_ptr);
    ROS_INFO("point size: %d", scan_ptr->width);
    result = align_map(scan_ptr, msg); // map -> car

    // broadcast transforms
    tf::Matrix3x3 m2c_rot;
    m2c_rot.setValue(
      static_cast<double>(result(0, 0)), static_cast<double>(result(0, 1)), static_cast<double>(result(0, 2)), 
      static_cast<double>(result(1, 0)), static_cast<double>(result(1, 1)), static_cast<double>(result(1, 2)),
      static_cast<double>(result(2, 0)), static_cast<double>(result(2, 1)), static_cast<double>(result(2, 2))
    );
    tf::Vector3 m2c_trans(result(0, 3), result(1, 3), result(2, 3));
    tf::Transform m2c_transform(m2c_rot, m2c_trans);
    br.sendTransform(tf::StampedTransform(m2c_transform.inverse(), msg->header.stamp, carFrame, mapFrame));

    // publish lidar pose
    geometry_msgs::PoseStamped pose;
    pose.header = msg->header;
    pose.header.frame_id = mapFrame;
    pose.pose.position.x = m2c_trans.getX();
    pose.pose.position.y = m2c_trans.getY();
    pose.pose.position.z = m2c_trans.getZ();
    pose.pose.orientation.x = m2c_transform.getRotation().getX();
    pose.pose.orientation.y = m2c_transform.getRotation().getY();
    pose.pose.orientation.z = m2c_transform.getRotation().getZ();
    pose.pose.orientation.w = m2c_transform.getRotation().getW();
    pub_pose.publish(pose);

    // Output Car pose (result)
    tf2::Quaternion m2c_quaternion( m2c_transform.getRotation().getX(), m2c_transform.getRotation().getY(),
                                    m2c_transform.getRotation().getZ(), m2c_transform.getRotation().getW());
    tf::Quaternion q(m2c_quaternion[0], m2c_quaternion[1], m2c_quaternion[2], m2c_quaternion[3]);
    double yaw, pitch, roll;
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
    outfile << ++cnt << "," << result(0, 3) << "," << result(1, 3) << "," << result(2, 3) << "," << yaw << "," << pitch << "," << roll << std::endl;

    if(cnt == 201){
      ROS_INFO("Bag data over, ICP finish calculation !");
    }

  }

  void gps_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
    std::cout << "Get GPS msg !" << std::endl;
    gps_point.x = msg->point.x;
    gps_point.y = msg->point.y;
    gps_point.z = msg->point.z;

    if(!initialied){
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
    }

    gps_ready = true;
    return;
  }

  Eigen::Matrix4f align_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points, const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points_trans(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f result;

    transformPointCloud(*scan_points, *scan_points_trans, c2l_transform); // transform from lidar to base_link

    if(!initialied){
      initial_guess_icp(scan_points, filtered_scan_ptr, filtered_map_ptr, transformed_scan_ptr);
    }
    else{
      // =============== Down Sampling =============== //
      // scan_points
      showMinMax(scan_points_trans);
      downsampling_filter("voxel", scan_range, scan_points_trans, filtered_scan_ptr);
      ROS_INFO("filter output scan point : %ld", filtered_scan_ptr->points.size());

      // map_points
      downsampling_filter("passthrough", map_range_pass, map_points, filtered_map_ptr);
      ROS_INFO("filter output map point : %ld", filtered_map_ptr->points.size());
    }

    // =============== ICP =============== //
    icp.setInputSource(filtered_scan_ptr);
    icp.setInputTarget(filtered_map_ptr);
    icp.setMaxCorrespondenceDistance(1);
    icp.setMaximumIterations(1000);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-5);
    icp.align(*transformed_scan_ptr, init_guess);
    result = icp.getFinalTransformation();

    ROS_INFO("Current score: %f", icp.getFitnessScore());
    ROS_INFO("---------------------");

    /* Publish points (transformed) */
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    sensor_msgs::PointCloud2::Ptr trans_msgs(new sensor_msgs::PointCloud2);
    // origin scan
    pcl::toROSMsg(*scan_points_trans, *trans_msgs);
    pcl_ros::transformPointCloud(result, *trans_msgs, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_points.publish(out_msg);
    // filtered scan
    pcl::toROSMsg(*filtered_scan_ptr, *trans_msgs);
    pcl_ros::transformPointCloud(result, *trans_msgs, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_filter_p.publish(out_msg);
      
    /* Use result as next initial guess */
    init_guess = result;
    return result;
  }

  void initial_guess_icp(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points, pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_scan_ptr,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_map_ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_scan_ptr){
    // =============== Down Sampling =============== //
    // scan_points
    ROS_INFO("Input scan point : %ld", scan_points->points.size());
    transformPointCloud(*scan_points, *scan_points, c2l_transform); // transform from lidar to base_link
    showMinMax(scan_points);
    downsampling_filter("voxel", scan_range, scan_points, filtered_scan_ptr);
    ROS_INFO("filter output scan point : %ld", filtered_scan_ptr->points.size());

    // map_points
    downsampling_filter("voxel", map_range, map_points, filtered_map_ptr);
    ROS_INFO("filter output map point : %ld", filtered_map_ptr->points.size());

    // =============== Initial guess =============== //
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> first_icp;
    float yaw, min_yaw, min_score = std::numeric_limits<float>::max();
    Eigen::Matrix4f min_pose(Eigen::Matrix4f::Identity());

    // Rotate to find best initial guess
    ROS_INFO("Start initial guess !");
    for(yaw = 2.45; yaw <= 2.5; yaw += 0.05){ //2*M_PI
      // Set initial guess of transformation
      // target = R*source + t , R is rotation, t is translation
      Eigen::AngleAxisf init_rotation(yaw, Eigen::Vector3f::UnitZ()); // we only consider yaw rotate
      Eigen::Translation3f init_translation(gps_point.x, gps_point.y, gps_point.z);
      init_guess = (init_translation * init_rotation).matrix();
      ROS_INFO("Testing yaw : %f", yaw);

      // Set the input source and target
      first_icp.setInputSource(filtered_scan_ptr);
      first_icp.setInputTarget(filtered_map_ptr);
      first_icp.setMaxCorrespondenceDistance(2); // Set the max correspondence distance to ?m (e.g., correspondences with higher distances will be ignored)
      first_icp.setMaximumIterations(1000);      // Set the maximum number of iterations (criterion 1)
      first_icp.setTransformationEpsilon(1e-8);  // Set the transformation epsilon (criterion 2)
      first_icp.setEuclideanFitnessEpsilon(1e-5);// Set the euclidean distance difference epsilon (criterion 3)
      first_icp.align(*transformed_scan_ptr, init_guess); // Perform the alignment
      // Check Converged
      bool converged = first_icp.hasConverged();
      ROS_INFO("has converged: %s", converged ? "true" : "false");
      // Get the score
      double c_score = first_icp.getFitnessScore(0.5);
      ROS_INFO("min score: %f, score: %f", min_score, c_score);
      
      if( c_score < min_score && converged){
        min_score = c_score;
        // Obtain the transformation that aligned cloud_source to cloud_source_registered
        min_yaw = yaw;
        min_pose = first_icp.getFinalTransformation();
        ROS_INFO("Update !!!");
      }
      ROS_INFO("---------------------");
    }
    ROS_INFO("Initial Set over");
    ROS_INFO("Initial Yaw : %f", min_yaw);
    ROS_INFO("---------------------");

    // set initial guess
    init_guess = min_pose;
    initialied = true;
  }

  void downsampling_filter(std::string type, float range[], const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_points, pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_ptr){
    filtered_ptr = input_points;

    /* voxel grid filter*/
    if(type == "voxel"){
      voxel_filter.setInputCloud(input_points);
      voxel_filter.setLeafSize(range[0], range[1], range[2]);
      voxel_filter.filter(*filtered_ptr);
    }
    else if(type == "passthrough"){
      pass_filter.setInputCloud(input_points);
      pass_filter.setFilterFieldName("x");
      pass_filter.setFilterLimits(init_guess(0,3) - range[0], init_guess(0,3) + range[0]);
      pass_filter.filter (*filtered_ptr);

      pass_filter.setInputCloud(filtered_ptr);
      pass_filter.setFilterFieldName("y");
      pass_filter.setFilterLimits(init_guess(1,3) - range[1], init_guess(1,3) + range[1]);
      pass_filter.filter (*filtered_ptr);
    }

    return;
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
