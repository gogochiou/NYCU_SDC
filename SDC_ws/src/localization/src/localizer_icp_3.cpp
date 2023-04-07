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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core> 

#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/PCLHeader.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>

/* Load map version */
// KF odom, GPS for x, y ; imu_yaw as real world yaw
// using odom position difference predict the position

class Localizer{
private:

  ros::NodeHandle _nh;
  ros::Subscriber sub_points, sub_gps, sub_imu, sub_odom;
  ros::Publisher pub_points, pub_filter_p, pub_filter_m, pub_pose, pub_map;
  tf::TransformBroadcaster br;

  //==== For car state, code state ====//
  bool map_ready = false, gps_ready = false, imu_ready = false, odom_ready = false, initialied = false;
  double init_yaw;
  Eigen::Matrix4f init_guess;

  //==== Point cloud and lidar used ====//
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_points;
  sensor_msgs::PointCloud2::ConstPtr record_pc;
  int pc_frame; // pc frame count
  ros::Time pc_lastTime;

  //==== GPS part ====//
  pcl::PointXYZ gps_point;
  ros::Time gps_lastTime;
  int gps_frame;
  double gps_diff_x;
  double gos_diff_y;
  

  //==== IMU part ====//
  ros::Time imu_lastTime;
  double imu_yaw;
  double lidar_yaw;

  //==== Odom part ====//
  ros::Time odom_lastTime;
  geometry_msgs::PoseStamped odom_pose;
  double odom_vx;
  double odom_vy;
  double odom_x;
  double odom_y; // absolute position for odom
  double odom_x_relate, odom_y_relate; // related position
  double diff_x, diff_y;
  double last_score, fix_rate;
  double freq_ratio;

  //==== KF part ====//
  Eigen::Matrix2f At;
  Eigen::Vector2f Bt; // it is equal to 1 in my case
  Eigen::Matrix2f update_matrix; // Ct
  Eigen::Matrix2f odom_cov;
  Eigen::Matrix2f gps_cov;
  Eigen::Matrix2f state_cov;
  
  //===== ICP and Filter ====//
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  pcl::RadiusOutlierRemoval<pcl::PointXYZI> ro_filter;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> so_filter;
  pcl::PassThrough<pcl::PointXYZI> pass_filter;
  pcl::ExtractIndices<pcl::PointXYZI> cliper;
  // Filter param
  float mapLeafSize, scanLeafSize; // not used in my version
  float voxel_scan_range[3]={0.15f, 0.15f, 0.15f}, voxel_map_range[3]={0.15f, 0.15f, 0.15f}; // for voxel filter
  float pass_scan_map[3]={10.0, 10.0, 10.0}, pass_map_range[3]={80.0, 80.0, 1.0};  // for passthrough filter
  // ICP param
  double MCDis;

  //====  Map, Output path and TF used ====//
  std::string result_save_path;
  std::ofstream outfile;
  std::string map_path; // loading map path
  geometry_msgs::Transform car2Lidar;
  std::string mapFrame, lidarFrame;

public:
  Localizer(ros::NodeHandle nh): map_points(new pcl::PointCloud<pcl::PointXYZI>) {
    std::vector<float> trans, rot;

    _nh = nh;

    //==== Reading Parameter ====//
    _nh.param<std::vector<float>>("baselink2lidar_trans", trans, std::vector<float>());
    _nh.param<std::vector<float>>("baselink2lidar_rot", rot, std::vector<float>());
    _nh.param<std::string>("result_save_path", result_save_path, "result.csv");
    _nh.param<std::string>("map_path", map_path, "nuscenes_map.pcd");
    _nh.param<double>("init_yaw", init_yaw, 1.57);
    _nh.param<double>("max_corresponse_dist", MCDis, 4);
    _nh.param<std::string>("mapFrame", mapFrame, "world");
    _nh.param<std::string>("lidarFrame", lidarFrame, "nuscenes_lidar");

    //==== Parameter initialize ====//
    pc_frame = 0;
    gps_frame = 0;

    imu_yaw = -2.18; 
    lidar_yaw = init_yaw;

    odom_vx = 7.02;
    odom_vy = -0.110347;
    diff_x = 0.0;
    diff_y = 0.0;
    odom_x_relate = 0.0;
    odom_y_relate = 0.0;

    last_score = 100.0;
    freq_ratio = 20.0/10; //12.5
    fix_rate = 1.3;

    At << 1.0, 0.0,
          0.0, 1.0;
    update_matrix << 1.0, 0.0,
                     0.0, 1.0;
    gps_cov << 0.3, 0.0,
               0.0, 0.3; // 0.1
    odom_cov << 0.05, 0.0,
                0.0 , 0.05;
    state_cov << 1.0, 0.0,
                 0.0, 1.0;

    //==== Open saving path ====//
    ROS_INFO("saving results to %s", result_save_path.c_str());
    outfile.open(result_save_path);
    outfile << "id,x,y,z,yaw,pitch,roll" << std::endl;

    //==== TF set ====//
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

    //==== Setting node subscriber and publisher ====//
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

    //==== Read map and Pub map ====//
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

  /* Gentaly end the node */
  ~Localizer(){
    if(outfile.is_open()) outfile.close();
  }
  
  /* Point cloud (Lidar) call back --> Main Loop Part */
  void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    std::cout << "Got lidar msg " << std::endl;
    std::cout << "PointCloud frame : " << ++pc_frame << std::endl;
    std::cout << "Msg time : " << msg->header.stamp << std::endl;

    //==== Initial state (GPS) and map ready for used ====//
    while(!(gps_ready & map_ready)){
      ROS_WARN("waiting for map and gps data ...");
      ros::Duration(0.05).sleep();
      ros::spinOnce();
    }

    ros::Duration pc_dt = msg->header.stamp - pc_lastTime;
    record_pc = msg;

    // KF_predict(odom_vx, odom_vy, pc_dt.toSec(), odom_cov);
    // pure_odom_predict(odom_vx, odom_vy, pc_dt.toSec());
    pointcloud_process(record_pc);

    pc_lastTime = msg->header.stamp;
  }

  /* GPS call back */
  void gps_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
    std::cout << "Got GPS msg : " << msg->header.seq + 1 << std::endl;
    gps_frame = msg->header.seq + 1;
    gps_point.x = msg->point.x;
    gps_point.y = msg->point.y;
    gps_point.z = msg->point.z;

    if(!initialied){
      std::cout << "GPS pose : " << gps_point.x << ", " << gps_point.y << std::endl;
      geometry_msgs::PoseStamped pose;
      pose.header = msg->header;
      pose.pose.position = msg->point;
      pub_pose.publish(pose);

      tf::Matrix3x3 rot;
      rot.setIdentity();
      tf::Vector3 trans(msg->point.x, msg->point.y, msg->point.z);
      tf::Transform transform(rot, trans);
      br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "world", "nuscenes_lidar"));

      //==== Setting initial state as gps pose ====//
      imu_lastTime = msg->header.stamp;
      odom_lastTime = msg->header.stamp;
      pc_lastTime = msg->header.stamp;
      odom_x = gps_point.x;
      odom_y = gps_point.y;
    }
    // else{
    //   KF_update(gps_point.x, gps_point.y, gps_cov);
    // }

    gps_lastTime = msg->header.stamp;
    gps_ready = true;
    return;
  }

  /* IMU call back */
  void imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    ros::Duration imu_duration = msg->header.stamp - imu_lastTime;
    double dt = imu_duration.toSec();

    double vth = msg->angular_velocity.z;
    imu_yaw += (vth*dt);
    lidar_yaw += (vth*dt);
    // yaw_update(); // imu is not that good for replacing yaw of icp init_guess

    // std::cout << "get imu msg : " << imu_yaw << std::endl;
    imu_lastTime = msg->header.stamp;
    imu_ready = true;
    return;
  }

  /* odom call back */
  void odom_callback(const nav_msgs::Odometry::ConstPtr& msg){
    std::cout << "======== Odom msg ========" << std::endl;
    ros::Duration odom_duration = msg->header.stamp - odom_lastTime;
    double dt = odom_duration.toSec();
    
    /* Use velocity integral --> bettet way */
    double vx = msg->twist.twist.linear.x;
    double vy = msg->twist.twist.linear.y;

    odom_vx = vx;
    odom_vy = vy;

    double pose_x = msg->pose.pose.position.x;
    double pose_y = msg->pose.pose.position.y;

    diff_x = pose_x*cos(imu_yaw)-pose_y*sin(imu_yaw) - odom_x_relate;
    diff_y = pose_x*sin(imu_yaw)+pose_y*cos(imu_yaw) - odom_y_relate;
    odom_x_relate = pose_x*cos(imu_yaw)-pose_y*sin(imu_yaw);
    odom_y_relate = pose_x*sin(imu_yaw)+pose_y*cos(imu_yaw);
    std::cout << "Current yaw : " << imu_yaw << std::endl;
    std::cout << odom_x_relate << ", " << odom_y_relate << std::endl;

    odom_lastTime = msg->header.stamp;
    odom_ready = true;
    std::cout << "================================" << std::endl;
    return;
	}

  void pure_odom_predict(float v_x, float v_y, float dt){
    std::cout << "========Pure odom predict========" << std::endl;
    std::cout << "Current pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "my yaw : " << imu_yaw << std::endl;
    float global_vx = v_x*cos(imu_yaw) - v_y*sin(imu_yaw);
    float global_vy = v_x*sin(imu_yaw) + v_y*cos(imu_yaw);
    std::cout << "vx, vy : " << global_vx << ", " << global_vy << std::endl;

    odom_x += global_vx*dt;
    odom_y += global_vy*dt;
    init_guess(0, 3) = odom_x;
    init_guess(1, 3) = odom_y;
    // init_guess(0, 3) += global_vx*dt;
    // init_guess(1, 3) += global_vy*dt;
    std::cout << "Predict pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "================================" << std::endl;
    return;
  }

  void yaw_update(){
    std::cout << "================================" << std::endl;
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix << init_guess(0, 0), init_guess(0, 1), init_guess(0, 2),
                       init_guess(1, 0), init_guess(1, 1), init_guess(1, 2),
                       init_guess(2, 0), init_guess(2, 1), init_guess(2, 2);
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX -> yaw, pitch, roll
    std::cout << "yaw(z) pitch(y) roll(x) = " << euler_angles.transpose() << std::endl;

    Eigen::Vector3d ea(lidar_yaw, euler_angles(1), euler_angles(2));
    rotation_matrix = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) * 
                      Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) * 
                      Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());

    for(int i =0; i<3; i++){
      for(int j=0; j<3; j++)
        init_guess(i, j) = rotation_matrix(i, j);
    }
    std::cout << "yaw(z) pitch(y) roll(x) = " << ea << std::endl;
    std::cout << "================================" << std::endl;
    return;
  }

  void KF_predict(float v_x, float v_y, float dt, Eigen::Matrix2f predict_cov){
    std::cout << "==========KF predict==========" << std::endl;
    std::cout << "Input velocity : " << v_x << ", " << v_y << std::endl;
    std::cout << "Duration time : " << dt << std::endl;
    std::cout << "Predict cov : " << predict_cov(0, 0) << ", " << predict_cov(1, 1) << std::endl;
    std::cout << "Current pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "Current cov : " << state_cov(0, 0) << ", " << state_cov(1, 1) << std::endl;

    Eigen::Vector2f input_velocity(v_x*cos(imu_yaw) - v_y*sin(imu_yaw), v_x*sin(imu_yaw) + v_y*cos(imu_yaw));
    std::cout << "Input transform : " << input_velocity(0) << ", " << input_velocity(1) << std::endl;
    Eigen::Vector2f current_state(init_guess(0, 3), init_guess(1, 3));
    
    Eigen::Vector2f miu = At*current_state + input_velocity*dt;
    Eigen::Matrix2f sigma = At*state_cov*At.transpose() + predict_cov;

    init_guess(0, 3) = miu(0);
    init_guess(1, 3) = miu(1);
    state_cov = sigma;

    std::cout << "Predict finish !" << std:: endl;
    std::cout << "Predict pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "Predict cov : " << state_cov(0, 0) << ", " << state_cov(1, 1) << std::endl;
    std::cout << "================================" << std::endl;
    return;
  }

  void KF_update(float update_x, float update_y, Eigen::Matrix2f update_cov){
    std::cout << "==========KF update==========" << std::endl;
    std::cout << "Update pose : " << update_x << ", " << update_y << std::endl;
    std::cout << "Update cov : " << update_cov(0, 0) << ", " << update_cov(1, 1) << std::endl;
    std::cout << "Current pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "Current cov : " << state_cov(0, 0) << ", " << state_cov(1, 1) << std::endl;

    Eigen::Vector2f update_state(update_x, update_y);
    Eigen::Vector2f current_state(init_guess(0, 3), init_guess(1, 3));
    Eigen::Matrix2f identity_matrix;
    identity_matrix << 1.0, 0.0,
                       0.0, 1.0;
    
    Eigen::Matrix2f Kt = state_cov*update_matrix.transpose()*(update_matrix*state_cov*update_matrix.transpose() + update_cov).inverse();
    Eigen::Vector2f miu = current_state + Kt*(update_state - update_matrix*current_state);
    Eigen::Matrix2f sigma = (identity_matrix - Kt*update_matrix)*state_cov;

    init_guess(0, 3) = miu(0);
    init_guess(1, 3) = miu(1);
    state_cov = sigma;

    std::cout << "KF finish !" << std:: endl;
    std::cout << "KF pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "KF cov : " << state_cov(0, 0) << ", " << state_cov(1, 1) << std::endl;
    std::cout << "================================" << std::endl;
    return;
  }

  void pointcloud_process(const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f result;

    //==== Doing ICP ====//
    pcl::fromROSMsg(*msg, *scan_ptr);
    result = align_map(scan_ptr, msg); // map -> lidar

    //==== Publish transformed points ====//
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    pcl_ros::transformPointCloud(result, *msg, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_points.publish(out_msg);

    //==== Broadcast transforms(TF of world -> lidar) ====// 
    tf::Matrix3x3 rot;
    rot.setValue(
      static_cast<double>(result(0, 0)), static_cast<double>(result(0, 1)), static_cast<double>(result(0, 2)), 
      static_cast<double>(result(1, 0)), static_cast<double>(result(1, 1)), static_cast<double>(result(1, 2)),
      static_cast<double>(result(2, 0)), static_cast<double>(result(2, 1)), static_cast<double>(result(2, 2))
    );
    tf::Vector3 trans(result(0, 3), result(1, 3), result(2, 3));
    tf::Transform transform(rot, trans);
    br.sendTransform(tf::StampedTransform(transform.inverse(), msg->header.stamp, lidarFrame, mapFrame));

    //==== publish lidar pose ====//
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

    //==== Setting output data : Car position in world ====//
    Eigen::Affine3d transform_c2l, transform_m2l;
    transform_m2l.matrix() = result.cast<double>();
    transform_c2l = (tf2::transformToEigen(car2Lidar));
    Eigen::Affine3d tf_p = transform_m2l * transform_c2l.inverse();
    geometry_msgs::TransformStamped transform_m2c = tf2::eigenToTransform(tf_p);
    // Transfer q->rpy
    tf::Quaternion q(transform_m2c.transform.rotation.x, transform_m2c.transform.rotation.y, transform_m2c.transform.rotation.z, transform_m2c.transform.rotation.w);
    tfScalar yaw, pitch, roll;
    tf::Matrix3x3 mat(q);
    mat.getEulerYPR(yaw, pitch, roll);
    // Output
    outfile << pc_frame << "," << tf_p.translation().x() << "," << tf_p.translation().y() << "," << 0.0 << "," << yaw << "," << pitch << "," << roll << std::endl;

    // if(pc_frame == 1)
    //   imu_yaw = (double)yaw;

    if(pc_frame == 389)
      ROS_INFO("Bag data over, ICP finish calculation !");
    else
      std::cout << "Output frame " << pc_frame << " complete !"<< std::endl;
    
    std::cout << "------------------"<< std::endl;
  }

  /* ICP works part */
  Eigen::Matrix4f align_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr scan_points, const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f result;

    //==== Downsampling scan_points ====//
    std::cout << "Input scan point : " << scan_points->points.size() << std::endl;
    filtered_scan_ptr = downsampling_filter("voxel", voxel_scan_range, scan_points);
    // filtered_scan_ptr = downsampling_filter("nofilter", voxel_scan_range, scan_points);
    // downsampling_filter("nofilter", voxel_scan_range, scan_points, filtered_scan_ptr);
    std::cout << "filter output scan point : " << filtered_scan_ptr->points.size()<< std::endl;

    //==== Downsampling map_points ====//
    std::cout << "Input map point : " << map_points->points.size() << std::endl;
    if(initialied)
      filtered_map_ptr = downsampling_filter("passthrough_xy", pass_map_range, map_points);
    else
      filtered_map_ptr = downsampling_filter("nofilter", pass_map_range, map_points);
    // filtered_map_ptr = downsampling_filter("nofilter", pass_map_range, map_points);
    std::cout << "filter output map point : " << filtered_map_ptr->points.size()<< std::endl;

    //==== Find the initial orientation for fist scan ====/
    if(!initialied){
      // initial_guess_icp(filtered_scan_ptr, filtered_map_ptr, transformed_scan_ptr);
      initial_guess_known(filtered_scan_ptr, filtered_map_ptr, transformed_scan_ptr);
    }
	
    //==== ICP ====//
    std::cout << "Start ICP " << std::endl;
    icp.setInputSource(filtered_scan_ptr);
    icp.setInputTarget(filtered_map_ptr);
    icp.setMaxCorrespondenceDistance(MCDis);
    icp.setMaximumIterations(1000);
    icp.setTransformationEpsilon(1e-12);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setRANSACOutlierRejectionThreshold(0.05);
    icp.align(*transformed_scan_ptr, init_guess);
    result = icp.getFinalTransformation();
    double score = icp.getFitnessScore();

    std::cout << "Current score : " << score << std::endl;

    //==== Publish filter points ====//
    sensor_msgs::PointCloud2::Ptr out_msg(new sensor_msgs::PointCloud2);
    sensor_msgs::PointCloud2::Ptr trans_msgs(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_scan_ptr, *trans_msgs);
    pcl_ros::transformPointCloud(result, *trans_msgs, *out_msg);
    out_msg->header = msg->header;
    out_msg->header.frame_id = mapFrame;
    pub_filter_p.publish(out_msg);

    //==== Publish filter map ====//
    sensor_msgs::PointCloud2::Ptr map_out(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_map_ptr, *map_out);
    map_out->header.frame_id = "world";
    pub_filter_m.publish(*map_out);
      
    //==== Use result as next initial guess ====//
    init_guess = result;

    //==== predtict next time position by odom ====//
    std::cout << "Guess : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    std::cout << "Diff : " << diff_x << ", " << diff_y << std::endl;
    init_guess(0, 3) += diff_x / freq_ratio;
    init_guess(1, 3) += diff_y / freq_ratio;

    if (score > last_score || !icp.hasConverged())
			freq_ratio * fix_rate;
		else
			freq_ratio / fix_rate;
		last_score = score;

    std::cout << "ICP pose : " << init_guess(0, 3) << ", " << init_guess(1, 3) << std::endl;
    return result;
  }

  /* Function for downsampling */
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

  /* Setting initial guess : known initial position */
  void initial_guess_known(pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_scan_ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_map_ptr,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_scan_ptr){

    // =============== Initial guess =============== //
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> first_icp;
    float yaw, min_yaw, min_score = std::numeric_limits<float>::max();
    Eigen::Matrix4f min_pose(Eigen::Matrix4f::Identity());

    // Setting Known initial pose
    ROS_INFO("Start initial guess !");
    init_guess << 	cos(init_yaw), -sin(init_yaw), 	0, 	1715.72,
						        sin(init_yaw),  cos(init_yaw), 	0, 	1014.53,
							      0            ,  0            , 	1, 	-0.365639,
							      0            ,  0            , 	0,  1;

    ROS_INFO("Initial Set over");
    ROS_INFO("Initial Yaw : %f", init_yaw);
    ROS_INFO("---------------------");

    // set initial guess
    initialied = true;
  }

  /* Setting initial guess : unknown initial yaw, searching by icp */
  void initial_guess_icp(pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_scan_ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_map_ptr,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_scan_ptr){

    // =============== Initial guess =============== //
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> first_icp;
    float yaw, min_yaw, min_score = std::numeric_limits<float>::max();
    Eigen::Matrix4f min_pose(Eigen::Matrix4f::Identity());

    // Rotate to find best initial guess
    ROS_INFO("Start initial guess !");
    for(yaw = 0.0; yaw <= 2*M_PI; yaw += 0.1){ //2*M_PI
      /* Set initial guess of transformation */
      // target = R*source + t , R is rotation, t is translation
      Eigen::AngleAxisf init_rotation(yaw, Eigen::Vector3f::UnitZ()); // we only consider yaw rotate
      Eigen::Translation3f init_translation(gps_point.x, gps_point.y, gps_point.z);
      init_guess = (init_translation * init_rotation).matrix();
      ROS_INFO("Testing yaw : %f", yaw);

      /* Set the input source and target */
      first_icp.setInputSource(filtered_scan_ptr);
      first_icp.setInputTarget(filtered_map_ptr);
      first_icp.setMaxCorrespondenceDistance(2); // Set the max correspondence distance to ?m (e.g., correspondences with higher distances will be ignored)
      first_icp.setMaximumIterations(1000);      // Set the maximum number of iterations (criterion 1)
      first_icp.setTransformationEpsilon(1e-12);  // Set the transformation epsilon (criterion 2)
      first_icp.setEuclideanFitnessEpsilon(1e-8);// Set the euclidean distance difference epsilon (criterion 3)
      first_icp.align(*transformed_scan_ptr, init_guess); // Perform the alignment
      // Check Converged
      bool converged = first_icp.hasConverged();
      ROS_INFO("has converged: %s", converged ? "true" : "false");
      // Get the score
      double c_score = first_icp.getFitnessScore();
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
    ROS_INFO("Initial Yaw : %f", init_yaw);
    ROS_INFO("---------------------");

    // set initial guess
    init_guess = min_pose;
    initialied = true;
  }

  /* Show point cloud min and max value of x, y, z */
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
