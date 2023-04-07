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
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include<Eigen/Dense>

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


bool init = false;
ros::Time t0;
ros::Time imu_lastTime;
ros::Time odom_lastTime;
float init_x;
float init_y;
float init_z;
float init_yaw;
geometry_msgs::PointStamped global_pose;

// for odom
geometry_msgs::PoseStamped odom_pose;
float odom_yaw;
float odom_x;
float odom_y;

// for odom
geometry_msgs::PoseStamped imu_pose;
float imu_yaw;



void gps_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
    if(!init){
        std::cout << "Init time : " << ros::Time::now() << std::endl;
        t0 = msg->header.stamp;
        imu_lastTime = t0;
        odom_lastTime = t0;
        init = true;
    }
    global_pose.point.x = msg->point.x;
    global_pose.point.y = msg->point.y;
    global_pose.point.z = msg->point.z;

    return;
}

void imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    std::cout << "get imu msg" << std::endl;
    ros::Duration imu_diff = msg->header.stamp - imu_lastTime;
    std::cout << "Imu duration : " << imu_diff.toSec() << std::endl;
    double dt = imu_diff.toSec();
    
    double ax = msg->linear_acceleration.x;
    double ay = msg->linear_acceleration.y;
    double vth = msg->angular_velocity.z;

    imu_yaw += (float)(vth*dt);
    double dx = ax*dt*dt*cos(imu_yaw) - ay*dt*dt*sin(imu_yaw);
    double dy = ax*dt*dt*sin(imu_yaw) + ay*dt*dt*cos(imu_yaw);
    // std::cout << "imu dx, dy : " << dx << ", " << dy << std::endl;
    imu_pose.pose.position.x += (float)dx;
    imu_pose.pose.position.y += (float)dy;
    geometry_msgs::Quaternion imu_quat = tf::createQuaternionMsgFromYaw(imu_yaw);
    imu_pose.pose.orientation = imu_quat;
    
    imu_lastTime = msg->header.stamp;
    return;
}

void odom_callback(const nav_msgs::Odometry::ConstPtr &msg){
    std::cout << "================================" << std::endl;
    std::cout << "get odom msg" << std::endl;
    ros::Duration odom_diff = msg->header.stamp - odom_lastTime;
    // std::cout << "Odom duration : " << odom_diff.toSec() << std::endl;
    double dt = odom_diff.toSec();
    
    /* Use velocity integral --> bettet way */
    double vx = msg->twist.twist.linear.x;
    double vy = msg->twist.twist.linear.y;
    double vth = msg->twist.twist.angular.z;

    odom_yaw += (float)(vth*dt);
    std::cout << "my yaw : " << imu_yaw << std::endl;
    double dx = vx*dt*cos(imu_yaw) - vy*dt*sin(imu_yaw);
    double dy = vx*dt*sin(imu_yaw) + vy*dt*cos(imu_yaw);
    std::cout << "dx, dy : " << dx << ", " << dy<< std::endl;
    odom_pose.pose.position.x += (float)dx;
    odom_pose.pose.position.y += (float)dy;
    std::cout << "Pose : " << odom_pose.pose.position.x << ", " << odom_pose.pose.position.y<< std::endl;
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(odom_yaw);
    odom_pose.pose.orientation = odom_quat;

    /* Use position diff to get next pose */
    // double odom_r, odom_p, odom_yaw;
    // tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    // tf::Matrix3x3 temp(q);
    // temp.getRPY(odom_r, odom_p, odom_yaw);
    // std::cout<< "odom oreintation : " << odom_r << std::endl;
    // float global_odom_x = msg->pose.pose.position.x*cos(imu_yaw) - msg->pose.pose.position.y*sin(imu_yaw);
    // float global_odom_y = msg->pose.pose.position.x*sin(imu_yaw) + msg->pose.pose.position.y*cos(imu_yaw);
    // float diff_x = global_odom_x - odom_x;
    // float diff_y = global_odom_y - odom_y;
    // odom_x = global_odom_x;
    // odom_y = global_odom_y;

    // odom_pose.pose.position.x += diff_x;
    // odom_pose.pose.position.y += diff_y;
    // odom_pose.pose.orientation = msg->pose.pose.orientation;
    
    odom_lastTime = msg->header.stamp;
    std::cout << "================================" << std::endl;
    return;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "localizer");
    ros::NodeHandle n("~");

    ros::Subscriber gps_sub = n.subscribe("/gps", 400000, &gps_callback);
    ros::Subscriber imu_sub = n.subscribe("/imu/data", 400000, &imu_callback);
    ros::Subscriber odom_sub = n.subscribe("/wheel_odometry", 4000000, &odom_callback);

    ros::Publisher odom_pub = n.advertise<geometry_msgs::PoseStamped>("/odom_pose", 10);
    ros::Publisher imu_pub = n.advertise<geometry_msgs::PoseStamped>("/imu_pose", 10);
    ros::Publisher gps_pub = n.advertise<geometry_msgs::PointStamped>("/gps_pose", 10);
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2>("/map", 1);

    std::string map_path;
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_points;

    n.param<float>("init_x", init_x, 1715.72);
    n.param<float>("init_y", init_y, 1014.53);
    n.param<float>("init_yaw", init_yaw, 2.544);
    n.param<std::string>("map_path", map_path, "nuscenes_map.pcd");

    // Map Pub
    map_points = (new pcl::PointCloud<pcl::PointXYZI>)->makeShared();
	if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_path, *map_points) == -1){                                                                           
	    PCL_ERROR("Couldn't read that pcd file\n");                         
		exit(0);
	}
    else{
      ROS_INFO("Got map ~~");
      sensor_msgs::PointCloud2::Ptr map_cloud(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*map_points, *map_cloud);
      map_cloud->header.frame_id = "map";
      map_pub.publish(*map_cloud);
      ROS_INFO("Pub Map");
    }

    bool init = false;
    init_z = 0;
    odom_x = 0.0;
    odom_y = 0.0;

    // initialize
    // pub odom
    geometry_msgs::Quaternion quat = tf::createQuaternionMsgFromYaw(init_yaw);
    odom_pose.header.stamp = ros::Time::now();
    odom_pose.header.frame_id = "map";
    odom_yaw = init_yaw;
    odom_pose.pose.position.x = init_x;
    odom_pose.pose.position.y = init_y;
    odom_pose.pose.position.z = init_z;
    odom_pose.pose.orientation = quat;
    odom_pub.publish(odom_pose);
    // pub imu
    imu_pose.header.stamp = ros::Time::now();
    imu_pose.header.frame_id = "map";
    imu_yaw = init_yaw;
    imu_pose.pose.position.x = init_x;
    imu_pose.pose.position.y = init_y;
    imu_pose.pose.position.z = init_z;
    imu_pose.pose.orientation = quat;
    imu_pub.publish(imu_pose);
    // gps
    global_pose.header.stamp = ros::Time::now();
    global_pose.header.frame_id = "map";
    
    
    ros::Rate loop(100);
    while(ros::ok()){

        ros::spinOnce();

        odom_pose.header.stamp = ros::Time::now();
        odom_pub.publish(odom_pose);

        imu_pose.header.stamp = ros::Time::now();
        imu_pub.publish(imu_pose);

        global_pose.header.stamp = ros::Time::now();
        gps_pub.publish(global_pose);

        std::cout << "-------------------" << std::endl;
        loop.sleep();
    }

    return 0;
}

/* Note */
/*
1. wheel_odom twist is 


*/
