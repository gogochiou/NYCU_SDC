#include <string>
#include <pcl/point_types.h>
#include <ros/ros.h>
// #include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include "map_loader.h"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/publisher.h>


class MapPublisher{
    ros::NodeHandle nh;
    ros::Publisher pub_map;
    ros::Subscriber sub_pose;
    sensor_msgs::PointCloud2::Ptr map_cloud;
    ros::Timer timer;
    MapLoader<pcl::PointXYZI> loader;
    float search_radius = 200.;
    double init_x, init_y;
public:
    MapPublisher(ros::NodeHandle _nh, const std::string map_path)
        :map_cloud(new sensor_msgs::PointCloud2), loader(map_path)
    {
        std::string pose_topic, map_topic;
        this->nh = _nh;

        this->nh.param<float>("search_radius", search_radius, 200.);
        this->nh.param<double>("init_x", init_x, 0.0);
        this->nh.param<double>("init_y", init_y, 0.0);

        loader.setSearchRadius(search_radius);

        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/map", 4000000);
        sub_pose = nh.subscribe("/gps", 4000000, &MapPublisher::point_cb, this);
        // timer = nh.createTimer(ros::Duration(30.), &MapPublisher::timer_cb, this, false, false);

        ROS_INFO("%s initialized", ros::this_node::getName().c_str());
    }
    void point_cb(const geometry_msgs::PointStamped::ConstPtr& msg){
        //ROS_INFO("pose cb");
        //ROS_INFO("searching: %f,%f", msg->point.x, msg->point.y);
        pcl::PointXYZ center;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        center.x = msg->point.x;
        center.y = msg->point.y;
        center.z = 0;
        int status = loader.getSubmaps(center, cloud);
        if(status == STATUS::FAIL){
            ROS_ERROR("Loading submap fail");
        }else if(status == STATUS::SAME){
            ROS_INFO("Use same map");
        }else{ // status == STATUS::NEW
            ROS_INFO("New submap published at center = (%f, %f)", msg->point.x, msg->point.y);

            pcl::toROSMsg(*cloud, *map_cloud);
            ROS_INFO("Point cloud size: %d", map_cloud->width);
            map_cloud->header.stamp = ros::Time::now();
            map_cloud->header.frame_id = "world";
            pub_map.publish(*map_cloud);
        }
        // timer.start();
        return;
    }
    bool init_map(){
        std::cout << "Sub num : " << pub_map.getNumSubscribers() << std::endl;
        if(pub_map.getNumSubscribers() > 1){
            pcl::PointXYZ center;
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            center.x = init_x;
            center.y = init_y;
            center.z = 0;
            int status = loader.getSubmaps(center, cloud);
            if(status == STATUS::FAIL){
                ROS_ERROR("Loading submap fail");
            }else if(status == STATUS::SAME){
                ROS_INFO("Use same map");
            }else{ // status == STATUS::NEW
                ROS_INFO("New submap published at center = (%f, %f)", init_x, init_y);
                pcl::toROSMsg(*cloud, *map_cloud);
                ROS_INFO("Point cloud size: %d", map_cloud->width);
                map_cloud->header.stamp = ros::Time::now();
                map_cloud->header.frame_id = "world";
                pub_map.publish(*map_cloud);
            }
            ROS_INFO("Map setting over");
            return true;
        }
        else{
            return false;
        }

    }

    void timer_cb(const ros::TimerEvent& event){
        ROS_INFO("Timer triggered");
        map_cloud->header.stamp = ros::Time::now();
        ROS_INFO("Point cloud size: %d", map_cloud->width);
        if(map_cloud->width != 0){
            pub_map.publish(*map_cloud);
        }
    }
};

int main(int argc, char * argv[]){
    std::string map_path;
    ros::init(argc, argv, "map_publisher");
    ros::NodeHandle n("~");
    n.param<std::string>("map_path", map_path, "/root/catkin_ws/src/data/nuscenes_maps");
	ROS_INFO("Map Path: %s", map_path.c_str());

    MapPublisher publisher(n, map_path);
    
    ros::Rate loop(10);
    while(ros::ok()){
        ROS_INFO("IN loop");
        // if(publisher.init_map() == false)
        //     ros::spinOnce();

        // publisher.init_map();
        loop.sleep();
    }
    ros::spin();
    return 0;

}
