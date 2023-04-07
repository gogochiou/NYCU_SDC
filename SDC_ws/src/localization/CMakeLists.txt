cmake_minimum_required(VERSION 3.0.2)
project(localization)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  tf2
  tf2_msgs
  tf_conversions
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES localization 
#  CATKIN_DEPENDS geometry_msgs pcl_ros roscpp rospy sensor_msgs tf2 tf2_msgs tf_conversions
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(localizer src/localizer_node.cpp)
target_link_libraries(localizer ${catkin_LIBRARIES})

add_executable(localizer_base src/localizer_node_base.cpp)
target_link_libraries(localizer_base ${catkin_LIBRARIES})

add_executable(localizer_ICP src/localizer_ICP_new.cpp)
target_link_libraries(localizer_ICP ${catkin_LIBRARIES})

add_executable(localizer_icp_1 src/localizer_icp_1.cpp)
target_link_libraries(localizer_icp_1 ${catkin_LIBRARIES})

add_executable(localizer_icp_2 src/localizer_icp_2.cpp)
target_link_libraries(localizer_icp_2 ${catkin_LIBRARIES})

add_executable(localizer_icp_2_imu src/localizer_icp_2_imu.cpp)
target_link_libraries(localizer_icp_2_imu ${catkin_LIBRARIES})

add_executable(localizer_icp_3 src/localizer_icp_3.cpp)
target_link_libraries(localizer_icp_3 ${catkin_LIBRARIES})

add_executable(localizer_icp_3_FakeKF src/localizer_icp_3_FakeKF.cpp)
target_link_libraries(localizer_icp_3_FakeKF ${catkin_LIBRARIES})

add_executable(CheckMsgPath src/CheckMsgPath.cpp)
target_link_libraries(CheckMsgPath ${catkin_LIBRARIES})

add_executable(pub_map src/pub_map_node.cpp)
target_link_libraries(pub_map ${catkin_LIBRARIES})

