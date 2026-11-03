
cmake_minimum_required(VERSION 3.8)
project(bruce_msgs)
 
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
 
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(sonar_oculus REQUIRED)
 
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ISAM2Update.msg"
  # "msg/KeyframeImage.msg"   # renomeado: ROS 2 exige CamelCase
  # "msg/PoseHistory.msg"     # renomeado: ROS 2 exige CamelCase
  "srv/QueryExplorationPath.srv"
  "srv/PredictSLAMUpdate.srv"
  "srv/GetOccupancyMap.srv"
  DEPENDENCIES geometry_msgs nav_msgs sensor_msgs
)
 
ament_export_dependencies(rosidl_default_runtime)
ament_package()
 
