cmake_minimum_required(VERSION 3.8)
project(bruce_slam)

# PCL 1.14 (Ubuntu 24.04) exige pelo menos C++14; o ROS 2 Jazzy usa C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # equivale ao -O3 do original
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED)
# find_package(rti_dvl REQUIRED)        # comentados no original
# find_package(sonar_oculus REQUIRED)
# find_package(bar30_depth REQUIRED)

include_directories(
  ${libpointmatcher_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Substitui catkin_python_setup(): instala src/bruce_slam como pacote Python
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# Módulos C++ expostos ao Python (viram bruce_slam.pcl e bruce_slam.cfar)
pybind11_add_module(pcl src/${PROJECT_NAME}/cpp/pcl.cpp)
target_link_libraries(pcl PRIVATE ${libpointmatcher_LIBRARIES} ${PCL_LIBRARIES})

pybind11_add_module(cfar src/${PROJECT_NAME}/cpp/cfar.cpp)

install(TARGETS pcl cfar
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

ament_package()