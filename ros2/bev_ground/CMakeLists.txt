cmake_minimum_required(VERSION 3.16)
project(bev_ground)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# No CUDA and no Argus here, on purpose. The node consumes ROS images that
# argus_capture_node has already put on the CPU, and the per-frame work is four
# cv::remap gathers into a 400x400 raster — under a megapixel of sampling in total. A
# CUDA round-trip would cost more in transfers than the arithmetic it saves. The
# panorama node's CUDA path exists because it captures via NVMM and never touches the
# CPU; that argument does not carry over. Revisit if the raster grows past ~1500x1500 or
# the optional depth stage lands.
add_executable(bev_ground_stitch_node src/bev_ground_stitch_node.cpp)
target_include_directories(bev_ground_stitch_node PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bev_ground_stitch_node ${OpenCV_LIBS} yaml-cpp)
ament_target_dependencies(bev_ground_stitch_node
  rclcpp sensor_msgs std_msgs geometry_msgs tf2 tf2_ros cv_bridge)

install(TARGETS bev_ground_stitch_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_package()
