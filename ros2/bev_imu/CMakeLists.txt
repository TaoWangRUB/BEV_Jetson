cmake_minimum_required(VERSION 3.16)
project(bev_imu)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# spidev + the GPIO character device are used directly: the sample loop must not do
# anything that can stall between the data-ready edge and the timestamp.
add_executable(imu_node src/imu_node.cpp)
target_link_libraries(imu_node pthread)
ament_target_dependencies(imu_node rclcpp sensor_msgs)

install(TARGETS imu_node DESTINATION lib/${PROJECT_NAME})
ament_package()
