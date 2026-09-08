cmake_minimum_required(VERSION 3.16)
project(bev_range)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# termios directly rather than a serial library: the console protocol is line-based ASCII
# and the only requirement is that this process is the SOLE owner of the port, which is a
# question of who opens it, not of which abstraction reads it.
add_executable(range_node src/range_node.cpp)
ament_target_dependencies(range_node rclcpp sensor_msgs)

install(TARGETS range_node DESTINATION lib/${PROJECT_NAME})
ament_package()
