cmake_minimum_required(VERSION 3.16)
project(bev_cuvslam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# --- locate the ported cuVSLAM (headers + the CUDA-10.2 build_tx2gpu .so) -------
# Override with -DCUVSLAM_ROOT=/path/to/third_party/cuVSLAM
set(CUVSLAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cuVSLAM" CACHE PATH "cuVSLAM source root")
find_path(CUVSLAM_INCLUDE_DIR cuvslam2.h
  PATHS "${CUVSLAM_ROOT}/libs/cuvslam" REQUIRED)
find_library(CUVSLAM_LIB cuvslam
  PATHS "${CUVSLAM_ROOT}/build_tx2gpu/bin" "${CUVSLAM_ROOT}/build/bin" REQUIRED)
message(STATUS "cuVSLAM header: ${CUVSLAM_INCLUDE_DIR}")
message(STATUS "cuVSLAM lib:    ${CUVSLAM_LIB}")

add_executable(cuvslam_multicam_node src/cuvslam_multicam_node.cpp)
target_include_directories(cuvslam_multicam_node PRIVATE
  "${CUVSLAM_INCLUDE_DIR}/.." ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuvslam_multicam_node ${CUVSLAM_LIB} ${OpenCV_LIBS} yaml-cpp)
ament_target_dependencies(cuvslam_multicam_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport message_filters)

install(TARGETS cuvslam_multicam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
