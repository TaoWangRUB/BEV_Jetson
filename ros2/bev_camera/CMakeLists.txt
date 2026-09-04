cmake_minimum_required(VERSION 3.16)
project(bev_camera)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
# yaml-cpp: the node cross-checks the loaded calibration against the live rig.
find_package(yaml-cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Per-frame timing metadata (frame number, capture id, SOF, exposure) - see docs/timestamps.md.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrameMeta.msg"
  DEPENDENCIES std_msgs)

# Jetson Multimedia API (Argus, EGLStream, nvbuf_utils). Bind-mount the host's
# /usr/src/jetson_multimedia_api into the build container and point here:
#   -DMMAPI_ROOT=/opt/mmapi
set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "jetson_multimedia_api root")
# tegra libs (libnvargus_socketclient, libnveglstream_camconsumer, libnvbuf_utils)
# are mounted by the nvidia runtime here:
set(TEGRA_LIB_DIR "/usr/lib/aarch64-linux-gnu/tegra" CACHE PATH "tegra libs")

add_executable(argus_capture_node src/argus_capture_node.cpp)
target_include_directories(argus_capture_node PRIVATE
  "${MMAPI_ROOT}/argus/include"
  "${MMAPI_ROOT}/include"
  include)
target_link_directories(argus_capture_node PRIVATE "${TEGRA_LIB_DIR}")
target_link_libraries(argus_capture_node
  nvargus_socketclient nveglstream_camconsumer nvbuf_utils
  yaml-cpp EGL GLESv2 pthread)
ament_target_dependencies(argus_capture_node rclcpp sensor_msgs std_msgs)
# link the node against the message types generated above, in this same package
rosidl_target_interfaces(argus_capture_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS argus_capture_node DESTINATION lib/${PROJECT_NAME})
# The port map and the frame-timing rule are shared with bev_cuvslam's fused node: one
# definition, so the two cannot drift apart.
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_package()
