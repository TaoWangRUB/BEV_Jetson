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
# Header-only, for the fused node: bev_camera/argus_timing.hpp carries the Argus port
# table and the exposure-midpoint rule, shared with argus_capture_node (4.7).
find_package(bev_camera REQUIRED)

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
  "${CUVSLAM_INCLUDE_DIR}/.." ${OpenCV_INCLUDE_DIRS} include)
target_link_libraries(cuvslam_multicam_node ${CUVSLAM_LIB} ${OpenCV_LIBS} yaml-cpp)
ament_target_dependencies(cuvslam_multicam_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport message_filters)

# --- CUDA: nvcc .cu (g++-8/sm_62) compiles inside colcon gcc-9 ---
# nvcc 10.2 needs host gcc<=8; pin nvcc + g++-8 + sm_62 so the default colcon build just works.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER g++-8)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 62)
endif()
enable_language(CUDA)

# --- fused zero-copy Argus -> cuVSLAM node (NVMM Y plane -> CUDA, no CPU round-trip) ---
# Needs the Jetson Multimedia API (Argus/EGLStream/nvbuf_utils, bind-mounted) + CUDA
# (driver cuGraphicsEGLRegisterImage + runtime). tegra libs mounted by the nvidia runtime.
set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "jetson_multimedia_api root")
set(TEGRA_LIB_DIR "/usr/lib/aarch64-linux-gnu/tegra" CACHE PATH "tegra libs")
set(CUDA_DIR "/usr/local/cuda" CACHE PATH "cuda toolkit")

add_executable(bev_cuvslam_fused_node src/bev_cuvslam_fused_node.cpp src/virtual_pinhole_kernel.cu)
set_target_properties(bev_cuvslam_fused_node PROPERTIES CUDA_ARCHITECTURES 62)
target_include_directories(bev_cuvslam_fused_node PRIVATE
  "${CUVSLAM_INCLUDE_DIR}/.."
  "${MMAPI_ROOT}/argus/include" "${MMAPI_ROOT}/include"
  "${CUDA_DIR}/include" ${OpenCV_INCLUDE_DIRS} include)
target_link_directories(bev_cuvslam_fused_node PRIVATE "${TEGRA_LIB_DIR}" "${CUDA_DIR}/lib64")
target_link_libraries(bev_cuvslam_fused_node
  ${CUVSLAM_LIB} yaml-cpp ${OpenCV_LIBS}
  nvargus_socketclient nveglstream_camconsumer nvbuf_utils EGL GLESv2
  cuda cudart pthread)
ament_target_dependencies(bev_cuvslam_fused_node
  rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  # bev_camera/argus_timing.hpp: the port table and the exposure-midpoint rule, shared with
  # argus_capture_node so the capture path and the VO path cannot disagree about which
  # image is cam1 or what instant it depicts (4.7).
  bev_camera)

# --- surround panorama: capture (NVMM->CUDA) + custom equirect stitch kernel -> /bev/panorama ---
add_executable(bev_panorama_node src/bev_panorama_node.cpp src/stitch_kernel.cu)
set_target_properties(bev_panorama_node PROPERTIES CUDA_ARCHITECTURES 62)
target_include_directories(bev_panorama_node PRIVATE
  "${MMAPI_ROOT}/argus/include" "${MMAPI_ROOT}/include" "${CUDA_DIR}/include" ${OpenCV_INCLUDE_DIRS})
target_link_directories(bev_panorama_node PRIVATE "${TEGRA_LIB_DIR}" "${CUDA_DIR}/lib64")
target_link_libraries(bev_panorama_node
  nvargus_socketclient nveglstream_camconsumer nvbuf_utils EGL GLESv2
  cuda cudart yaml-cpp ${OpenCV_LIBS} pthread)
ament_target_dependencies(bev_panorama_node rclcpp sensor_msgs)

install(TARGETS cuvslam_multicam_node bev_cuvslam_fused_node bev_panorama_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
