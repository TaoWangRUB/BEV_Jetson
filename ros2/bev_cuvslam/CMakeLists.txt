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

# Jetson Argus targets (fused + panorama) need MMAPI + tegra libs. Host bag-replay
# only needs the modular multicam node. Default: ON when the MMAPI tree is present.
set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "jetson_multimedia_api root")
if(DEFINED ENV{BEV_BUILD_ARGUS})
  set(_bev_argus_default "$ENV{BEV_BUILD_ARGUS}")
elseif(EXISTS "${MMAPI_ROOT}/argus/include")
  set(_bev_argus_default ON)
else()
  set(_bev_argus_default OFF)
endif()
option(BEV_BUILD_ARGUS "Build Argus fused/panorama nodes (Jetson only)" ${_bev_argus_default})

if(BEV_BUILD_ARGUS)
  # Header-only shared timing helpers used by the fused node.
  find_package(bev_camera REQUIRED)
endif()

# --- locate cuVSLAM (TX2 build_tx2gpu OR host build_host) -----------------------
set(CUVSLAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cuVSLAM" CACHE PATH "cuVSLAM source root")
find_path(CUVSLAM_INCLUDE_DIR cuvslam2.h
  PATHS "${CUVSLAM_ROOT}/libs/cuvslam" REQUIRED)
find_library(CUVSLAM_LIB cuvslam
  PATHS
    "${CUVSLAM_ROOT}/build_host/bin"
    "${CUVSLAM_ROOT}/build_tx2gpu/bin"
    "${CUVSLAM_ROOT}/build/bin"
  REQUIRED)
message(STATUS "cuVSLAM header: ${CUVSLAM_INCLUDE_DIR}")
message(STATUS "cuVSLAM lib:    ${CUVSLAM_LIB}")
message(STATUS "BEV_BUILD_ARGUS: ${BEV_BUILD_ARGUS}")

add_executable(cuvslam_multicam_node src/cuvslam_multicam_node.cpp)
target_include_directories(cuvslam_multicam_node PRIVATE
  "${CUVSLAM_INCLUDE_DIR}/.." ${OpenCV_INCLUDE_DIRS} include)
target_link_libraries(cuvslam_multicam_node ${CUVSLAM_LIB} ${OpenCV_LIBS} yaml-cpp)
ament_target_dependencies(cuvslam_multicam_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport message_filters)

set(BEV_CUVSLAM_TARGETS cuvslam_multicam_node)

if(BEV_BUILD_ARGUS)
  # --- CUDA: nvcc .cu (g++-8/sm_62 on TX2; override via CMAKE_CUDA_* on newer kits) ---
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

  set(TEGRA_LIB_DIR "/usr/lib/aarch64-linux-gnu/tegra" CACHE PATH "tegra libs")
  set(CUDA_DIR "/usr/local/cuda" CACHE PATH "cuda toolkit")

  add_executable(bev_cuvslam_fused_node src/bev_cuvslam_fused_node.cpp src/virtual_pinhole_kernel.cu)
  set_target_properties(bev_cuvslam_fused_node PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
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
    bev_camera)

  add_executable(bev_panorama_node src/bev_panorama_node.cpp src/stitch_kernel.cu)
  set_target_properties(bev_panorama_node PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_include_directories(bev_panorama_node PRIVATE
    "${MMAPI_ROOT}/argus/include" "${MMAPI_ROOT}/include" "${CUDA_DIR}/include" ${OpenCV_INCLUDE_DIRS})
  target_link_directories(bev_panorama_node PRIVATE "${TEGRA_LIB_DIR}" "${CUDA_DIR}/lib64")
  target_link_libraries(bev_panorama_node
    nvargus_socketclient nveglstream_camconsumer nvbuf_utils EGL GLESv2
    cuda cudart yaml-cpp ${OpenCV_LIBS} pthread)
  ament_target_dependencies(bev_panorama_node rclcpp sensor_msgs)

  list(APPEND BEV_CUVSLAM_TARGETS bev_cuvslam_fused_node bev_panorama_node)
endif()

install(TARGETS ${BEV_CUVSLAM_TARGETS} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
