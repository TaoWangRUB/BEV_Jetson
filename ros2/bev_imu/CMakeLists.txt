cmake_minimum_required(VERSION 3.16)
project(bev_imu)

find_package(ament_cmake REQUIRED)

# Python node: the timing-critical part (data-ready edge stamping) is delegated to the
# J106 project's reader, so there is one implementation of it rather than two.
install(PROGRAMS src/imu_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME imu_node)

ament_package()
