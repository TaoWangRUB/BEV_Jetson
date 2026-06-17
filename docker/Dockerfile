# BEV / cuVSLAM dev image — Ubuntu 24.04 + ROS 2 Jazzy (aarch64 / x86_64)
#
# Mirrors the ackermann_rover container strategy: CUDA is NOT installed in the
# image. The host CUDA toolkit is bind-mounted at /usr/local/cuda at runtime and
# the NVIDIA container runtime injects the driver libs (see docker-compose.yml).
#
# IMPORTANT (Jetson TX2): the host CUDA on a TX2 is 10.2 (JetPack 4.6 ceiling).
# cuVSLAM v15 targets CUDA 12/13 with C++17 device code + cudaMallocAsync, so the
# cuVSLAM build is expected to FAIL on TX2. This image is still the correct base
# for VINS-Fisheye / D2SLAM, which build against CUDA 10.2.

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Make apt resilient to flaky mirrors on slow/Jetson links.
RUN printf '%s\n' \
    'Acquire::Retries "5";' \
    'Acquire::http::Timeout "30";' \
    'Acquire::https::Timeout "30";' \
    'Acquire::ForceIPv4 "true";' \
    > /etc/apt/apt.conf.d/99resilient-network

ARG ROS_DISTRO=jazzy
ARG ROS_UBUNTU_CODENAME=noble

RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PYTHONWARNINGS=ignore::DeprecationWarning

RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release build-essential git python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 apt repository (Jazzy on Ubuntu Noble)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${ROS_UBUNTU_CODENAME} main" \
    > /etc/apt/sources.list.d/ros2.list

ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}

# CUDA is mounted at runtime; bake only the env so tools find it once present.
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib

# ROS 2 base + the packages the cuVSLAM/VIO ROS wrapper needs.
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-rviz2 \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update || true

# cuVSLAM build dependencies (third_party/cuVSLAM). gcc-11 mirrors the rover's
# CUDA-11.4 host-compiler choice. NOTE: nvcc 10.2 (TX2) requires host gcc<=8, so
# this is NOT sufficient on TX2 — kept for parity with x86_64 / Orin.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build pkg-config \
    gcc-11 g++-11 \
    libeigen3-dev liblmdb-dev libgtest-dev \
    libopencv-dev python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host 'nvidia' (uid/gid 1000) for clean bind-mounts.
ARG USERNAME=nvidia
ARG USER_UID=1000
ARG USER_GID=1000
RUN set -eux; \
    apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*; \
    userdel -r ubuntu 2>/dev/null || true; \
    getent group video >/dev/null || groupadd --system video; \
    groupadd --gid "${USER_GID}" "${USERNAME}" 2>/dev/null || true; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" || \
      usermod --uid "${USER_UID}" --gid "${USER_GID}" "${USERNAME}"; \
    usermod -aG dialout,video "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

ENV HOME=/home/${USERNAME}
COPY --chown=${USER_UID}:${USER_GID} docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER ${USERNAME}
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
