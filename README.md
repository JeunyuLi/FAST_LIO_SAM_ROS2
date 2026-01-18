# FAST_LIO_SAM
## Introduction
基于FAST_LIO2 ROS2版本的代码，增加了基于位置先验和ICP的回环检测和基于gtsam的图优化，目前适配了livox mid360 及velodyne雷达。


## Installation
1. 安装基础依赖
    ```bash
    sudo apt update
    sudo apt install -y \
        ros-humble-pcl-ros \
        ros-humble-pcl-conversions \
        ros-humble-tf2-eigen \
        ros-humble-cv-bridge \
        ros-humble-image-transport \
        ros-humble-compressed-image-transport
    # 安装gtsam
    sudo add-apt-repository ppa:borglab/gtsam-release-4.1
    sudo apt install libgtsam-dev libgtsam-unstable-dev
    ```
2. 安装 Livox SDK2 和驱动
    ```bash
    cd ~/fast_lio_sam_ros2_ws/src
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd Livox-SDK2 && mkdir build && cd build
    cmake .. && make -j$(nproc)
    sudo make install
    ```
3. 安装 Livox ROS2 驱动 
    ```bash
    cd ~/fast_lio_sam_ros2_ws/src
    git clone https://github.com/Ericsii/livox_ros_driver2.git
    # For ROS2 Humble:
    source /opt/ros/humble/setup.sh
    colcon build --symlink-install --packages-select livox_ros_driver2
    source install/setup.bash
    ```
4. 下载 Fast_LIO_SAM
   ```bash
   cd ~/fast_lio_sam_ros2_ws/src
   git clone https://github.com/JeunyuLi/FAST_LIO_SAM_ROS2.git
   ```
5. 编译
    ```bash
    cd ~/fast_lio_sam_ros2_ws
    rosdep install --from-paths src --ignore-src -y
    source install/setup.bash
    colcon build --symlink-install 
    source ./install/setup.bash
    ```
6. 运行
    ```bash
    source ./install/setup.bash
    ros2 launch fast_lio_sam mapping.launch.py config_file:=velodyne_16.yaml
    ros2 bag play T3F2-2021-08-02-15-00-12_ros2/
    ```

7. 对于mid360
    ```bash
    ros2 launch fast_lio_sam mapping.launch.py config_file:=mid360.yaml
    # 引入livox的消息类型
    source /home/slam/fast_lio_sam_ros2_ws/install/setup.bash
    ros2 bag play 
    ```

## ROS1 to ROS2
1. rosbag的转换
   ```bash
   pip install rosbags==0.10.10 # 目前0.11版本的有点问题
   rosbags-convert --src ros1bag.bag --dst ros2bag/
   ```


## Acknowledgment
感谢[FAST_LIO2](https://github.com/hku-mars/FAST_LIO2)，[FAST_LIO2_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2)，[FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM) 的工作
