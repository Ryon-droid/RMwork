cmake_minimum_required(VERSION 3.8)
project(color_detection)

# 查找ROS2核心依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# 查找OpenCV
find_package(OpenCV REQUIRED)  

# 添加摄像头发布节点
add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node
  rclcpp         
  sensor_msgs     
  cv_bridge      
)
target_link_libraries(camera_node ${OpenCV_LIBS})  # 链接OpenCV库

# 添加图像处理节点
add_executable(image_processor src/image_processor.cpp)
ament_target_dependencies(image_processor
  rclcpp
  sensor_msgs
  cv_bridge
)
target_link_libraries(image_processor ${OpenCV_LIBS})

# 安装可执行文件
install(TARGETS
  camera_node
  image_processor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
