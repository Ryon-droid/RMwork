#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node {
public:
    CameraNode() : Node("camera_node") {
        // 初始化摄像头
        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "摄像头初始化失败！设备号0不可用");
            return;
        } else {
            RCLCPP_INFO(this->get_logger(), "摄像头初始化成功（设备号0）");
        }

        // 创建图像发布者
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "image_raw", 10);

        // 创建定时器（20Hz，每50ms发布一次图像）
        timer_ = this->create_wall_timer(
            50ms,
            std::bind(&CameraNode::publish_image, this));
    }

private:
    void publish_image() {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "读取摄像头帧失败");
            return;
        }

        // 转换并发布图像消息
        auto msg = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->get_clock()->now();
        msg->header.frame_id = "camera_link";
        publisher_->publish(*msg);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    cv::VideoCapture cap_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();
    return 0;
}
