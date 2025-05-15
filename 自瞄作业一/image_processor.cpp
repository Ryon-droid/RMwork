#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

class ImageProcessor : public rclcpp::Node {
public:
    ImageProcessor() : Node("image_processor") {
        // 初始化HSV阈值
        blue_h_min = 100;   blue_h_max = 124;  // 蓝色H范围
        blue_s_min = 43;   blue_s_max = 255;  // 蓝色S范围
        blue_v_min = 46;   blue_v_max = 255;  // 蓝色V范围

        red_h_min1 = 0;    red_h_max1 = 10;   // 红色H区间1
        red_h_min2 = 160;  red_h_max2 = 180;  // 红色H区间2
        red_s_min = 80;    red_s_max = 255;   // 红色S范围
        red_v_min = 80;    red_v_max = 255;   // 红色V范围

        subscriber_ = create_subscription<sensor_msgs::msg::Image>(
            "image_raw", 10,
            std::bind(&ImageProcessor::process_image, this, std::placeholders::_1));
    }

private:
    // HSV阈值参数（类成员变量，方便直接修改）
    int blue_h_min, blue_h_max, blue_s_min, blue_s_max, blue_v_min, blue_v_max;
    int red_h_min1, red_h_max1, red_h_min2, red_h_max2, red_s_min, red_s_max, red_v_min, red_v_max;

    void process_image(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            //转换为HSV颜色空间
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            //蓝色色块HSV分割
            cv::Mat blue_mask;
            cv::inRange(hsv, 
                        cv::Scalar(blue_h_min, blue_s_min, blue_v_min),  // 蓝色HSV下限
                        cv::Scalar(blue_h_max, blue_s_max, blue_v_max),  // 蓝色HSV上限
                        blue_mask);

            //红色色块HSV分割
            cv::Mat red_mask1, red_mask2, red_mask;
            cv::inRange(hsv, 
                        cv::Scalar(red_h_min1, red_s_min, red_v_min),  // 红色区间1下限
                        cv::Scalar(red_h_max1, red_s_max, red_v_max),  // 红色区间1上限
                        red_mask1);
            cv::inRange(hsv, 
                        cv::Scalar(red_h_min2, red_s_min, red_v_min),  // 红色区间2下限
                        cv::Scalar(red_h_max2, red_s_max, red_v_max),  // 红色区间2上限
                        red_mask2);
            cv::bitwise_or(red_mask1, red_mask2, red_mask);  // 合并双区间掩码

            //形态学去噪
            cv::Mat blue_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, blue_kernel);  // 开运算去噪点
            cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, blue_kernel); // 闭运算填空洞

            cv::Mat red_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, red_kernel);
            cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, red_kernel);

            //显示结果
            cv::imshow("Blue Mask (HSV Only)", blue_mask);
            cv::imshow("Red Mask (HSV Only)", red_mask);
            cv::waitKey(1);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "图像转换失败: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProcessor>());
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}
