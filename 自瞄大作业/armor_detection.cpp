#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

// 定义颜色类型
enum ColorType { RED, BLUE };

// 参数配置结构体
struct ArmorParam {
    // 灯条筛选参数
    float light_min_area = 85;         // 最小灯条面积
    float light_max_ratio = 8.0;        // 最大长宽比
    float light_contour_min_solidity = 0.4; // 最小凸度
    float light_color_detect_extend_ratio = 1.2; // 颜色检测扩展比例
    
    // 匹配参数
    float light_max_angle_diff_ = 18.0f;       // 最大角度差（度）
    float light_max_height_diff_ratio_ = 0.4f;  // 最大长度差比例
    float light_max_y_diff_ratio = 0.6f;       // 最大Y轴差比例
    float light_min_x_diff_ratio = 0.15f;       // 最小X轴差比例
    
    ColorType enemy_color = BLUE;                // 敌方灯带颜色
    
    // 二值化卷积核大小参数
    int morph_erode_size = 2;                   // 腐蚀操作卷积核大小
    int morph_dilate_size = 2;                  // 膨胀操作卷积核大小
};

// 灯条描述结构体
struct LightDescriptor {
    Point2f center;       // 中心坐标
    float angle;          // 旋转角度（度）
    float length;         // 长边长度
    float width;          // 短边长度
    RotatedRect rect;     // 旋转矩形
    
    // 构造函数
    LightDescriptor() = default;
    LightDescriptor(RotatedRect r) : rect(r) {
        center = r.center;
        angle = r.angle;
        length = max(r.size.width, r.size.height);
        width = min(r.size.width, r.size.height);
        // 角度矫正（确保长边垂直）
        if (angle < -45) {
            angle += 90;
            swap(length, width);
        }
    }
};

// 装甲板描述结构体
struct ArmorDescriptor {
    LightDescriptor left, right;  // 左右灯条
    RotatedRect boundingRect;     // 包围框（新增）
    float score;                  // 匹配得分
    vector<Point2f> cornerPoints; // 角点
    
    // 构造函数
    ArmorDescriptor() = default;
    ArmorDescriptor(LightDescriptor l, LightDescriptor r) 
        : left(l), right(r), score(0.0f) {
        // 计算包围框并存储
        vector<Point2f> allPoints;
        Point2f lPoints[4], rPoints[4];
        l.rect.points(lPoints);
        r.rect.points(rPoints);
        for (int i = 0; i < 4; i++) {
            allPoints.push_back(lPoints[i]);
            allPoints.push_back(rPoints[i]);
        }
        boundingRect = minAreaRect(allPoints);
    }

    // 模板匹配函数
    float templateMatch(Mat &src, vector<Mat> &templates) {
        float maxScore = 0.0f;
        for (auto &temp : templates) {
            Mat result;
            matchTemplate(src, temp, result, TM_CCOEFF_NORMED);
            double maxVal;
            minMaxLoc(result, nullptr, &maxVal);
            if (maxVal > maxScore) maxScore = maxVal;
        }
        return maxScore;
    }
};

// 卡尔曼滤波跟踪器类
class KalmanTracker {
private:
    KalmanFilter KF;
    Mat_<float> measurement;
    Point2f predictedPosition;
    bool initialized;
    
public:
    KalmanTracker() : KF(4, 2, 0), measurement(2, 1, CV_32F) {
        // 初始化状态转移矩阵
        KF.transitionMatrix = (Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        
        // 初始化测量矩阵
        KF.measurementMatrix = Mat::eye(2, 4, CV_32F);
        
        // 初始化过程噪声协方差
        setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
        
        // 初始化测量噪声协方差
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        
        // 初始化后验误差极方差
        setIdentity(KF.errorCovPost, Scalar::all(1));
        
        initialized = false;
        predictedPosition = Point2f();
    }

    // 初始化或更新跟踪器
    void initOrUpdate(const Point2f& position) {
        if (!initialized) {
            KF.statePost = (Mat_<float>(4, 1) << position.x, position.y, 0, 0);
            initialized = true;
        } else {
            // 更新测量值
            measurement(0) = position.x;
            measurement(1) = position.y;
            KF.correct(measurement);
        }
    }

    // 预测下一位置
    Point2f predict() {
        if (initialized) {
            Mat prediction = KF.predict();
            predictedPosition = Point2f(prediction.at<float>(0), prediction.at<float>(1));
        }
        return predictedPosition;
    }

    // 获取上一次预测的位置
    Point2f getLastPrediction() const {
        return predictedPosition;
    }

    // 重置跟踪器
    void reset() {
        initialized = false;
    }
    
    // 检查是否已初始化
    bool isInitialized() const {
        return initialized;
    }
};

// 装甲检测器类
class ArmorDetector {
private:
    ArmorParam _param;
    vector<Mat> _templates;    // 检测模板（根据颜色选择）
    vector<ArmorDescriptor> _armors;
    ArmorDescriptor _bestArmor;
    KalmanTracker _tracker;     // 卡尔曼跟踪器（基于装甲中心）
    float _scaleFactor; // 缩放因子
    Mat _binaryImage;   // 保存二值化图像结果

public:
    // 检测结果枚举
    enum DetectResult { NO_ARMOR = 0, FOUND_ARMOR = 1 };

    // 获取二值化参数的公共方法
    int getErodeSize() const { return _param.morph_erode_size; }
    int getDilateSize() const { return _param.morph_dilate_size; }

    // 初始化函数（加载对应颜色模板）
    bool init(string templatePath, ColorType color, float scaleFactor) {
        _param.enemy_color = color;
        _scaleFactor = scaleFactor;
        Mat temp = imread(templatePath, IMREAD_GRAYSCALE);
        if (temp.empty()) {
            cerr << "Template load failed: " << templatePath << endl;
            return false;
        }
        _templates.push_back(temp);
        return true;
    }

    // 获取二值化图像
    const Mat& getBinaryImage() const {
        return _binaryImage;
    }

    // 核心检测函数
    int detect(Mat &srcImg) {
        clock_t start = clock(); // 开始计时
        _armors.clear();
        _bestArmor = ArmorDescriptor();

        // 1. 图像预处理
        Mat binary = preprocessImage(srcImg);
        if (binary.empty()) return NO_ARMOR;

        // 保存二值化图像供显示
        _binaryImage = binary.clone();

        // 2. 灯条检测
        vector<LightDescriptor> lights = detectLightBars(binary);
        if (lights.size() < 2) {
            // 没有检测到目标时仅预测位置
            _tracker.predict();
            return NO_ARMOR;
        }

        // 3. 灯条匹配（会自动计算包围框）
        matchLightBars(lights);
        if (_armors.empty()) {
            _tracker.predict();
            return NO_ARMOR;
        }

        // 4. 模板筛选
        filterByTemplate(srcImg);
        if (_armors.empty()) {
            _tracker.predict();
            return NO_ARMOR;
        }

        // 5. 选择最佳目标
        selectBestArmor();

        // 6. 位置预测和跟踪（基于包围框中心）
        if (_bestArmor.score > 0) {
            predictPosition();
        } else {
            _tracker.predict();
            return NO_ARMOR;
        }

        clock_t end = clock(); // 结束计时
        double duration = double(end - start) / CLOCKS_PER_SEC * 1000; // 计算耗时
        cout << "Processing time: " << duration << " ms" << endl;

        return FOUND_ARMOR;
    }

    // 获取当前检测到的装甲板中心（蓝色包围框对角线交点）
    Point2f getTargetCenter() const {
        return _bestArmor.boundingRect.center;
    }
    
    // 获取预测的下一位置
    Point2f getPredictedPosition() const {
        return _tracker.getLastPrediction();
    }

    // 获取最佳装甲（用于外部绘制）
    const ArmorDescriptor& getBestArmor() const { return _bestArmor; }

private:
    // 图像预处理（优化形态学操作）
    Mat preprocessImage(Mat &src) {
        vector<Mat> channels;
        split(src, channels);
        Mat grayImg;

        // 颜色通道相减
        if (_param.enemy_color == RED) {
            subtract(channels[2], channels[0], grayImg);  // 红色灯带
        } else {
            subtract(channels[0], channels[2], grayImg);  // 蓝色灯带
        }

        // 阈值化和形态学处理
        Mat binary;
        threshold(grayImg, binary, _param.light_min_area, 255, THRESH_BINARY);
        
        // 使用配置的卷积核大小
        erode(binary, binary, getStructuringElement(MORPH_RECT, Size(_param.morph_erode_size, _param.morph_erode_size)));
        dilate(binary, binary, getStructuringElement(MORPH_RECT, Size(_param.morph_dilate_size, _param.morph_dilate_size)));
        
        return binary;
    }

    // 灯条检测
    vector<LightDescriptor> detectLightBars(Mat &binImg) {
        vector<vector<Point>> contours;
        findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<LightDescriptor> lights;
        for (auto &cnt : contours) {
            // 确保轮廓点数量足够
            if (cnt.size() < 5) continue;

            float area = contourArea(cnt);
            if (area < _param.light_min_area * 0.8) continue; // 放宽最小面积限制

            RotatedRect rect = fitEllipse(cnt);
            LightDescriptor light(rect);

            // 长宽比过滤
            float ratio = light.length / light.width;
            if (ratio > _param.light_max_ratio * 1.2) continue; // 放宽长宽比限制

            lights.push_back(light);
        }
        return lights;
    }

    // 灯条匹配（计算并存储包围框）
    void matchLightBars(vector<LightDescriptor> &lights) {
        // 按X坐标排序
        sort(lights.begin(), lights.end(),
            [](const LightDescriptor &a, const LightDescriptor &b) {
                return a.center.x < b.center.x;
            });

        // 两两匹配并计算包围框
        for (size_t i = 0; i < lights.size(); i++) {
            for (size_t j = i + 1; j < lights.size(); j++) {
                if (checkMatchCondition(lights[i], lights[j])) {
                    ArmorDescriptor armor(lights[i], lights[j]);
                    _armors.push_back(armor);
                }
            }
        }
    }

    // 匹配条件检查
    bool checkMatchCondition(const LightDescriptor &l, const LightDescriptor &r) {
        float angleDiff = abs(l.angle - r.angle);
        float lenDiff = abs(l.length - r.length) / max(l.length, r.length);
        float yDiffRatio = abs(l.center.y - r.center.y) / ((l.length + r.length) / 2);
        float xDiffRatio = abs(l.center.x - r.center.x) / ((l.length + r.length) / 2);

        return (angleDiff < _param.light_max_angle_diff_ * 1.5 && 
                lenDiff < _param.light_max_height_diff_ratio_ * 1.5 && 
                yDiffRatio < _param.light_max_y_diff_ratio * 1.5 && 
                xDiffRatio > _param.light_min_x_diff_ratio * 0.8);
    }

    // 模板筛选（限制搜索区域）
    void filterByTemplate(Mat &srcImg) {
        vector<ArmorDescriptor> validArmors;
        for (auto &armor : _armors) {
            Rect roi = getArmorROI(armor);
            if (roi.width <= 0 || roi.height <= 0) continue;

            Mat roiImg = srcImg(roi);
            cvtColor(roiImg, roiImg, COLOR_BGR2GRAY);
            float score = armor.templateMatch(roiImg, _templates);

            if (score > 0.4f) {  // 降低匹配阈值
                armor.score = score;
                validArmors.push_back(armor);
            }
        }
        _armors = validArmors;
    }

    // 获取装甲中间ROI
    Rect getArmorROI(const ArmorDescriptor &armor) const {
        Point2f center = armor.boundingRect.center; // 使用包围框中心
        float width = abs(armor.right.center.x - armor.left.center.x) * 0.7;
        float height = (armor.left.length + armor.right.length) / 2 * 0.5;
        return Rect(center.x - width/2, center.y - height/2, width, height);
    }

    // 选择最佳得分装甲
    void selectBestArmor() {
        if (_armors.empty()) return;
        sort(_armors.begin(), _armors.end(),
            [](const ArmorDescriptor &a, const ArmorDescriptor &b) {
                return a.score > b.score;
            });
        _bestArmor = _armors[0];
    }

    // 位置预测（基于装甲中心）
    void predictPosition() {
        // 使用包围框中心更新跟踪器
        Point2f armorCenter = _bestArmor.boundingRect.center;
        _tracker.initOrUpdate(armorCenter);
        _tracker.predict(); // 预测下一位置
    }
};

// 主函数
int main() {
    // 配置参数
    string videoPath = "armor_videoblue.mp4";       // 测试视频路径
    ColorType enemyColor = BLUE;                   // 敌方灯带颜色（RED/BLUE）

    // 动态设置模板路径
    string templatePath;
    float scaleFactor = 2.0f; 
    float rotationAngle = 0.6f;

    if (enemyColor == RED) {
        templatePath = "templates/red_armor2.png";
        scaleFactor = 1.2f;
        rotationAngle = 10.0f;
    } else if (enemyColor == BLUE) {
        templatePath = "templates/blue_armor2.png";
        scaleFactor = 1.2f;
        rotationAngle = 10.0f;
    } else {
        cerr << "Unknown enemy color" << endl;
        return -1;
    }

    // 初始化检测器
    ArmorDetector detector;
    if (!detector.init(templatePath, enemyColor, rotationAngle)) {
        cerr << "Initialization failed" << endl;
        return -1;
    }

    // 创建窗口
    namedWindow("Armor Detection", WINDOW_NORMAL);
    namedWindow("Binary Image", WINDOW_NORMAL);

    // 打开视频文件
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Unable to open video file: " << videoPath << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {        
        // 检测装甲板
        int result = detector.detect(frame);

        // 绘制预测的下一位置（基于装甲中心）
        Point2f predictedPos = detector.getPredictedPosition();
        if (predictedPos.x > 0 && predictedPos.y > 0 && predictedPos.x < frame.cols && predictedPos.y < frame.rows) {
            // 绘制预测位置标记（红色十字）
            line(frame, Point(predictedPos.x - 10, predictedPos.y), Point(predictedPos.x + 10, predictedPos.y), Scalar(0, 0, 255), 2);
            line(frame, Point(predictedPos.x, predictedPos.y - 10), Point(predictedPos.x, predictedPos.y + 10), Scalar(0, 0, 255), 2);
            putText(frame, "Predicted", Point(predictedPos.x + 15, predictedPos.y - 15), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        // 绘制检测结果
        if (result == ArmorDetector::FOUND_ARMOR) {
            const ArmorDescriptor& bestArmor = detector.getBestArmor();
            Point2f armorCenter = detector.getTargetCenter(); // 蓝色包围框中心（对角线交点）
            
            // 显示匹配得分
            putText(frame, "Score: " + to_string(bestArmor.score),
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

            // 绘制灯条矩形
            const LightDescriptor& leftLight = bestArmor.left;
            const LightDescriptor& rightLight = bestArmor.right;
            
            Point2f leftRectPoints[4];
            leftLight.rect.points(leftRectPoints);
            for (int j = 0; j < 4; j++) {
                line(frame, leftRectPoints[j], leftRectPoints[(j+1)%4], Scalar(0, 255, 0), 2);
            }
            
            Point2f rightRectPoints[4];
            rightLight.rect.points(rightRectPoints);
            for (int j = 0; j < 4; j++) {
                line(frame, rightRectPoints[j], rightRectPoints[(j+1)%4], Scalar(0, 255, 0), 2);
            }
            
            // 绘制蓝色包围框及其对角线
            Point2f boundingRectPoints[4];
            bestArmor.boundingRect.points(boundingRectPoints);
            for (int j = 0; j < 4; j++) {
                line(frame, boundingRectPoints[j], boundingRectPoints[(j+1)%4], Scalar(255, 0, 0), 2);
            }
            line(frame, boundingRectPoints[0], boundingRectPoints[2], Scalar(255, 0, 0), 2); // 对角线1
            line(frame, boundingRectPoints[1], boundingRectPoints[3], Scalar(255, 0, 0), 2); // 对角线2
            
            // 绘制装甲中心（绿色圆点，蓝色框对角线交点）
            circle(frame, armorCenter, 5, Scalar(0, 255, 0), -1);
            putText(frame, "Center", Point(armorCenter.x + 15, armorCenter.y - 15), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        // 显示画面标题
        putText(frame, "Armor Detection", Point(frame.cols/2 - 150, 30), 
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 200, 200), 2);

        // 显示二值化处理提示
        putText(frame, "Binary Processing: Erode=" + to_string(detector.getErodeSize()) + 
                " Dilate=" + to_string(detector.getDilateSize()),
                Point(10, frame.rows - 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        // 显示画面
        imshow("Armor Detection", frame);
        imshow("Binary Image", detector.getBinaryImage());

        // 按ESC退出
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
