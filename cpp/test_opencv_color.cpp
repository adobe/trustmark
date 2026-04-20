#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("../images/ufo_240.jpg");

    cv::Vec3b pixel = img.at<cv::Vec3b>(0, 0);

    std::cout << "OpenCV first pixel:" << std::endl;
    std::cout << "  Channel 0: " << (int)pixel[0] << std::endl;
    std::cout << "  Channel 1: " << (int)pixel[1] << std::endl;
    std::cout << "  Channel 2: " << (int)pixel[2] << std::endl;
    std::cout << "\nOpenCV uses BGR format by default" << std::endl;

    return 0;
}
