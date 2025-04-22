#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}
