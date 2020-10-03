#ifndef CV_FUNCTIONS_H
#define CV_FUNCTIONS_H

#include <opencv2/opencv.hpp>

namespace cvf{
    cv::Mat readImage(std::string imgPath);
    void WHC2CWH(const cv::Mat &mat, unsigned char dest[]);
    std::vector<float> prepareImage(const cv::Mat &raw, const int width, const int height);

}

#endif