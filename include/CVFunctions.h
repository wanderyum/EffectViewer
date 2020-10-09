#ifndef CV_FUNCTIONS_H
#define CV_FUNCTIONS_H

#include <opencv2/opencv.hpp>

namespace cvf{

    /** @brief 用于从指定路径读取一张图片。
     * 图片格式支持jpg、png等。
     * 
     * @param imgPath 图片路径(std::string)。
     * 
     * @return 3通道矩阵(cv::Mat，BGR存储)。
     */
    cv::Mat readImage(const std::string &imgPath);

    /** @brief 用于将输入(unsigned char[])的存储格式从W*H*C转换为C*H*W。
     * 
     * 这里假设待转换的数据具有3个通道。
     * 
     * @param source 待转换的数据(unsigned char[])。
     * @param dest 转换后的数据((unsigned char[]))。
     */
    void WHC2CWH(const unsigned char source[], unsigned char dest[], const int width, const int height);
    
    /** @brief 用于将输入(unsigned char[])的存储格式从C*H*W转换为W*H*C。
     * 
     * 这里假设待转换的数据具有3个通道。
     * 
     * @param source 待转换的数据(unsigned char[])。
     * @param dest 转换后的数据((unsigned char[]))。
     */
    void CWH2WHC(const unsigned char source[], unsigned char dest[], const int width, const int height);

    /** @brief 用于计算两个矩形之间的交并比IOU。
     * 
     * @param x11 矩形1左上顶点X坐标(float)。
     * @param y11 矩形1左上顶点Y坐标(float)。
     * @param x11 矩形1右下顶点X坐标(float)。
     * @param y11 矩形1右下顶点Y坐标(float)。
     * @param x11 矩形2左上顶点X坐标(float)。
     * @param y11 矩形2左上顶点Y坐标(float)。
     * @param x11 矩形2右下顶点X坐标(float)。
     * @param y11 矩形2右下顶点Y坐标(float)。
     * 
     * @return 交并比IOU(float)。
     */
    float intersectionOverUnion(float x11, float y11, float x12, float y12,
                                float x21, float y21, float x22, float y22);

}

#endif