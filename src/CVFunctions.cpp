#include "../include/CVFunctions.h"

namespace cvf{
    cv::Mat readImage(const std::string &imgPath){
        cv::Mat img = cv::imread(imgPath, cv::ImreadModes::IMREAD_COLOR);
        return img;
    }
    void WHC2CWH(const unsigned char source[], unsigned char dest[], const int width, const int height){
        for (int i=0; i<2; i++){
            for (int j=0; j<width; j++){
                for (int k=0; k<height; k++){
                    dest[i*width*height+k*width+j] = source[k*3*width+j*3+i];
                }
            }
        }
    }
    void WHC2CWH(const float source[], float dest[], const int width, const int height){
        for (int i=0; i<2; i++){
            for (int j=0; j<width; j++){
                for (int k=0; k<height; k++){
                    dest[i*width*height+k*width+j] = source[k*3*width+j*3+i];
                }
            }
        }
    }
    void CWH2WHC(const unsigned char source[], unsigned char dest[], const int width, const int height){
        for (int i=0; i<2; i++){
            for (int j=0; j<width; j++){
                for (int k=0; k<height; k++){
                    dest[k*3*width+j*3+i] = source[i*width*height+k*width+j];
                }
            }
        }
    }
    float intersectionOverUnion(float x11, float y11, float x12, float y12,
                                float x21, float y21, float x22, float y22){
        /* 左上角坐标 */
        float x1 = x11>x21?x11:x21;
        float y1 = y11>y21?y11:y21;

        /* 右下角坐标 */
        float x2 = x12<x22?x12:x22;
        float y2 = y12<y22?y12:y22;

        float w = x2>x1?x2-x1:0;
        float h = y2>y1?y2-y1:0;

        float inter_ = w * h;
        float union_ = (x12-x11)*(y12-y11)+(x22-x21)*(y22-y21)-inter_;
        return inter_/union_;
    }
}