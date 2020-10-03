#include "../include/CVFunctions.h"

namespace cvf{
    cv::Mat readImage(std::string imgPath){
        cv::Mat img = cv::imread(imgPath, cv::ImreadModes::IMREAD_COLOR);
        
        /* 将其宽度缩放为4的倍数，以便QImage绘制(其存储空间按4对齐) */
        int width = img.cols;
        if (width % 4){
            cv::resize(img, img, cv::Size(((width/4)+1)*4, img.rows));
        }
        
        /* 从BGR转化为RGB */
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        return img;
    }
    void WHC2CWH(const cv::Mat &mat, unsigned char dest[]){
        int width = mat.cols;
        int height = mat.rows;
        unsigned char *pData = mat.data;
        for (int i=0; i<2; i++){
            for (int j=0; j<width; j++){
                for (int k=0; k<height; k++){
                    dest[i*width*height+k*width+j] = pData[k*3*width+j*3+i];
                }
            }
        }
        //memcpy(pData, dest, 3*width*height*sizeof(unsigned char));
    }
    std::vector<float> prepareImage(const cv::Mat &raw, const int width, const int height){
        std::vector<float> ret(width*height*3);
        cv::Mat resized;
        /* 缩放到指定尺寸 */
        cv::resize(raw, resized, cv::Size(width, height));

        /* 将W x H x C转换为C x W x H */
        for (int i=0; i<2; i++){
            for (int j=0; j<width; j++){
                for (int k=0; k<height; k++){
                    ret[i*width*height+k*width+j] = resized.data[k*3*width+j*3+i]/255.0f;
                }
            }
        }
        return ret;
    }
}