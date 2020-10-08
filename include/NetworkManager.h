#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "ORTNetworks.h"

struct NetworkRecord;
class NetworkManager;

struct NetworkRecord{
    ortn::Network networkType;
    std::string path;
    ortn::ORT_Model* model;
};

class NetworkManager{
public:
    static NetworkManager* inst();

    /** @brief 用于向mRecords中添加神经网络。
     * 
     * @param networkType 神经网络类型(ortn::Network)。
     * @param path 神经网络onnx文件的路径(std::string)。
     */
    void addRecord(ortn::Network networkType, const std::string &path);
    
    void deleteRecord(int index);
    void clearRecords();
    
    void loadNetworks();

    /** @brief 用于将输入从[0, 255]的unsigned char转换为[0, 1]的float。
     * 
     * @param raw 待转换的数据(cv::Mat<unsigned char>)。
     * 
     * @return 转换后的数据(cv::Mat<float>)。
     */
    cv::Mat preprocess(const cv::Mat &raw);

    /** 用于按照mRecords中的顺序依次对输入进行运算，并返回结果。
     * 
     * @param input 输入矩阵(cv::Mat, RGB存储, W*H*C顺序, CV_32FC3)。
     * 
     * @return 运算结果(ortn::ORT_Result)。
     */
    ortn::ORT_Result compute(cv::Mat &input);

    cv::Mat infer(const cv::Mat &input);

    cv::Mat postprocess(const ortn::ORT_Result &res);

private:
    static NetworkManager *ptr;
    std::vector<NetworkRecord> mRecords;
    cv::Mat inputMat;

    void markObject(cv::Mat &src, const std::string &className, 
                    float conf, float x1, float y1, float x2, float y2);

    NetworkManager();
    ~NetworkManager();
};

#endif