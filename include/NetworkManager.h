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

/** @class NetworkManager
 * @brief 该类用于管理神经网络，并且同时只能存在一个实例。
 * 可以通过NetworkManager::inst()来访问该实例的指针。
 */
class NetworkManager{
public:
    /** @brief 用于得到全局唯一的NetworkManager实例。
     * 
     * @return 全局唯一的NetworkManager实例的指针。
     */
    static NetworkManager* inst();

    /** @brief 用于向mRecords中添加神经网络。
     * 
     * @param networkType 神经网络类型(ortn::Network)。
     * @param path 神经网络onnx文件的路径(std::string)。
     */
    void addRecord(ortn::Network networkType, const std::string &path);
    
    /** @brief 用于移除mRecord中指定序号的神经网络。
     * 
     * @param index 待移除神经网络的序号(int)。
     */
    void deleteRecord(int index);

    /** @brief 用于清空mRecord中的神经网络。
     */
    void clearRecords();
    
    /** @brief 用于读取mRecord中的所有神经网络。
     */
    void loadNetworks();

    /** @brief 用于对图片进行预处理。目前仅对输入做一个备份。
     * 
     * @param raw 待转换的数据(cv::Mat<unsigned char>)。
     * 
     * @return 转换后的数据(cv::Mat<unsigned char>)。
     */
    cv::Mat preprocess(const cv::Mat &raw);

    /** @brief 用于按照mRecords中的顺序依次对输入进行运算，并返回结果。
     * 
     * @param input 输入矩阵(cv::Mat, RGB存储, W*H*C顺序, CV_8UC3)。
     * 
     * @return 运算结果(ortn::ORT_Result)。
     */
    ortn::ORT_Result compute(cv::Mat &input);

    /** @brief 用于对得到的结果进行处理，使其方便被展示。
     * 
     * @remarks 对于目标检测任务，返回的是含有标记框的图像。
     * 
     * @param res 经过神经网络运算得到的结果(ortn::ORT_Result)。
     * 
     * @return 处理后的结果(cv::Mat)。 
     */
    cv::Mat postprocess(const ortn::ORT_Result &res);

    /** @brief 用于用神经网络处理输入图片并返回最终处理过的结果。
     * 
     * @remarks 对于目标检测任务，返回的是含有标记框的图像。
     * 
     * @param input 输入图片(cv::Mat)。
     * 
     * @return  最终处理过的结果(cv::Mat)。
     */
    cv::Mat infer(const cv::Mat &input);
private:
    static NetworkManager *ptr;
    std::vector<NetworkRecord> mRecords;
    cv::Mat inputMat;

    /** @brief 用于在目标检测任务中标记检测到的单个目标。
     * 
     * @param src 待标记的源图片文件(cv::Mat)。
     * @param className 检测到目标的所属类别(std::string)。
     * @param conf 检测的置信度(float)。
     * @param x1 标记框左上角的X坐标([0, 1]的float)。
     * @param y1 标记框左上角的Y坐标([0, 1]的float)。
     * @param x2 标记框右下角的X坐标([0, 1]的float)。
     * @param y2 标记框右下角的Y坐标([0, 1]的float)。
     */
    void markObject(cv::Mat &src, const std::string &className, 
                    float conf, float x1, float y1, float x2, float y2);

    NetworkManager();
    ~NetworkManager();
};

#endif