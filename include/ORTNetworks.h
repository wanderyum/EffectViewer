#ifndef ORT_FUNCTIONS_H
#define ORT_FUNCTIONS_H

#include <fstream>
#include <core/session/onnxruntime_cxx_api.h>
#include "CVFunctions.h"

namespace ortn{
    enum struct ORT_Network;
    enum struct ResultType;
    enum struct ResultFormat;
    enum struct MissionType;
    struct ORT_Result;
    struct Yolo_Candidate;
    class ORT_Model;
    class ORT_YOLOv4;

    
    enum struct Network{
        YOLO_V4
    };

    enum struct ResultType{
        FINAL,          /* 最终结果(其后不能级联别的网络) */
        INTERMEDIATE,   /* 中间结果(其后可以级联别的网络) */
        NO_RESULT       /* 没有可用结果 */
    };

    enum struct ResultFormat{
        CV_MAT,         /* 运算结果格式为cv::Mat */
        ORT_VALUE       /* 运算结果格式为Ort::Value */
    };

    enum struct MissionType{
        CLASSIFICATION,     /* 目标分类 */
        OBJECT_DETECTION    /* 目标检测 */
    };

    /** @struct ORT_Result
     * @brief 这个结构体用来表示一个神经网络的计算结果。
     * 
     * @param resType 结果类型(ResultType枚举)。
     * @param resForm 结果格式(ResultFormat枚举)。
     * @param misType 任务类型(MissionType枚举)。
     * @param matData 运算结果(cv::Mat)，如果结果格式不是CV_MAT则为空。
     * @param ortData 运算结果(Ort::Value)，如果结果格式不是ORT_VALUE则为空。
     * @param stringVector 类别名称等其他附加文字信息(std::vector<std::string>)。
     */
    struct ORT_Result{
        ResultType resType;
        ResultFormat resForm;
        MissionType misType;
        cv::Mat matData;
        Ort::Value ortData;
        std::vector<std::string> stringVector;
    };

    struct Yolo_Candidate{
        int class_index;
        float conf;
        float point1_x;
        float point1_y;
        float point2_x;
        float point2_y;
    };

    class ORT_Model{
    public:
        Ort::Session *session;

        ORT_Model();
        ~ORT_Model();

        /** @brief 用于从指定路径读取模型。
         * 
         * @param model_path 模型的路径(std::string)。
         */
        virtual void loadModel(const std::string &model_path);

        /** @brief 用于打印模型的输入输出信息(输入/输出数目、维数等)。
         */
        void printIOInfo();

        /** @brief 用于通过神经网络根据输入进行运算，并返回计算结果。
         * 
         * @param input_values 输入数据(cv::Mat)。
         * 
         * @return 计算结果(ortn::ORT_Result)。
         */
        virtual ORT_Result run(cv::Mat &input_values) = 0;

        /** @brief 用于判断该模型是否已经被加载。
         * 
         * @return 模型是否已经被加载(bool)。
         */
        bool isLoaded();

    protected:
        Ort::Env env;
        Ort::SessionOptions session_options;
        /* 模型信息 */
        size_t num_input_nodes;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t> > input_nodes_dims;
        size_t num_output_nodes;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t> > output_nodes_dims;
        bool loaded;
        bool channel_first;

        /** @brief 用于对输入进行预处理。
         * 
         * @param input_values 输入图片(cv::Mat, RGB, W*H*C, CV_8UC3)。
         * 
         * @return 模型的输入(Ort::Value, C*W*H)。
         */
        virtual Ort::Value prepareInput(const cv::Mat &input_values) = 0;
    };

    class ORT_YOLOv4: public ORT_Model{
    public:
        ORT_YOLOv4();

        /** @brief 用于通过神经网络根据输入进行运算，并返回计算结果。
         * 
         * @param input_values 输入数据(cv::Mat)。
         * 
         * @return 计算结果(ortn::ORT_Result)。
         */
        virtual ORT_Result run(cv::Mat &input_values);
    private:
        float threshold_conf;
        float threshold_nms;
        std::vector<std::string> names;
        std::vector<Yolo_Candidate> result;
        cv::Mat raw;
        cv::Mat temp;

        /** @brief 用于对输入图片进行处理，生成模型的输入(矩阵)。
         * 
         * @detail 处理具体包括resize、从W*H*C转换为C*W*H、
         * 从[0, 255]的unsigned char转换到[0, 1]的float, 
         * 以及创建对应的Ort::Value。
         * 
         * @param input_values 输入图片(cv::Mat, RGB, W*H*C, CV_8UC3)。
         * 
         * @return 模型的输入(Ort::Value, C*W*H)。
         */
        virtual Ort::Value prepareInput(const cv::Mat &input_values);
        void oneClassNMS(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, int start, int end, float threshold_nms, bool print);
        void nonMaximumSuppression(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, float threshold_nms=0.6f);

        /** @brief 用于处理原始计算结果。
         *
         * @param arr 原始计算结果(std::vector<Ort::Value>)。
         * 
         * @return 处理后的计算结果(ortn::ORT_Result)。
         */
        ORT_Result parseResults(std::vector<Ort::Value> &arr);
        void postProcess(std::vector<Ort::Value> &arr, float threshold_conf=0.4f, float threshold_nms=0.6f);
        std::vector<std::string> loadNames(const std::string &path);
    };
}

#endif