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

        void loadModel(const std::string &model_path);
        void printIOInfo();
        virtual ORT_Result run(cv::Mat &input_values) = 0;
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

        virtual Ort::Value prepareInput(const cv::Mat &input_values) = 0;
    };

    class ORT_YOLOv4: public ORT_Model{
    public:
        ORT_YOLOv4();
        virtual ORT_Result run(cv::Mat &input_values);
    private:
        float threshold_conf;
        float threshold_nms;
        std::vector<std::string> names;
        std::vector<Yolo_Candidate> result;
        cv::Mat raw;
        cv::Mat temp;

        /** @brief 用于对输入图片进行处理，生成模型的输入(矩阵)。
         * 处理包括缩放、存储顺序从W*H*C转换为C*W*H。
         * 
         * @param input_values 输入图片(cv::Mat, W*H*C)。
         * 
         * @return 模型的输入(Ort::Value, C*W*H)。
         */
        virtual Ort::Value prepareInput(const cv::Mat &input_values);
        void oneClassNMS(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, int start, int end, float threshold_nms, bool print);
        void nonMaximumSuppression(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, float threshold_nms=0.6f);

        ORT_Result parseResults(std::vector<Ort::Value> &arr);
        void postProcess(std::vector<Ort::Value> &arr, float threshold_conf=0.4f, float threshold_nms=0.6f);
        std::vector<std::string> loadNames(const std::string &path);
    };
}

#endif