#ifndef ORT_FUNCTIONS_H
#define ORT_FUNCTIONS_H

#include <fstream>
#include <onnxruntime_cxx_api.h>
#include "CVFunctions.h"

namespace ortf{
    struct Yolo_Candidate;
    class ONNX_RT;
    class ONNXRT_YOLOv4;

    struct Yolo_Candidate{
        int class_index;
        float conf;
        float point1_x;
        float point1_y;
        float point2_x;
        float point2_y;
    };

    class ONNX_RT{
    public:
        Ort::Session *session;

        ONNX_RT();
        ~ONNX_RT();

        void loadModel(const std::string &model_path);
        void printIOInfo();
        virtual void run(std::vector<float> &input_values) = 0;

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
    };

    class ONNXRT_YOLOv4: public ONNX_RT{
    public:
        virtual void run(std::vector<float> &input_values);
        void drawResults(cv::Mat &mat);
    private:
        std::vector<std::string> names;
        std::vector<Yolo_Candidate> result;
        void oneClassNMS(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, int start, int end, float threshold_nms, bool print);
        void nonMaximumSuppression(const std::vector<Yolo_Candidate> &candidate, std::vector<Yolo_Candidate> &res, float threshold_nms=0.6f);
        void postProcess(std::vector<Ort::Value> &arr, float threshold_conf=0.4f, float threshold_nms=0.6f);
        std::vector<std::string> loadNames(const std::string &path);
    };

    float intersectionOverUnion(float x11, float y11, float x12, float y12,
                                float x21, float y21, float x22, float y22);
}

#endif