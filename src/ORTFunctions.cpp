#include "../include/ORTFunctions.h"

namespace ortf{
    ONNX_RT::ONNX_RT():
        session(nullptr),
        env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_RT") {
        /* 初始化session选项 */
        session_options.SetIntraOpNumThreads(1);

        /* 设置优化级别，可选项：
        ORT_DISABLE_ALL -> 关闭所有优化
        ORT_ENABLE_BASIC -> 开启基本优化 (比如redundant node removals)
        ORT_ENABLE_EXTENDED -> 开启进一步优化 (包含基本优化外加node fusions等更复杂的优化)
        ORT_ENABLE_ALL -> 开启所有可用优化 */
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
    ONNX_RT::~ONNX_RT(){
        if (session){
            delete session;
        }
    }

    void ONNX_RT::loadModel(const std::string &model_path){
        #ifdef _WIN32
        std::wstring w_path = std::wstring(model_path.begin(), model_path.end());
        session = new Ort::Session(env, w_path.c_str(), session_options);
        #else
        session = new Ort::Session(env, model_path.c_str(), session_options);
        #endif

        Ort::AllocatorWithDefaultOptions allocator;
        /* 获取模型输入信息 */
        num_input_nodes = session->GetInputCount();
        for (int i=0; i<num_input_nodes; i++) {
            input_node_names.push_back(session->GetInputName(i, allocator));

            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_nodes_dims.push_back(tensor_info.GetShape());
        }

        /* 获取模型输入信息 */
        num_output_nodes = session->GetOutputCount();
        for (int i=0; i<num_output_nodes; i++) {
            output_node_names.push_back(session->GetOutputName(i, allocator));

            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_nodes_dims.push_back(tensor_info.GetShape());
        }
    }
    void ONNX_RT::printIOInfo(){

        /* 打印输入信息 */
        printf("Number of inputs = %zu\n", num_input_nodes);
        /* 遍历所有输入结点 */
        for (int i = 0; i < num_input_nodes; i++) {
            /* 打印输入节点名称 */
            printf("  Input %d : name=%s\n", i, input_node_names[i]);

            /* 打印输入结点类型 */
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("  Input %d : type=%d\n", i, type);

            /* 打印输入的维度 */
            printf("  Input %d : dimension=(%d", i, input_nodes_dims[i][0]);
            for (int j = 1; j < input_nodes_dims[i].size(); j++)
                printf(", %d", input_nodes_dims[i][j]);
            printf(")\n");
        }

        /* 打印输出信息 */
        printf("Number of outputs = %zu\n", num_output_nodes);
        /* 遍历所有输出结点 */
        for (int i = 0; i < num_output_nodes; i++) {
            /* 打印输出节点名称 */
            printf("  Output %d : name=%s\n", i, output_node_names[i]);

            /* 打印输出结点类型 */
            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("  Output %d : type=%d\n", i, type);

            /* 打印输出的维度 */                      
            printf("  Output %d : dimension=(%d", i, output_nodes_dims[i][0]);
            for (int j = 1; j < output_nodes_dims[i].size(); j++)
                printf(", %d", output_nodes_dims[i][j]);
            printf(")\n");
        }
    }
    
    void ONNXRT_YOLOv4::run(std::vector<float> &input_values){
        /* 创建输入 */
        auto memory_info = Ort::MemoryInfo::CreateCpu(  OrtAllocatorType::OrtArenaAllocator, 
                                                        OrtMemType::OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(  memory_info, 
                                                                    input_values.data(), 
                                                                    input_values.size(), 
                                                                    input_nodes_dims[0].data(), 
                                                                    4);
        /* 进行识别 */
        auto output_tensors = session->Run( Ort::RunOptions{nullptr}, 
                                            input_node_names.data(),
                                            &input_tensor,
                                            1,
                                            output_node_names.data(),
                                            2);
        
        postProcess(output_tensors);

    }
    void ONNXRT_YOLOv4::drawResults(cv::Mat &mat){
        for (int i=0; i<result.size(); i++) {
            cv::Point p1, p2;
            p1.x = mat.cols * result[i].point1_x;
            p1.y = mat.rows * result[i].point1_y;
            p2.x = mat.cols * result[i].point2_x;
            p2.y = mat.rows * result[i].point2_y;

            cv::rectangle(mat, p1, p2, cv::Scalar(255,0,0), 2);
            cv::putText(mat, names[result[i].class_index], p1, 
                        cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 
                        2, cv::Scalar(255,255,0), 2);
            //cv::imwrite("./result.jpg", mat);
        }
    }
    void ONNXRT_YOLOv4::oneClassNMS(const std::vector<Yolo_Candidate> &candidate, 
                                    std::vector<Yolo_Candidate> &res, 
                                    int start, int end, float threshold_nms, bool print=false){
        std::vector<int> table_active(end-start, 1);
        for(int i=0; i<table_active.size(); i++) {
            if (table_active[i]){
                res.push_back(candidate[start+i]);
                for (int j=i+1; j<table_active.size(); j++) {
                    if (table_active[j] && intersectionOverUnion(   candidate[start+i].point1_x, candidate[start+i].point1_y,
                                                                    candidate[start+i].point2_x, candidate[start+i].point2_y,
                                                                    candidate[start+j].point1_x, candidate[start+j].point1_y,
                                                                    candidate[start+j].point2_x, candidate[start+j].point2_y)>threshold_nms) {
                        table_active[j] = 0;
                        if (print) {
                            printf("%s->%s = %f\n", names[candidate[start+i].class_index].c_str(),names[candidate[start+j].class_index].c_str(), intersectionOverUnion(   candidate[start+i].point1_x, candidate[start+i].point1_y,
                                                                    candidate[start+i].point2_x, candidate[start+i].point2_y,
                                                                    candidate[start+j].point1_x, candidate[start+j].point1_y,
                                                                    candidate[start+j].point2_x, candidate[start+j].point2_y));
                        }
                    }
                }
            }
        }
    }
    void ONNXRT_YOLOv4::nonMaximumSuppression(  const std::vector<Yolo_Candidate> &candidate, 
                                                std::vector<Yolo_Candidate> &res, 
                                                float threshold_nms){
        int begin = 0;
        int end = 0;
        int present_class = candidate[0].class_index;
        while(true){
            end++;
            /* 若遍历完成 */
            if (end>=candidate.size()){
                oneClassNMS(candidate, res, begin, end, threshold_nms);
                break;
            }
            // /* 若类别发生变化 */
            // else if(candidate[end].class_index != present_class){
            //     present_class = candidate[end].class_index;
            //     oneClassNMS(candidate, res, begin, end, threshold_nms);
            //     begin = end;
            // }
        }
        // std::vector<Yolo_Candidate> tmp = res;
        // res.clear();
        // std::sort(tmp.begin(), tmp.end(), [](const Yolo_Candidate &c1, const Yolo_Candidate &c2){return c1.conf>c2.conf;});
        // oneClassNMS(tmp, res, 0, tmp.size(), 0.75, true);
    }
    void ONNXRT_YOLOv4::postProcess(std::vector<Ort::Value> &arr, float threshold_conf, float threshold_nms){
        float *res1 = arr[0].GetTensorMutableData<float>();
        float *res2 = arr[1].GetTensorMutableData<float>();
        std::vector<Yolo_Candidate> candidate;
        std::vector<Yolo_Candidate> res;

        /* 筛选出置信度高于阈值的结果，保存在candidate中 */
        for(int i=0; i<output_nodes_dims[0][1]; i++){
            float max_value = res2[i*80];
            int max_index = 0;
            for (int j=0; j<80; j++){
                if (res2[i*80+j] > max_value) {
                    max_value = res2[i*80+j];
                    max_index = j;
                }
            }
            if (max_value > threshold_conf){
                Yolo_Candidate cand = { max_index, max_value,
                                        res1[i*4], res1[i*4+1],
                                        res1[i*4+2], res1[i*4+3]};
                candidate.push_back(cand);
            }
        }
        if (candidate.size() == 0){
            return;
        }
        /* 按类别、置信度排序 */
        std::sort(candidate.begin(), candidate.end(), [](const Yolo_Candidate &c1, const Yolo_Candidate &c2){return c1.conf>c2.conf;});
        //std::stable_sort(candidate.begin(), candidate.end(), [](const Yolo_Candidate &c1, const Yolo_Candidate &c2){return c1.class_index<c2.class_index;});

        names = loadNames("../../data/coco.names");

        nonMaximumSuppression(candidate, res, threshold_nms);

        for (int i=0; i<res.size(); i++){
            printf("Index: %d\n", i);
            printf("  Class: %d\n", res[i].class_index);
            printf("  Class: %s\n", names[res[i].class_index].c_str());
            printf("  Confidence: %f\n", res[i].conf);
            printf("  Point 1: (%f, %f)\n", res[i].point1_x, res[i].point1_y);
            printf("  Point 2: (%f, %f)\n", res[i].point2_x, res[i].point2_y);
        }

        result = res;
    }
    std::vector<std::string> ONNXRT_YOLOv4::loadNames(const std::string &path){
        std::vector<std::string> names;
        std::string tmp;
        std::ifstream infile(path);
        if (!infile.is_open()){
            std::cerr << "Could not open: " << path << std::endl;
            return names;
        }
        while (!infile.eof()) {
            std::getline(infile, tmp);
            names.push_back(tmp);
        }
        infile.close();
        return names;
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