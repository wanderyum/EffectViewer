#include "../include/NetworkManager.h"

NetworkManager *NetworkManager::ptr = nullptr;
NetworkManager *NetworkManager::inst(){
    if (!ptr) {
        ptr = new NetworkManager;
    }
    return ptr;
}
void NetworkManager::addRecord(ortn::Network networkType, const std::string &path){
    NetworkRecord record = {networkType, path, nullptr};
    ortn::ORT_Model* pModel;
    switch(networkType) {
        case (ortn::Network::YOLO_V4):
            pModel = new ortn::ORT_YOLOv4();
            break;
        default:
            pModel = nullptr;
    }
    if (pModel) {
        record.model = pModel;
        mRecords.push_back(record);
    }
}
void NetworkManager::deleteRecord(int index){
    if (index < mRecords.size()) {
        delete mRecords[index].model;
        mRecords.erase(mRecords.begin()+index);
    }
}
void NetworkManager::clearRecords(){
    for (int i=0; i<mRecords.size(); i++) {
        delete mRecords.back().model;
        mRecords.pop_back();
    }
}
void NetworkManager::loadNetworks(){
    for (auto &item: mRecords) {
        if (!(item.model->isLoaded())) {
            item.model->loadModel(item.path);
            item.model->printIOInfo();
        }
    }
}
cv::Mat NetworkManager::preprocess(const cv::Mat &raw){

    /* 复制一份输入备用 */
    inputMat = raw.clone();

    return raw;
}
ortn::ORT_Result NetworkManager::compute(cv::Mat &input){

    if (mRecords.size()) {
        ortn::ORT_Result ret = mRecords[0].model->run(input);

        for (int i=1; i<mRecords.size(); i++) {
            ;
        }
        return ret;
    }
    else {
        ortn::ORT_Result ret = {ortn::ResultType::NO_RESULT,
                                ortn::ResultFormat::CV_MAT,
                                ortn::MissionType::CLASSIFICATION,
                                cv::Mat(), 
                                Ort::Value(nullptr)};
        return ret;
    }
}
cv::Mat NetworkManager::postprocess(const ortn::ORT_Result &res){
    if (res.resType == ortn::ResultType::FINAL) {
        if (res.misType == ortn::MissionType::OBJECT_DETECTION) {
            if (res.resForm == ortn::ResultFormat::CV_MAT) {
                for (int i=0; i<res.matData.rows; i++) {
                    float *pData = ((float*)res.matData.data);
                    markObject( inputMat, 
                                res.stringVector[(int)pData[i*6]],
                                pData[i*6+1],
                                pData[i*6+2],
                                pData[i*6+3],
                                pData[i*6+4],
                                pData[i*6+5]);
                }
                return inputMat;
            }
        }
    }
    return cv::Mat();
}
cv::Mat NetworkManager::infer(const cv::Mat &input){
    cv::Mat img = NetworkManager::inst()->preprocess(input);

    ortn::ORT_Result res_raw = NetworkManager::inst()->compute(img);

    return NetworkManager::inst()->postprocess(res_raw);
}
void NetworkManager::markObject(cv::Mat &src, const std::string &className, 
                    float conf, float x1, float y1, float x2, float y2) {
    cv::Point p1, p2;
    p1.x = (int)(src.cols * x1);
    p1.y = (int)(src.rows * y1);
    p2.x = (int)(src.cols * x2);
    p2.y = (int)(src.rows * y2);

    cv::rectangle(src, p1, p2, cv::Scalar(255,0,0), 2);
    cv::putText(src, className, p1, 
                cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 
                2, cv::Scalar(255,255,0), 2);
}
NetworkManager::NetworkManager(){}
NetworkManager::~NetworkManager(){
    for (auto item: mRecords){
        delete item.model;
    }
}