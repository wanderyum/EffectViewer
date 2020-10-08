#include "../include/MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent) {

    setupConfig();
    setupMainWindow();
    setupSource();
    setupInterfaces();

    changeInterface(0);

    connect(mSource, SIGNAL(currentIndexChanged(int)), this, SLOT(sourceChanged(int)));
    //connect(this, SIGNAL(sourceReturn(int)), this, SLOT(changeInterface(int)));
}
MainWindow::~MainWindow(){}

void MainWindow::setupConfig(){
    mIntConfig.font.setFamily(QString::fromUtf8("\346\245\267\344\275\223"));
    mIntConfig.font.setPointSize(19);

    mIntConfig.windowSize.setWidth(1400);
    mIntConfig.windowSize.setHeight(900);

    mIntConfig.frameSize.setWidth(1300);
    mIntConfig.frameSize.setHeight(700);

    mIntConfig.portSize.setWidth(800);
    mIntConfig.portSize.setHeight(600);
}
void MainWindow::setupMainWindow(){
    this->resize(mIntConfig.windowSize);
    this->setMaximumSize(mIntConfig.windowSize);
    this->setMinimumSize(mIntConfig.windowSize);
    this->setWindowTitle(tr("EffectViewer v0.1"));
}
void MainWindow::setupSource(){
    QLabel *labSource = new QLabel(this);
    labSource->setObjectName(QString::fromUtf8("label_source"));
    labSource->setFont(mIntConfig.font);
    labSource->setText(tr("Data Source: "));
    labSource->adjustSize();
    labSource->move(500, 50);
    
    mSource = new QComboBox(this);
    mSource->setObjectName(QString::fromUtf8("source"));
    mSource->addItem(tr("Picture"));
    mSource->addItem(tr("Video"));
    mSource->addItem(tr("Camera"));
    mSource->setFont(mIntConfig.font);
    mSource->move(labSource->x()+labSource->width(),labSource->y());
}
void MainWindow::setupInterfaces(){
    QSize frameSize(1300,700);

    mPicture = new PictureFrame(this, mIntConfig);

    mVideo = new QFrame(this);
    mVideo->setFrameShape(QFrame::Box);
    mVideo->setLineWidth(3);
    mVideo->resize(frameSize);
    mVideo->move(50, 150);
    mVideo->setVisible(false);

    mCamera = new CameraFrame(this, mIntConfig);
}
void MainWindow::sourceChanged(int index){
    changeInterface(index);
}
void MainWindow::changeInterface(int val){
    qDebug("Source: %d", val);
    switch(val){
        case 0: /* 图片 */
            mPicture->setVisible(true);
            mPicture->resetFrame();
            mCamera->setVisible(false);
            mCamera->mpTimer->stop();
            break;
        case 1: /* 视频 */
            mPicture->setVisible(false);
            mCamera->setVisible(false);
            mCamera->mpTimer->stop();
            break;
        case 2: /* 摄像头 */
            mCamera->setVisible(true);
            mCamera->mpTimer->start(100);
            mPicture->setVisible(false);
            break;
        default:
            break;
    }
}

MediaFrame::MediaFrame(QWidget *parent):
    QFrame(parent){}
MediaFrame::~MediaFrame(){}
void MediaFrame::BGR2RGB(const cv::Mat &bgr){
    cv::cvtColor(bgr, mRaw_RGB, cv::COLOR_BGR2RGB);
}

PictureFrame::PictureFrame(QWidget *parent, const InterfaceConfig &mIntConfig):
    MediaFrame(parent) {

    this->setFrameShape(QFrame::Box);
    this->setLineWidth(3);
    this->resize(mIntConfig.frameSize);
    this->move(50, 150);
    this->setVisible(false);

    mPort = new ViewPort(this, mIntConfig);

    mChooseImage = new QPushButton(this);
    mChooseImage->setFont(mIntConfig.font);
    mChooseImage->setText(tr("Please choose an image:"));
    mChooseImage->adjustSize();
    mChooseImage->move(930, 70);

    mpLoadNetworks = new QPushButton(this);
    mpLoadNetworks->setFont(mIntConfig.font);
    mpLoadNetworks->setText(tr("Load Networks"));
    mpLoadNetworks->adjustSize();
    mpLoadNetworks->move(930, 550);

    mApply = new QPushButton(this);
    mApply->setFont(mIntConfig.font);
    mApply->setText(tr("Cancel"));
    mApply->adjustSize();
    mApply->setText(tr("Apply"));
    mApply->move(1150, 550);

    connect(mChooseImage, SIGNAL(clicked()), this, SLOT(chooseImage()));
    connect(mpLoadNetworks, SIGNAL(clicked()), this, SLOT(loadNetworks()));
    connect(mApply, SIGNAL(clicked()), this, SLOT(applyNetworks()));
}
PictureFrame::~PictureFrame(){}

void PictureFrame::resetFrame(){

}
void PictureFrame::loadImage(const QString &fileName){
    /* 带有中文的路径不经toLocal8Bit转换会导致open cv无法读取 */
    mIntermediate = cvf::readImage(fileName.toLocal8Bit().toStdString());
}
void PictureFrame::chooseImage(){
    QString fileName = QFileDialog::getOpenFileName(this, 
                                                    tr("Please Choose an image: "),
                                                    ".",
                                                    tr("Image(*jpg *jpeg *png)"));
    qDebug() << "file name: " << fileName;
    if (fileName.size()){
        loadImage(fileName);
        BGR2RGB(mIntermediate);
        mPort->mTarget = &mRaw_RGB;
        //mPort->update();
    }
}
void PictureFrame::loadNetworks(){
    NetworkManager::inst()->clearRecords();
    NetworkManager::inst()->addRecord(ortn::Network::YOLO_V4, "yolov4_1_3_608_608_static.onnx");
    NetworkManager::inst()->loadNetworks();
}
void PictureFrame::applyNetworks(){

    if (mPort->mTarget == &mRaw_RGB) {
        cv::Mat img = NetworkManager::inst()->preprocess(mRaw_RGB);

        ortn::ORT_Result res_raw = NetworkManager::inst()->compute(img);

        mProcessed = NetworkManager::inst()->postprocess(res_raw);

        mApply->setText(tr("Cancel"));
        mPort->mTarget = &mProcessed;
        mPort->update();
    }
    else {
        mApply->setText(tr("Apply"));
        mPort->mTarget = &mRaw_RGB;
        mPort->update();
    }
}

CameraFrame::CameraFrame(QWidget *parent, const InterfaceConfig &mIntConfig):
    MediaFrame(parent){
    this->setFrameShape(QFrame::Box);
    this->setLineWidth(3);
    this->resize(mIntConfig.frameSize);
    this->move(50, 150);
    this->setVisible(false);

    mCapture.open(0);
    // mCapture.read(mIntermediate);
    // qDebug() << "Capture size: (" << mIntermediate.cols << ", " << mIntermediate.rows << ")" << endl;
    
    mPort = new ViewPort(this, mIntConfig);
    loadCamera();
    mPort->mTarget = &mRaw_RGB;

    mpLoadNetworks = new QPushButton(this);
    mpLoadNetworks->setFont(mIntConfig.font);
    mpLoadNetworks->setText(tr("Load Networks"));
    mpLoadNetworks->adjustSize();
    mpLoadNetworks->move(930, 550);

    mApply = new QPushButton(this);
    mApply->setFont(mIntConfig.font);
    mApply->setText(tr("Cancel"));
    mApply->adjustSize();
    mApply->setText(tr("Apply"));
    mApply->move(1150, 550);

    mpTimer = new QTimer(this);

    connect(mpTimer, SIGNAL(timeout()), this, SLOT(loadCamera()));
    connect(mpLoadNetworks, SIGNAL(clicked()), this, SLOT(loadNetworks()));
    connect(mApply, SIGNAL(clicked()), this, SLOT(applyNetworks()));
}
CameraFrame::~CameraFrame(){
    mCapture.release();
}

void CameraFrame::loadCamera(){
    //qDebug() << "loading camera..." << endl;
    mCapture.read(mIntermediate);
    cv::cvtColor(mIntermediate, mRaw_RGB, cv::COLOR_BGR2RGB);
    if (mPort->mTarget == &mProcessed) {
        mProcessed = NetworkManager::inst()->infer(mRaw_RGB);
    }
    
    mPort->update();

    // mCapture.read(mMat);
    // cv::cvtColor(mMat, mMat, cv::COLOR_BGR2RGB);
    // showImage(mMat);
}
void CameraFrame::loadNetworks(){
    NetworkManager::inst()->clearRecords();
    NetworkManager::inst()->addRecord(ortn::Network::YOLO_V4, "yolov4_1_3_608_608_static.onnx");
    NetworkManager::inst()->loadNetworks();
}
void CameraFrame::applyNetworks(){
    if (mPort->mTarget == &mRaw_RGB) {

        mProcessed = NetworkManager::inst()->infer(mRaw_RGB);

        mApply->setText(tr("Cancel"));
        mPort->mTarget = &mProcessed;
        mPort->update();
    }
    else {
        mApply->setText(tr("Apply"));
        mPort->mTarget = &mRaw_RGB;
        mPort->update();
    }
}

ViewPort::ViewPort(QWidget *parent, const InterfaceConfig &mIntConfig):
    QFrame(parent) {

    this->resize(mIntConfig.portSize);
    this->move(50, 50);
    
    centerX = this->x()+this->width()/2;
    centerY = this->y()+this->height()/2;

    mTarget = nullptr;
}
ViewPort::~ViewPort(){}

void ViewPort::paintEvent(QPaintEvent *event){
    if (mTarget != nullptr){

        double ratio_x = 1.0 * width() / mTarget->cols;
        double ratio_y = 1.0 * height() / mTarget->rows;
        double ratio = ratio_x<ratio_y?ratio_x:ratio_y;

        int w = ratio*(mTarget->cols);
        int h = ratio*(mTarget->rows);
        
        /* 将其宽度缩放为4的倍数，以便QImage绘制(其存储空间按4对齐) */
        if (w%4) w = ((w/4)+1)*4;

        cv::resize(*mTarget, temp, cv::Size2i(w, h));

        mToShow = QImage(temp.cols, temp.rows, QImage::Format_RGB888);;
        memcpy(mToShow.bits(), temp.data, temp.cols*temp.rows*temp.elemSize());
        int drawX = centerX - mToShow.width()/2;
        int drawY = centerY - mToShow.height()/2;

        mPainter.begin(this);
        mPainter.drawImage(drawX, drawY, mToShow);
        mPainter.end();
    }
}