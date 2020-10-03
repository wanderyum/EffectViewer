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

    mCamera = new QFrame(this);
    mCamera->setFrameShape(QFrame::Box);
    mCamera->setLineWidth(3);
    mCamera->resize(frameSize);
    mCamera->move(50, 150);
    mCamera->setVisible(false);
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
            break;
        case 1: /* 视频 */
            mPicture->setVisible(false);
            break;
        case 2: /* 摄像头 */
            mPicture->setVisible(false);
            break;
        default:
            break;
    }
}

PictureFrame::PictureFrame(QWidget *parent, const InterfaceConfig &mIntConfig):
    QFrame(parent) {

    this->setFrameShape(QFrame::Box);
    this->setLineWidth(3);
    this->resize(mIntConfig.frameSize);
    this->move(50, 150);
    this->setVisible(false);

    mQImg = nullptr;

    mPort = new ViewPort(this, mIntConfig, &mQImg);

    mChooseImage = new QPushButton(this);
    mChooseImage->setFont(mIntConfig.font);
    mChooseImage->setText(tr("Please choose an image:"));
    mChooseImage->adjustSize();
    mChooseImage->move(930, 70);

    mApply = new QPushButton(this);
    mApply->setFont(mIntConfig.font);
    mApply->setText(tr("Apply"));
    mApply->adjustSize();
    mApply->move(930, 400);

    connect(mChooseImage, SIGNAL(clicked()), this, SLOT(chooseImage()));
    connect(mApply, SIGNAL(clicked()), this, SLOT(applyNetworks()));
}
PictureFrame::~PictureFrame(){}

void PictureFrame::resetFrame(){

}
void PictureFrame::readAndShow(const QString &fileName){
    /* 带有中文的路径不经toLocal8Bit转换会导致open cv无法读取 */
    mMat = cvf::readImage(fileName.toLocal8Bit().toStdString());

    showImage(mMat);
}
void PictureFrame::showImage(const cv::Mat &mat){
    QImage tmp(mat.cols, mat.rows, QImage::Format_RGB888);
    memcpy(tmp.bits(), mat.data, mat.cols*mat.rows*mat.elemSize());
    float ratio_x = 1.0f * mPort->width() / tmp.width();
    float ratio_y = 1.0f * mPort->height() / tmp.height();
    float ratio = ratio_x<ratio_y?ratio_x:ratio_y;
    
    if (mQImg != nullptr) {delete mQImg;}
    mQImg = new QImage(tmp.scaled((int)(ratio*tmp.width()),(int)(ratio*tmp.height())));
    mPort->update();
}
void PictureFrame::chooseImage(){
    QString fileName = QFileDialog::getOpenFileName(this, 
                                                    tr("Please Choose an image: "),
                                                    ".",
                                                    tr("Image(*jpg *jpeg *png)"));
    qDebug() << "file name: " << fileName;
    if (fileName.size()){
        readAndShow(fileName);
    }
    
}
void PictureFrame::applyNetworks(){
    ortf::ONNXRT_YOLOv4 ort1;
    ort1.loadModel("yolov4_1_3_608_608_static.onnx");
    ort1.printIOInfo();

    //readAndShow("D:\\Github\\github\\pytorch-YOLOv4\\data\\dog.jpg");
    std::vector<float> input_values = cvf::prepareImage(mMat, 608, 608);
    ort1.run(input_values);
    ort1.drawResults(mMat);
    showImage(mMat);
}

ViewPort::ViewPort(QWidget *parent, const InterfaceConfig &mIntConfig, QImage **mQImg):
    QFrame(parent) {
    this->mQImg = mQImg;

    this->resize(mIntConfig.portSize);
    this->move(50, 50);
    
    centerX = this->x()+this->width()/2;
    centerY = this->y()+this->height()/2;
}
ViewPort::~ViewPort(){}

void ViewPort::paintEvent(QPaintEvent *event){
    if (*mQImg != nullptr){
        
        int drawX = centerX - (*mQImg)->width()/2;
        int drawY = centerY - (*mQImg)->height()/2;

        mPainter.begin(this);
        mPainter.drawImage(drawX, drawY, **mQImg);
        mPainter.end();
    }
}