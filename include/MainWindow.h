#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets/QWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QFileDialog>
#include <QtCore/QTimer>
#include <QtGui/QPainter>
#include <QtGui/QImage>

#include <QtCore/QDebug>

#include "CVFunctions.h"
#include "NetworkManager.h"

struct InterfaceConfig;
class MainWindow;
class MediaFrame;
class PictureFrame;
class CameraFrame;
class ViewPort;

struct InterfaceConfig{
    QFont font;
    QSize windowSize;
    QSize frameSize;
    QSize portSize;
};

class MainWindow: public QWidget {

    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    //void setupUi(QWidget *MyWidget);


private:
    InterfaceConfig mIntConfig;
    QComboBox *mSource;
    PictureFrame *mPicture;
    QFrame *mVideo;
    CameraFrame *mCamera;

    void setupConfig();
    void setupMainWindow();
    void setupSource();
    void setupInterfaces();
    void changeInterface(int val);

signals:
    //void sourceReturn(int);

private slots:
    void sourceChanged(int index);
    
};

class MediaFrame: public QFrame{
    Q_OBJECT
public:
    explicit MediaFrame(QWidget *parent);
    ~MediaFrame();

protected:
    void BGR2RGB(const cv::Mat &bgr);

    cv::Mat mRaw_RGB;
    cv::Mat mProcessed;
    cv::Mat mIntermediate;
};

class PictureFrame: public MediaFrame{
    Q_OBJECT

public:
    explicit PictureFrame(QWidget *parent, const InterfaceConfig &mIntConfig);
    ~PictureFrame();
    void resetFrame();

private:
    ViewPort *mPort;
    QPushButton *mChooseImage;
    QPushButton *mpLoadNetworks;
    QPushButton *mApply;
    
    void loadImage(const QString &fileName);

private slots:
    void chooseImage();
    void loadNetworks();
    void applyNetworks();
};

class CameraFrame: public MediaFrame{
    Q_OBJECT

public:
    QTimer *mpTimer;

    explicit CameraFrame(QWidget *parent, const InterfaceConfig &mIntConfig);
    ~CameraFrame();

private:
    ViewPort *mPort;
    QPushButton *mpLoadNetworks;
    QPushButton *mApply;

    cv::VideoCapture mCapture;

    //void showImage(const cv::Mat &mat);

private slots:
    void loadNetworks();
    void applyNetworks();
    void loadCamera();
};

class ViewPort: public QFrame {
public:
    cv::Mat *mTarget;
    explicit ViewPort(QWidget *parent, const InterfaceConfig &mIntConfig);
    ~ViewPort();

protected:
    void paintEvent(QPaintEvent *event);

private:
    QPainter mPainter;
    int centerX;
    int centerY;
    QImage mToShow;
    cv::Mat temp;
};

#endif