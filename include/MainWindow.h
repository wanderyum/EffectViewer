#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets/QWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QFileDialog>
#include <QtGui/QPainter>
#include <QtGui/QImage>

#include <QtCore/QDebug>

#include "CVFunctions.h"
#include "ORTFunctions.h"

struct InterfaceConfig;
class MainWindow;
class PictureFrame;
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
    QFrame *mCamera;

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

class PictureFrame: public QFrame{
    Q_OBJECT

public:
    
    explicit PictureFrame(QWidget *parent, const InterfaceConfig &mIntConfig);
    ~PictureFrame();
    void resetFrame();

private:
    ViewPort *mPort;
    QPushButton *mChooseImage;
    QPushButton *mApply;
    QImage *mQImg;
    cv::Mat mMat;
    
    void readAndShow(const QString &fileName);
    void showImage(const cv::Mat &mat);

private slots:
    void chooseImage();
    void applyNetworks();
};

class ViewPort: public QFrame {
public:
    explicit ViewPort(QWidget *parent, const InterfaceConfig &mIntConfig, QImage **mQImg);
    ~ViewPort();

protected:
    void paintEvent(QPaintEvent *event);

private:
    QPainter mPainter;
    int centerX;
    int centerY;
    QImage **mQImg;
};

#endif