#include "../include/MainWindow.h"
#include <QApplication>
#include <QDialog>
#include <QLabel>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow mW;
    
    mW.show();
    return a.exec();
}
