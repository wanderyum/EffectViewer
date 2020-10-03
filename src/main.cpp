#include "../include/MainWindow.h"
#include <QApplication>
#include <QDialog>
#include <QLabel>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow mW;
    // QDialog w;
    // w.resize(400, 300);
    // QLabel label(&w);
    // label.move(120, 120);
    // label.setText(QObject::tr("Hello World! Qt!"));
    
    mW.show();
    return a.exec();
}
