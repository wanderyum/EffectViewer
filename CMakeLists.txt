# cmake版本
cmake_minimum_required(VERSION 3.2)

# 项目名称
project(EffectViewer)

# 让cmake自动去寻找Qt5
SET(Qt5_DIR "D:\\ProgramFiles\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\cmake\\Qt5" CACHE PATH "Initial cahe" FORCE)
SET(CMAKE_BUILD_TYPE "Release")
FIND_PACKAGE(Qt5 COMPONENTS Widgets REQUIRED)

# 采用C++14标准
SET(CMAKE_CXX_STANDARD 14)

# 添加所有的源文件
SET(EffectViewer_SOURCES 
	main.cpp
)

# 添加所有的头文件
#SET(EffectViewer_HEADERS 
#	mywidget.h
#	mydialog.h
#)

# 添加所有的.ui文件
#SET(EffectViewer_FORMS 
#	mywidget.ui
#	mydialog.ui
#)

# 添加资源文件
#SET(EffectViewer_RESOURCES 
#	img.qrc
#)
#调用预编译器moc，需要使用 QT5_WRAP_CPP宏
QT5_WRAP_CPP(EffectViewer_HEADERS_MOC ${EffectViewer_HEADERS})
#使用uic处理.ui文件
QT5_WRAP_UI(EffectViewer_FORMS_HEADERS ${EffectViewer_FORMS})
#使用rcc处理.qrc文件
QT5_ADD_RESOURCES(EffectViewer_RESOURCES_RCC ${EffectViewer_RESOURCES})

# 这些生成的中间文件都会在build目录下，这样的话，编译器则不能定位由uic程序产生的诸如_ui_mainwindow.h等文件。所以，我们需要把build目录添加到包含目录中
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

# 生成可执行文件
ADD_EXECUTABLE(EffectViewer 
	${EffectViewer_SOURCES} 
	${EffectViewer_HEADERS_MOC} 
    ${EffectViewer_FORMS_HEADERS}
	${EffectViewer_RESOURCES_RCC} 
)

# 为target添加需要链接的共享库
TARGET_LINK_LIBRARIES(EffectViewer ${Qt5Widgets_LIBRARIES})