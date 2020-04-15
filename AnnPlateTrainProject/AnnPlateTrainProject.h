// AnnPlateTrain.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#ifndef ANN_PLATE_TRAIN_PROJECT_H
#define ANN_PLATE_TRAIN_PROJECT_H

#include <iostream>
#include<opencv2/opencv.hpp>
#include<string>
#include"utils.h"

using namespace std;
using namespace cv;
using namespace ml;

#define ANN_CHAR_XML "E:/VisualStudio/FILE/ann/plateNum/model/plateNumAnnChar.xml"            //字母数字训练模型保存路径
#define ANN_CHINESE_XML "E:/VisualStudio/FILE/ann/plateNum/model/plateNumAnnChinese.xml"      //汉字训练模型保存路径
#define ANN_CHAR "E:/VisualStudio/FILE/ann/plateNum/samples/annChar"                          //字母数字路径
#define ANN_CHINESE "E:/VisualStudio/FILE/ann/plateNum/samples/annChinese"                    //汉字路径

struct TrainStruct {
	string fileName;
	int label;
};

void trainAnnChar();
void trainAnnChinese();
void getAnnHogFeatures(const Mat image, Mat& features);

#endif // !ANN_PLATE_TRAIN_H
