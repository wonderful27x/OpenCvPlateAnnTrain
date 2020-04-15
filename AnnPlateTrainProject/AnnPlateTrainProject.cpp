// AnnPlateTrainProject.cpp: 定义应用程序的入口点。
//

#include "AnnPlateTrainProject.h"

//在人工智能里面有一个非常重要的核心部分那就是学习，也就是我们常说的机器学习，学习的算法有很多种，
//其中有一种以模拟人的大脑来实现，这就是人工神经网络(ANN:Artificial Neural Networks)。
//人工神经网络由无数的神经元组成，神经元由输入、激活函数、输出构成，
//每个神经元的输入由一个或多个其他神经元的输出乘以权重组成，也就是神经元的输入等于其他神经元的输出的加权，
//然后将这个输入作用于一个激活函数得到此神经元的输出。
//比较经典的人工神经网络由三层神经元组成，输入层、隐含层（隐含层可以有多层）、输出层，每一层由一个或多个神经元组成，
//前一层的输出作为下一层的输入依次传递，不能跨级，即前一层神经元只能与下一层直连，
//这就是多层前馈神经网络或称为多层前馈感知机（MLP:Multi-Layer Perceptron）

//本项目车牌字符识别就是运用MLP进行学习训练，步骤大概如下：
//1、把需要识别的物体通过某种途径获得x(输入层个数)个数据
//2、把输入数据分成y(隐藏层神经元个数)个数据(神经元) 根据隐藏层数重复这个步骤
//3、把第2步的输出 输入给输出层 然后输出z(输出层个数)个结果
//https://docs.opencv.org/4.1.1/dc/dd6/ml_intro.html#ml_intro_ann
//https://www.cnblogs.com/xinxue/p/5789421.html
//http://www.2cto.com/kf/201610/553336.html
//https://www.cnblogs.com/ronny/p/opencv_road_more_01.html

string message;
bool running = false;
long timeCount = 0;
void timeCountTask();
void timeStart(string startMessage,string runningMessage);
void timeEnd(string endMessage);
thread timeThread;

int main()
{
	trainAnnChar();
	trainAnnChinese();
	return 0;
}

void trainAnnChar()
{
	cout << "车牌字符识别-字母数字ann模型训练" << endl;
	//cout << "训练数据加载中..." << endl;
	timeStart("", "训练数据加载中...");

	//加载图片并打上标签
	vector<TrainStruct> trainDatas;
	//图片数据0-33
	for (int i = 0; i < 34; i++) {
		char path[100];
		sprintf(path, "%s/%d", ANN_CHAR, i);
		//读取目录下所有字符（同一个目录下存放的是同一个字符的n个样本）
		vector<string> files;
		getFiles(path, files);
		//打标签并保存，这里的标签对应的其实就是目录的索引，一种映射，
		//比如0目录下存放了100个“粤”字符，那么每个字符的标签就是0；
		for (string file : files) {
			trainDatas.push_back({ file,i });
		}
	}
	//预处理
	Mat samples;
	vector<int> labels;
	for (TrainStruct data : trainDatas) {
		//读出灰度图
		Mat image = imread(data.fileName, IMREAD_GRAYSCALE);
		if (image.empty()) {
			printf("加载样本失败，image: %s \n", data.fileName.c_str());
			continue;
		}
		//二值化
		threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
		//提取hog特征
		Mat features;
		getAnnHogFeatures(image, features);
		//归一化一通道一行
		features = features.reshape(1, 1);
		//2个集合 一个样本一个标签
		//Mat支持每一行插入（Mat::push_back），这一点与vector类似，
		//将一行Mat插入到另外 一个Mat我们只需要保证两者的cols是相同的。
		//于是循环结束后samples就变成了m行n列的矩阵，m就等于样本的数量，n取决于提取的hog特征，
		//所以samples的每一行就代表了一个样本的特征，而label对应的索引位置处的值则代表了该样本的标签。
		samples.push_back(features);
		labels.push_back(data.label);
	}
	//转换一下格式
	samples.convertTo(samples, CV_32FC1);
	//样本与标签的格式：设存在3个样本，一共分为2类，第0，1个样本为第0类，第2个样本是第1类
	//样本数据如下:（假设这就是样本samples）
	//1,1,1,1
	//1,1,1,1
	//1,1,1,1
	//对应的标签数据如下：
	//0,0  （代表标签0）
	//0,0  （代表标签0）
	//0,1  （代表标签1）

	//创建标签
	//由上面的分析知道samples的每一行对应一个样本，而由于标签也需要一个矩阵来表示，所以需要转换，
	//可以这样理解，标签矩阵应该与samples一一对应，即samples的第一行这个样本的标签对应标签矩阵第一行的值，
	//因此标签矩阵的行数等于samples的行数，由于一共有31中不同的标签，所以标签矩阵的列数等于31，
	//于是赋值就变成了下面的形式，比如第一样本的标签值是5，则样本标签矩阵第一行赋值为：00001 00000 00000 00000 00000 00000 0
	Mat trainLabels = Mat::zeros(labels.size(), 34, CV_32FC1);
	for (int i = 0; i < trainLabels.rows; i++) {
		trainLabels.at<float>(i, labels[i]) = 1.0f;
	}
	//创建训练数据，必须是CV_32FC1 （32位浮点类型，单通道）。数据必须是CV_ROW_SAMPLE的，即特点向量以行来存储
	Ptr<TrainData> trainData = TrainData::create(samples, SampleTypes::ROW_SAMPLE, trainLabels);
	//创建人工神经网络-多层感知机
	Ptr<ANN_MLP> classifier = ANN_MLP::create();
	Mat layers;
	//三层神经元：输入、隐藏、输出，如果有多个隐藏层可以创建更多的层数
	layers.create(1, 3, CV_32SC1);
	//输入层神经元数量(samples.cols应该和特征有关，但是输入层神经元为什么等于它，没理解)
	layers.at<int>(0) = samples.cols;
	//隐藏层神经元数量，这个需要调整
	layers.at<int>(1) = 68;
	//输出层神经元数量，等于标签类型数
	layers.at<int>(2) = 34;
	//设置神经网络层数
	classifier->setLayerSizes(layers);
	//设置激活函数
	//激活函数的作用是将这一层的输入（上一层的输出）转换成这一层的输出
	//p1->SIGMOID_SYM:激活函数类型，p2->1:激活函数参数α，p3->1激活函数参数β，opencv中使用的SIGMOID_SYM 激活函数是sigmoid的变形，
	//当α=β=1时，该函数把可能在较大范围内变化的输入值，“挤压” 到 (-1, 1) 的输出范围内,激活函数原型见链接
	//https://www.cnblogs.com/xinxue/p/5789421.html
	classifier->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	//设置神经网络的学习算法，MLP有多种方法实现学习，其中BACKPROP比较常用，收敛速度快
	//BACKPROP：BP算法，即误差逆向传播算法，使用此种算法的神经网络也叫BP神经网络
	classifier->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	//针对于反向传播法，主要是两个参数，一个是权值更新率bp_dw_scale和权值更新冲量bp_moment_scale。
	//这两个量一般情况设置为0.1就行了；太小了网络收敛速度会很慢，太大了可能会让网络越过最小值点。
	classifier->setBackpropWeightScale(0.1);   // 权值更新率 0.1
	classifier->setBackpropMomentumScale(0.1); // 权值更新冲量0.1
	//设置终止条件
	//p1->COUNT：终止条件为最大迭代次数 p2->20000：迭代20000次 p3->FLT_EPSILON误差FLT_EPSILON（当type=COUNT时这个参数无效）
	classifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 30000, FLT_EPSILON));

	timeEnd("训练数据加载完成");
	//cout << "训练数据加载完成，训练中..." << endl;

	timeStart("训练数据加载完成","字母数字训练中...");
	//int64 timeStart = getTickCount();
	classifier->train(trainData);
	//int64 timeEnd = getTickCount();
	timeEnd("字母数字训练完成");
	//保存模型
	classifier->save(ANN_CHAR_XML);
	//printf("训练完成，用时：%I64us\n", (timeEnd - timeStart) / 1000);
}


//车牌汉字训练
void trainAnnChinese()
{
	cout << "车牌字符识别-汉字ann模型训练" << endl;
	//cout << "训练数据加载中..." << endl;
	timeStart("", "训练数据加载中...");
	//加载图片并打上标签
	vector<TrainStruct> trainDatas;
	//图片数据0-30
	for (int i = 0; i < 31; i++) {
		char path[100];
		sprintf(path,"%s/%d",ANN_CHINESE,i);
		//读取目录下所有字符（同一个目录下存放的是同一个字符的n个样本）
		vector<string> files;
		getFiles(path, files);
		//打标签并保存，这里的标签对应的其实就是目录的索引，一种映射，
		//比如0目录下存放了100个“粤”字符，那么每个字符的标签就是0；
		for (string file : files) {
			trainDatas.push_back({file,i});
		}
	}
	//预处理
	Mat samples;
	vector<int> labels;
	for (TrainStruct data: trainDatas) {
		//读出灰度图
		Mat image = imread(data.fileName, IMREAD_GRAYSCALE);
		if (image.empty()) {
			printf("加载样本失败，image: %s \n", data.fileName.c_str());
			continue;
		}
		//二值化
		threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
		//提取hog特征
		Mat features;
		getAnnHogFeatures(image,features);
		//归一化一通道一行
		features = features.reshape(1, 1);
		//2个集合 一个样本一个标签
		//Mat支持每一行插入（Mat::push_back），这一点与vector类似，
		//将一行Mat插入到另外 一个Mat我们只需要保证两者的cols是相同的。
		//于是循环结束后samples就变成了m行n列的矩阵，m就等于样本的数量，n取决于提取的hog特征，
		//所以samples的每一行就代表了一个样本的特征，而label对应的索引位置处的值则代表了该样本的标签。
		samples.push_back(features);
		labels.push_back(data.label);
	}
	//转换一下格式
	samples.convertTo(samples,CV_32FC1);
	//样本与标签的格式：设存在3个样本，一共分为2类，第0，1个样本为第0类，第2个样本是第1类
	//样本数据如下:（假设这就是样本samples）
	//1,1,1,1
	//1,1,1,1
	//1,1,1,1
	//对应的标签数据如下：
	//0,0  （代表标签0）
	//0,0  （代表标签0）
	//0,1  （代表标签1）

	//创建标签
	//由上面的分析知道samples的每一行对应一个样本，而由于标签也需要一个矩阵来表示，所以需要转换，
	//可以这样理解，标签矩阵应该与samples一一对应，即samples的第一行这个样本的标签对应标签矩阵第一行的值，
	//因此标签矩阵的行数等于samples的行数，由于一共有31中不同的标签，所以标签矩阵的列数等于31，
	//于是赋值就变成了下面的形式，比如第一样本的标签值是5，则样本标签矩阵第一行赋值为：00001 00000 00000 00000 00000 00000 0
	Mat trainLabels = Mat::zeros(labels.size(),31,CV_32FC1);
	for (int i = 0; i < trainLabels.rows; i++) {
		trainLabels.at<float>(i, labels[i]) = 1.0f;
	}
	//创建训练数据，必须是CV_32FC1 （32位浮点类型，单通道）。数据必须是CV_ROW_SAMPLE的，即特点向量以行来存储
	Ptr<TrainData> trainData = TrainData::create(samples, SampleTypes::ROW_SAMPLE, trainLabels);
	//创建人工神经网络-多层感知机
	Ptr<ANN_MLP> classifier = ANN_MLP::create();
	Mat layers;
	//三层神经元：输入、隐藏、输出，如果有多个隐藏层可以创建更多的层数
	layers.create(1,3, CV_32SC1);
	//输入层神经元数量(samples.cols应该和特征有关，但是输入层神经元为什么等于它，没理解)
	layers.at<int>(0) = samples.cols;
	//隐藏层神经元数量，这个需要调整
	layers.at<int>(1) = 62;
	//输出层神经元数量，等于标签类型数
	layers.at<int>(2) = 31;
	//设置神经网络层数
	classifier->setLayerSizes(layers);
	//设置激活函数
	//激活函数的作用是将这一层的输入（上一层的输出）转换成这一层的输出
	//p1->SIGMOID_SYM:激活函数类型，p2->1:激活函数参数α，p3->1激活函数参数β，opencv中使用的SIGMOID_SYM 激活函数是sigmoid的变形，
	//当α=β=1时，该函数把可能在较大范围内变化的输入值，“挤压” 到 (-1, 1) 的输出范围内,激活函数原型见链接
	//https://www.cnblogs.com/xinxue/p/5789421.html
	classifier->setActivationFunction(ANN_MLP::SIGMOID_SYM,1,1);
	//设置神经网络的学习算法，MLP有多种方法实现学习，其中BACKPROP比较常用，收敛速度快
	//BACKPROP：BP算法，即误差逆向传播算法，使用此种算法的神经网络也叫BP神经网络
	classifier->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	//针对于反向传播法，主要是两个参数，一个是权值更新率bp_dw_scale和权值更新冲量bp_moment_scale。
	//这两个量一般情况设置为0.1就行了；太小了网络收敛速度会很慢，太大了可能会让网络越过最小值点。
	classifier->setBackpropWeightScale(0.1);   // 权值更新率 0.1
	classifier->setBackpropMomentumScale(0.1); // 权值更新冲量0.1
	//设置终止条件
	//p1->COUNT：终止条件为最大迭代次数 p2->20000：迭代20000次 p3->FLT_EPSILON误差FLT_EPSILON（当type=COUNT时这个参数无效）
	classifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,30000, FLT_EPSILON));

	timeEnd("训练数据加载完成");
	//cout << "训练数据加载完成，训练中..." << endl;

	timeStart("训练数据加载完成", "汉字训练中...");
	//int64 timeStart = getTickCount();
	classifier->train(trainData);
	//int64 timeEnd = getTickCount();
	timeEnd("汉字训练完成");
	//保存模型
	classifier->save(ANN_CHINESE_XML);
	//printf("训练完成，用时：%I64us\n", (timeEnd - timeStart) / 1000);
}


//提取HOG特征
//HOG特征：方向梯度直方图，使用于轮廓提取
void getAnnHogFeatures(const Mat image, Mat& features)
{
	static int count = -1;
	//创建HOG特征
	HOGDescriptor hog(Size(128, 64), Size(16, 16), Size(8, 8), Size(8, 8), 3);
	vector<float> descriptor;
	Mat trainImg = Mat(hog.winSize, CV_32S);
	//归一化
	resize(image, trainImg, hog.winSize);
	//计算hog特征
	hog.compute(trainImg, descriptor, Size(8, 8));
	//转成mat
	Mat featureMat = Mat(descriptor);
	featureMat.copyTo(features);
	if (count == 1) {
		cout << "winW: " << hog.winSize.width << "	winH: " << hog.winSize.height << endl;
		count = -1;
	}
}

//计时任务
void timeCountTask() {
	while (running) {
		timeCount++;
		printf("=========%s%ld========\n", message.c_str(),timeCount);
		this_thread::sleep_for(chrono::milliseconds(1000));
	}
}

//开始计时
void timeStart(string startMessage,string runningMessage) {
	timeThread = thread(timeCountTask);
	printf("%s\n", startMessage.c_str());
	message = runningMessage;
	timeCount = 0;
	running = true;
	timeThread.detach();
}

//停止计时
void timeEnd(string endMessage) {
	running = false;
	printf("=========%s用时%lds========\n\n", endMessage.c_str(),timeCount);
}
