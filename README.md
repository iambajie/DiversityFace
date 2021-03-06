# 人脸识别中不同方法对比分析

## 项目设置

（1）问题描述

对于待分类的图像，根据模式识别算法判断它和哪一个图像最相似

（2）数据集来源

实验数据集来自Biometric Ideal Test官网http://biometrics.idealtest.org/dbDetailForUser.do?id=9 ，保存在FaceV5文件夹下。

选择其中100个人的不同的4张图片作为数据集，选择每个人的一张图片作为测试集共100张图像，保存在used文件夹下，将剩余的图像作为训练集，共300张图像，保存在unused文件夹下。每张图片的第0 ~ 2位用数字表示，用来标识每个人的身份，最终测试样本预测的类别也是需要根据图片的名称来进行判断是否识别准确。

（3）评价标准

不同方法的评价指标采用单一实数指标：正确率accuracy=预测正确的样本数/预测样本总数。

knn分类器中的超参数K取值区间设置为[1,5]，分类的距离采用L2二阶范数、相关性系数、闵可夫斯基距离

## 不同的算法比对

模式识别算法中的人脸识别分为两部分：特征提取和数据比对两个阶段

### 特征提取

（1）基于颜色像素的RGB模型

特征：将训练样本和输入样本分别计算出每个图像的颜色直方图hist，对直方图归一化后保存的数据作为图像的特征。

（2）基于统计特征脸的PCA方法

PCA主成分分析原理总体来说是通过K-L变换将高维向量变成低维向量，形成特征子空间，每个图像就通过投影到该子空间上作为自身特征向量进行后续识别。其中主成分分析法是指找到一个空间，即形成的特征子空间，在该空间上消除了数据的相关性，每个类别数据能够很好的分离。通过K-L变换求出了特征空间，将训练样本和测试样本中所有图片投影在特征空间中就能求得每个图片的特征向量。

（3）基于神经网络的模型

从官网下载VGG16的预训练模型，保留全连接层，分别将训练样本和测试样本输入网络，经过卷积层、池化层、全连接层，最终输出一个4096维列向量作为每个图片的特征向量。

### 数据比对

采用最近邻分类knn分类器，根据已保存的训练样本和测试样本的特征向量，找到距离测试样本最近的类别作为预测的人脸，具体计算步骤：
（1）定义分类器中的超参数K，表示进行投票决策的样本的数目。
（2）遍历所有测试样本，计算样本的特征向量和其他每个训练样本的距离，按照由近至远进行排序。
（3）计算离测试样本最近的K个样本，统计各个分类，将最多数量的分类代表的人脸作为当前测试样本的预测值。

### 特征提取+数据比对

基于迁移学习的模型

基于迁移学习的识别方法将提取特征和数据比都交给网络自己完成，最终输出人脸的识别结果。基于迁移学习的方法VGG16Fc采用基于模型的迁移学习的方法，保留VGG16至全连接层的预训练模型，在模型最后增加两层新的全连接层，作为需要微调的神经元部分。

预训练部分的网络用来识别图片的轮廓、线段、人脸位置、表情等图像信息，微调部分的网络进行分类识别。根据训练样本中每个图像的特征向量和图像所代表的类别训练后两层全连接层，为防止过拟合，增加Dropout层，随机丢弃一些神经元节点，最终将测试样本的特征向量输入训练好的网络中得到预测的人脸结果。

### 总体过程

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200706151929.png)

## 结果

（1）不同距离的度量特征向量

RGB+KNN方法中，使用相关性系数度量距离的平均识别率为55%，使用闵可夫斯基距离平均识别率为50.4%，使用L2范数平均识别率为55.4%。

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200706152534.png)

PCA+KNN方法中，使用相关性系数度量距离的平均识别率为75.2%，使用闵可夫斯基距离平均识别率为69.8%，使用L2范数平均识别率为72.6%。

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200706153119.png)

VGG+KNN方法中，使用相关性系数度量距离的平均识别率为89.4%，使用闵可夫斯基距离平均识别率为85.6%，使用L2范数平均识别率为89.4%。

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200706153149.png)

分析1：从结果可以看出：不同的度量向量的方法对算法具有一定影响。当选择传统的方法基于像素和统计的识别方法时，使用L2范数和相关性系数识别准确率接近，并且识别的结果优于闵可夫斯基距离，使用闵可夫斯基距离识别最差，因为它将各个分量同等看待，没有考虑到向量之间的相关性，所以识别效果差。

当选择深度学习的方法，使用神经网络时，三种度量方式识别都很准确，因为神经网络比传统只基于颜色像素、统计的特征脸学习了更多图片的信息，因此表示每个图像特征向量更准确，同一个人的特征向量距离很近，不同的人的特征向量距离非常远，所以选择不同的距离计算方式对识别效果基本无太大影响。

分析2：针对最近邻分类器K的取值，从结果可以看出，随着K的增加，预测的准确率先升高再降低，说明增加K会提高预测准确率，但是K太大，会导致最近的样本中其他样本数量过多，而导致分类错误。由于训练集中每个人的图像数量为3张，因此当K取值为2的时候分类效果最好，即K略小于训练集中同一个人的图片数量时分类效果最好，所以需要合理选择K的值。由于这里训练集图片数量较少，K=1时也能取得较好的分类效果。

（2）不同的人脸识别的方法

按照（1）中选择L2范数作为KNN分类器中度量距离的方式。

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200706153408.png)

分析1：从结果可以看出，基于颜色像素的方法仅考虑了人脸中的颜色，没有考虑人脸的轮廓的相似度等，所以该方法识别率最低，平均识别为55.4%，基于统计特征脸的方法考虑到了人脸的轮廓，利用特征脸，将图像投影到特征空间中，更多的考虑到了人脸的特点，所以识别效果较好，平均识别率达72.6%，但是这两种方法和使用神经网络的深度学习方法相比，识别效果都不佳，神经网络识别的方法从图像中提取CNN特征，能更好的对图像分类，平均识别率为89.4%。

分析2：将传统分类方法和神经网络分类的对比，基于神经网络的识别方法在提取出图像的特征后外接knn分类器进行识别，通过距离对图像进行分类，而基于迁移学习的方法是让网络学习图像的向量和对应人脸之间的对应关系，从而对测试图像进行识别分类，识别结果准确率86%，低于传统分类方法的识别结果，但并不完全因为最近邻分类方法由于网络学习分类的方法，基于迁移学习的方法在训练集上的精度最终能达97.78%，但是在测试集上的识别精度不佳，可能是由于训练样本集太小，只有300张图像，导致了过拟合。# diversityface
