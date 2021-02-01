# face detection pyqt5

## 项目准备

本项目使用ide为vs2019，python环境为anaconda2020，所用第三方库为

1. tensorflow 1.13.1

2. keras 2.2.4

3. scikit-learn 0.20.3

4. numpy

5. opecv-python 4.1.0.25

6. pyqt5 5.15.0

   由于不同版本的第三方库所包含的函数名和特定字符可能有所不同，所以安装的第三方库最好为指定版本的。

   可在anaconda promt里输入

   pip install 模块名==版本 -i 下载源网站

   即可快速下载

   下载源网站可在下列几个选择

   清华大学：https://pypi.tuna.tsinghua.edu.cn/simple

   中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/

   阿里云：https://mirrors.aliyun.com/pypi/simple/

   豆瓣：http://pypi.douban.com/simple/

该程序可自己采集数据集，或自己寻找数据集

LFW数据集

链接：https://pan.baidu.com/s/1JQdlBC_Prv65ybY5f2vPeg 
提取码：yxl5 

Celeba数据集

链接：https://pan.baidu.com/s/1doLmb1SRZAvfsv07_HRoaA 
提取码：1jvr 



## 程序简介

**face detection primer**为gui界面初级版，界面如下

![image-20210201095100584](C:\Users\TuZhou\AppData\Roaming\Typora\typora-user-images\image-20210201095100584.png)

这是用Pyqt5手写的简易界面，主程序在face_detection.py中，输入名字点击开始录入可以检测采集人脸，采集人脸部分主要是使用了Opencv的级联分类器，采集的数据保存在data文件夹；采集人脸后可以训练模型，可自己收集足够多的人脸再训练模型，采用框架主要为keras版本的tensorflow；训练好的模型会保存在Model文件夹，训练模型的执行函数在face_detection.py中，其功能实现部分在face_train.py和load_dataset.py中。最后点击开始识别按钮可以开始识别，该文件夹中保存的model精度不够，可自己使用足够多的人脸训练模型。



**face detection advance**在版本一基础上优化了界面，并为训练模型模块单开线程，如下图

![image-20210201095936567](C:\Users\TuZhou\AppData\Roaming\Typora\typora-user-images\image-20210201095936567.png)

gui界面是使用qt designer设计的，主程序仍然在face_detection.py中，如下图使用了QThread为训练模型模块单开线程，如此训练模型过程中界面就不会进入“假死”状态。

![image-20210201100041841](C:\Users\TuZhou\AppData\Roaming\Typora\typora-user-images\image-20210201100041841.png)