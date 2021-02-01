

import sys
import os
import numpy
import ctypes
import cv2
from PIL import Image,  ImageDraw, ImageFont
from face_train import Model, Dataset

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget,QLabel, QLineEdit, QProgressBar, QProgressDialog,QHBoxLayout,QVBoxLayout
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt, QBasicTimer, QTimer
from PyQt5.QtGui import QFont
from PyQt5 import QtGui

#import face_train as ft





class Example(QWidget):
    
    def __init__(self):
        super().__init__()

        self.timer_camera1 = QTimer() #定义定时器，用于控制显示视频的帧率
        self.timer_camera2 = QTimer()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      #视频流       #视频流
        self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头

        
        self.initUI() #界面绘制交给InitUi方法
        

    def initUI(self):


        '''信息显示'''
        self.label_show_camera = QLabel(self)   #定义显示视频的Label
        self.label_show_camera.setFixedSize(641,481)    #给显示视频的Label设置大小为641x481
        self.label_show_camera.move(10, 10)
      


        #创建名字输入文本框
        self.name_input = QLineEdit(self)

        #创建输入名字标签
        self.name_label = QLabel(self)
        self.name_label.setText("请输入你的名字")
        self.name_label.setFont(QFont('SansSerif', 12))
        self.name_label.setAutoFillBackground(True)
        palette = QPalette()  #创建调色板
        palette.setColor(QPalette.Window, Qt.red)  #设置背景色
        self.name_label.setPalette(palette)
        self.name_label.setAlignment(Qt.AlignRight)
 
       
        self.btn1 = QPushButton('开始录入', self)
        self.btn1.clicked.connect(self.onClick)
       
        
        self.btn2 = QPushButton('开始训练', self)
        self.btn2.resize(self.btn2.sizeHint())   #设置默认大小
        self.btn2.clicked.connect(self.train_click)


        self.btn3 = QPushButton('开始识别', self)  
        self.btn3.clicked.connect(self.button_clicked_detection)

        self.btn4 = QPushButton('退出程序', self)  
        self.btn4.clicked.connect(self.quit_job)

        #设置窗口的位置和大小
        self.resize(900, 600)

        #设置控件坐标
        self.lay_widget()
 
        #使窗口处于屏幕中心
        self.center()

        #设置窗口的标题
        self.setWindowTitle('人脸识别系统')
                
        #显示窗口
        self.show()
    

    

    #退出程序
    def quit_job(self):
        sys.exit()


    #使窗口保持在屏幕中间
    def center(self):
        
        #获得窗口
        qr = self.frameGeometry()
        #获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        #显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    

    #调整控件位置
    def lay_widget(self):
       widget_x = self.x() + self.width() - 100
       widget_y = self.y() + 50
       self.name_label.move(widget_x - 65, widget_y + 20)
       self.name_input.move(widget_x - 80, widget_y + 50)
       self.btn1.move(widget_x - 65, widget_y + 80)
       self.btn2.move(widget_x, widget_y + 170)
       self.btn3.move(widget_x, widget_y + 240)
       self.btn4.move(widget_x, widget_y + self.height() - 100)



    #图片采集
    def onClick(self):
         
        #输入名字作为保存文件夹名
        new_user_name = self.name_input.text()
            

        #采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢                                                                                              #相机的ID号
        #images_num = 200                                                                                         #采集图片数量
        self.path = '.\\data\\' + new_user_name   #图像保存位置
        #打开摄像头
        self.button_open_camera_clicked()
        self.timer_camera1.timeout.connect(self.CatchPICFromVideo)



    #打开录入人脸的摄像头
    def button_open_camera_clicked(self):
        if self.timer_camera1.isActive() == False:   #若定时器未启动
            flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:       #flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera1.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.btn1.setText('关闭相机')
        else:
            self.timer_camera1.stop()  #关闭定时器
            self.cap.release()        #释放视频流
            self.label_show_camera.clear()  #清空视频显示区域
            self.name_input.clear()  #清空输入框中的内容
            self.btn1.setText('打开相机')


    
 

    #检查图片保存路径是否存在，不存在则建立
    def CreateFolder(self):
        #去除首位空格
        del_path_space = self.path.strip()
        #去除尾部'\'
        del_path_tail = del_path_space.rstrip('\\')
        #判读输入路径是否已存在
        isexists = os.path.exists(del_path_tail)
        if not isexists:
            os.makedirs(del_path_tail)
            return True
        else:
            return False


    #人脸录入
    def CatchPICFromVideo(self):

        #检查输入路径是否存在——不存在就创建
        self.CreateFolder()
        catch_pic_num = 200  #录入人脸数目，录入越多训练越慢，但更精确
        path_name = self.path  #录入的数据保存路径

        # 告诉OpenCV使用人脸识别分类器
        classfier = cv2.CascadeClassifier(r"D:\user\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")

        #识别出人脸后要画的边框的颜色，RGB格式
        color = (0, 255, 0)
        #self.ca = 1
        num = 0
        while self.cap.isOpened():
            ok, self.image = self.cap.read()  # 读取一帧数据
            if not ok:
                break

            #facecolor = self.ca
            #self.ca = 2
            #print(facecolor)
            grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
            

            # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
            if len(faceRects) > 0:  # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect
                    if w > 200:

                        # 将当前帧保存为图片
                        img_name = '%s\%d.jpg' % (path_name, num)

                        #image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        image = grey[y:y+h,x:x+w]           #保存灰度人脸图
                        cv2.imwrite(img_name, image)

                        num += 1
                        if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                            break

                        #画出矩形框的时候稍微比识别的脸大一圈
                        cv2.rectangle(self.image, (x - 10, y - 10), (x + w + 20, y + h + 20), color, 2)

                        # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(self.image, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

            # 超过指定最大保存数量结束程序
            if num > (catch_pic_num):
                self.name_input.clear()  #录入结束后清空输入框中的内容
                break
            show = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色
            #show = cv2.resize(self.image,(840,680))
            showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage

            # 显示图像
            #cv2.imshow(window_name, self.image)
            #按键盘‘Q’中断采集
            cv2.waitKey(10)
            

        # 释放摄像头并销毁所有窗口
        self.cap.release()
        cv2.destroyAllWindows()


    #人脸识别槽函数
    def button_clicked_detection(self):
        self.button_open_camera_det()
        self.timer_camera2.timeout.connect(self.face_det)


    #打开人脸识别的摄像头
    def button_open_camera_det(self):
        if self.timer_camera2.isActive() == False:   #若定时器未启动
            flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:       #flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera2.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.btn3.setText('关闭相机')
        else:
            self.timer_camera2.stop()  #关闭定时器
            self.cap.release()        #释放视频流
            self.label_show_camera.clear()  #清空视频显示区域
            self.btn3.setText('开始识别')

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


    #人脸识别
    def face_det(self):
    
    
        if len(sys.argv) != 1:
            print("Usage:%s camera_id\r\n" % (sys.argv[0]))
            sys.exit(0)

        #加载模型
        model = Model()
        model.load_model(file_path='./model/aggregate.face.model.h5')

        # 框住人脸的矩形边框颜色
        color = (0, 255, 0)

        

        # 人脸识别分类器本地存储路径
        cascade_path = r"D:\user\Software\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

        # 循环检测识别人脸
        while self.cap.isOpened():
            ret, frame = self.cap.read()  # 读取一帧视频
            #facecolor = frame

            if ret is True:

                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)

            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y: y + h, x: x + w]       #(改)
                    faceID = model.face_predict(image)
                    #print(model.)

                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    #face_id判断（改）
                    for i in range(len(os.listdir('./data/'))):
                        if i == faceID:
                            #文字提示是谁
                            cv2.putText(frame,os.listdir('./data/')[faceID],
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽
                        else:
                             文字提示是谁
                             cv2.putText(frame, 'stranger',
                                            (x + 30, y + 30),  # 坐标
                                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                            1,  # 字号
                                            (255, 0, 255),  # 颜色
                                            2)  # 字的线宽
                             frame = self.cv2ImgAddText(frame, "陌生人", x+30, y+30, (255, 0 , 255), 25)
                             frame = Image.blend(frame, img, 0.3)
                             cv2.imshow('show', img)

            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色
            #show = cv2.resize(self.image,(840,680))
            showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
            #hanziimg = QtGui.QImage(img.data,img.shape[1],img.shape[0],QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage
            #self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(hanziimg))
            #等待10毫秒看是否有按键输入,如果注释掉就无法显示视频
            cv2.waitKey(10)
            
           

        # 释放摄像头并销毁所有窗口
        self.cap.release()
        cv2.destroyAllWindows()

    #训练模型
    def train_click(self):
        #if __name__ == '__main__':

        
    
        user_num = len(os.listdir('./data/'))

        dataset = Dataset('./data/')
        self.btn2.setText('训练中')
        dataset.load()

        model = Model()
        model.build_model(dataset,nb_classes=user_num)

        # 先前添加的测试build_model()函数的代码
        model.build_model(dataset,nb_classes=user_num)
        # 测试训练函数的代码
        model.train(dataset)
    
    
        model.save_model(file_path='./model/aggregate.face.model.h5')
        self.btn2.setText('训练结束')
        #model.evaluate(dataset)


if __name__ == '__main__':
    #whnd = ctypes.windll.kernel32.GetConsoleWindow()    
    #if whnd != 0:    
    #    ctypes.windll.user32.ShowWindow(whnd, 0)    
    #    ctypes.windll.kernel32.CloseHandle(whnd) 
   
     
    #创建应用程序和对象
   
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_()) 
    