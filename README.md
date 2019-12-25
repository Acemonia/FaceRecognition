## 介绍
- 模型的检测与识别通过CNN实现
- 界面用Python的GUI工具集Tkinter实现
- 用到了一些OpenCV的API

## 功能
- 摄像头识别
- 图片识别
- 关键点检测

## 不足
- 模型精度低, 识别率较差
- 由于Tkinter, 存在一定的Bug, 未响应. 
- 由于Tkinter技术受限, "选择摄像头"后仅单次拍照保存
- 代码命名和结构不太规范

## 后期
- 更换模型
- 改成Web系统
- 转移到Linux平台开发
- 改成摄像头录像实时识别

## 文件说明
由于太大未上传
- CNN目录下的model目录
- interface目录下的candidate-faces目录
- interface目录下的两个.dat文件

## 运行方式
1. 配置Python环境安装库(模型已经训练完成)
2. 运行Interface/GUI_test.py

## 界面
![界面图片](https://github.com/Acemonia/FaceRecognition/raw/master/Interface/interface_image.jpg)

左图为摄像头拍摄, 右图为数据库的图
