import time
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image,ImageTk
from skimage import io
import sys,os,dlib,glob, cv2
import numpy as np
import shutil
img = None
# 遍历文件夹，得到文件夹名

def TraverseFolder(f):
    folder_name = []
    folder_path = os.listdir(f)
    for f1 in folder_path:
        tmp_path = os.path.join(f,f1)
        if os.path.isdir(tmp_path):
            folder_name.append(f1)
    return folder_name

# 计算得出人脸向量

def GetFaceVector(path):
    global facerec
    img = io.imread(path)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        # ##########################这一行语句不明白
        shape = sp(img, d)
        ############################################
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor)

    return d_test

# 计算欧氏距离

def ComputeDistence(path):
    dist = []
    d_test = GetFaceVector(path)
    descriptors = np.load('model.npy')
    for i in descriptors:
        dist_ = np.linalg.norm(i-d_test)
        dist.append(dist_)
    return dist

# 得到人脸坐标以及关键点
def GetFaceLocationAndLandmark(w, h, w_box, h_box, path):
    global detector
    img = io.imread(path)
    # 特征提取器的实例化
    dets = detector(img)

    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    # print("人脸数：", len(dets))

    location = []
    # 输出人脸矩形拉伸后的四个坐标点
    for i, d in enumerate(dets):
        shape = sp(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()]) * factor
        location.append(d.top() * factor)
        location.append(d.bottom() * factor)
        location.append(d.left() * factor)
        location.append(d.right() * factor)
        return location, landmark.tolist()

# 检测人脸
def DetectFace(path):
    dist = ComputeDistence(path)
    # 候选人名单
    global faces_folder_home_path
    candidate = TraverseFolder(faces_folder_home_path)
    # 候选人和距离组成一个dict
    c_d = dict(zip(candidate,dist))
    cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
    # 标准为0.4
    print(cd_sorted)
    if cd_sorted[0][1] < 0.5:
        print("The person is: ",cd_sorted[0][0], "\n ")
    else:
        return "null"
    return cd_sorted[0][0]


def Resize(w, h, w_box, h_box, pil_image):
    # 对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例

    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def CameraToImg():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    User = "./candidate-faces/User"
    shutil.rmtree(User)
    os.makedirs(User)
    count = 0

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        path = User + "/%d.jpg"
        cv2.imwrite(path % count, test_img)  # save frame as JPG file
        cv2.waitKey(1)
        count += 1
        resized_img = cv2.resize(test_img, (700, 500))
        cv2.imshow('Face Detection', resized_img)

        if count >= 50 or cv2.getWindowProperty('Face Detection', cv2.WND_PROP_AUTOSIZE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

def ChooseCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    ret, test_img = cap.read()
    cv2.imwrite("./tmp.jpg", test_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    cap.release()
    cv2.destroyAllWindows()

    path = "./tmp.jpg"
    showPic(path)

# 选择文件
def ChoosePic():
    path = askopenfilename()
    showPic(path)

def showPic(path):
    global img
    w_box = 400
    h_box = 400

    face_path.set(path)
    img_open = Image.open(e1.get())
    print(e1.get())
    print(img_open)
    #获取图像的原始大小
    w, h = img_open.size
    img_open_resized = Resize(w, h, w_box, h_box, img_open)
    img=ImageTk.PhotoImage(img_open_resized)
    # print(img)

    path2 = DetectPic()
    showFigure(img, path2)
    KeyPointOfPic()
    DetectPic()

def showFigure(img, path2):
    w_box = 400
    h_box = 400
    if path2 != "null":
        img_open = Image.open("./candidate-faces/" + path2 + "/1.jpg")
        w, h = img_open.size
        img_open_resized = Resize(w, h, w_box, h_box, img_open)
        global img2
        img2 = ImageTk.PhotoImage(img_open_resized)
        c.create_image(400, 0, anchor=NW, image=img2)

    c.create_image(0, 0, anchor=NW, image=img)
    c.update_idletasks()


# 检测图片
def DetectPic():
    w_box = 400
    h_box = 400
    path = e1.get()
    img_open = Image.open(e1.get())
    #获取图像的原始大小
    w, h = img_open.size
    location,landmark = GetFaceLocationAndLandmark(w, h, w_box, h_box, path)
    c.create_rectangle(location[2],location[0],location[3],location[1],outline='red',tags = 'FaceRec')
    c.delete('KeyPoint')

    global result
    path = face_path.get()
    temp = DetectFace(path)
    result.set(str(temp))
    return str(temp)

# 关键点检测
def KeyPointOfPic():
    w_box = 400
    h_box = 400
    path = e1.get()
    img_open = Image.open(e1.get())
    #获取图像的原始大小
    w, h = img_open.size
    location,landmark = GetFaceLocationAndLandmark(w, h, w_box, h_box, path)
    c.delete('FaceRec')
    for i in range(len(landmark)):
        c.create_rectangle(landmark[i][0], landmark[i][1], landmark[i][0] + 1, landmark[i][1] + 1,
            outline = 'blue', fill="blue", tags = 'KeyPoint')


# 1.68点人脸识别模型
predictor_path = './shape_predictor_68_face_landmarks.dat'
print('已加载68点人脸关键点模型')
# 2.人脸识别模型
face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'
print('已加载人脸识别模型')
# 1.加载正脸检测器face_rec_model_path
detector = dlib.get_frontal_face_detector()
print('已加载正脸检测器')
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
print('已加载人脸关键点检测器')
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# 4.候选人脸文件夹
faces_folder_home_path = './candidate-faces'


root=Tk()
root.geometry('800x500')
face_path=StringVar()
figure_path=StringVar()
result = StringVar()
# l1=Label(root)
# l1.pack()
c = Canvas(root, width = 800, height = 400, bg = 'gray')
c.pack()

e1=Entry(root, state='readonly', text=face_path)
e2=Entry(root, state='readonly', text=figure_path)
e1.pack()
e2.pack()

m_frame = Frame(root)
m_frame.pack()
Label(m_frame, text = 'This is ').grid(row = 0, column = 0)
Label(m_frame, textvariable = result).grid(row = 0, column = 1)
Button(m_frame,text='选择摄像头',command=ChooseCamera).grid(row = 1, column = 0)
Button(m_frame,text='选择图片',command=ChoosePic).grid(row = 1, column = 1)
Button(m_frame,text='识别图片',command=DetectPic).grid(row = 1, column = 2)
Button(m_frame,text='连续拍摄',command=CameraToImg).grid(row = 1, column = 3)
Button(m_frame,text='关键点检测',command=KeyPointOfPic).grid(row = 1, column = 4)
root.mainloop()

