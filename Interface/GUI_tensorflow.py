from tkinter import *  
from tkinter.filedialog import askopenfilename  
from PIL import Image,ImageTk  
from skimage import io
import sys,os,dlib,glob
import numpy as np
import random
img = None
img1 = None
img2 = None

# 检测人脸
def DetectFace(path):
    dist = ComputeDistence(path)
    # 候选人名单
    global faces_folder_home_path
    candidate = TraverseFolder(faces_folder_home_path)
    # 候选人和距离组成一个dict
    c_d = dict(zip(candidate,dist))
    # print(c_d)
    cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
    print("\n The person is: ",cd_sorted[0][0])
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


# 选择文件
def ChoosePic(): 
    global img, img1, img2, result1, result2, predict_num
    w_box = 400
    h_box = 200

    path_='./olivettifaces.gif' 
    true_face_num = random.randint(0,39)
    test_face_num = predict_num[true_face_num]

    row_start_train = true_face_num % 20 * 57
    column_start_train = int(true_face_num / 20) * 470

    row_start_test = test_face_num % 20 * 57
    column_start_test = int(test_face_num / 20) * 47 + 47 * 8 + int(test_face_num / 20) * 470

    img_open = Image.open(path_)

    img_crop1 = img_open.crop((column_start_test, row_start_test, column_start_test + 47, row_start_test + 57))
    #获取图像的原始大小
    w1, h1 = img_crop1.size 
    img_open_resized1 = Resize(w1, h1, w_box, h_box, img_crop1)
    img1=ImageTk.PhotoImage(img_open_resized1)  
    # l1.config(image=img)  
    # l1.image=img #keep a reference 
    c1.delete('all')
    c1.create_image(0, 0, anchor = NW, image = img1)    
    # c1.update_idletasks()

    img_crop2 = img_open.crop((column_start_train, row_start_train, column_start_train + 47*8, row_start_train + 57))
    #获取图像的原始大小
    w2, h2 = img_crop2.size 
    img_open_resized2 = Resize(w2, h2, w_box, h_box, img_crop2)
    img2=ImageTk.PhotoImage(img_open_resized2)  
    # l1.config(image=img)  
    # l1.image=img #keep a reference 
    c2.delete('all')
    c2.create_image(0, 0, anchor = NW, image = img2)  

    result1.set(str(predict_num[true_face_num] + 1))
    result2.set(str(true_face_num + 1))

    if true_face_num == predict_num[true_face_num]:
        result_img_path = './true.jpg'
    else:
        result_img_path = './false.jpg'

    img_open = Image.open(result_img_path) 
    w, h = img_open.size 
    img_open_resized = Resize(w, h, 200, 200, img_open)
    img=ImageTk.PhotoImage(img_open_resized)  
    c3.delete('all')
    c3.create_image(0, 0, anchor = NW, image = img)  
  
    # c2.update_idletasks()



root=Tk() 
root.geometry('400x500')

m_frame1 = Frame(root)
m_frame1.pack()

result1 = StringVar()
result2 = StringVar()   
# l1=Label(root)  
# l1.pack() 
c1 = Canvas(m_frame1, width = 200, height = 200, bg = 'light gray')
c1.grid(row = 0, column = 0)

c3 = Canvas(m_frame1, width = 200, height = 200)
c3.grid(row = 0, column = 1)



c2 = Canvas(root, width = 400, height = 100, bg = 'light gray')
c2.pack()

m_frame2 = Frame(root)
m_frame2.pack()
 

predict_num = np.loadtxt('test_result.txt')
Label(m_frame2, text = "Predict result: He/she's number is ", justify=RIGHT).grid(row = 0, column = 0, sticky = W)
Label(m_frame2, textvariable = result1).grid(row = 0, column = 1)

Label(m_frame2, text = "He/She's true number is", justify=LEFT).grid(row = 1, column = 0, sticky = W)
Label(m_frame2, textvariable = result2).grid(row = 1, column = 1)

Button(m_frame2,text='选择图片',command=ChoosePic).grid(row = 2, column = 0) 
root.mainloop()  

