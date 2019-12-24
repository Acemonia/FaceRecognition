#coding=utf-8
#tensorflow 1.3.1
#python 3.6
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy
from PIL import Image,ImageTk  
import random
import tkinter as tk 

#获取dataset
def load_data(dataset_path):
    # 读取图片
    img = Image.open(dataset_path)
    # 定义一个20 × 20的训练样本，一共有40个人，每个人都10张样本照片，将图片转化为矩阵
    img_ndarray = np.asarray(img, dtype='float64') / 256
    #img_ndarray = np.asarray(img, dtype='float32') / 32

    # 记录脸数据矩阵，57 * 47为每张脸的像素矩阵
    faces = np.empty((400, 57 * 47))

    for row in range(20):
        for column in range(20):
            faces[20 * row + column] = np.ndarray.flatten(
                img_ndarray[row * 57: (row + 1) * 57, column * 47 : (column + 1) * 47]
            )

    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1

    # 将数据分成训练集，验证集，测试集
    train_data = np.empty((320, 57 * 47))
    train_label = np.zeros((320, 40))
    vaild_data = np.empty((40, 57 * 47))
    vaild_label = np.zeros((40, 40))
    test_data = np.empty((40, 57 * 47))
    test_label = np.zeros((40, 40))

    for i in range(40):
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]

        vaild_data[i] = faces[i * 10 + 8]
        vaild_label[i] = label[i * 10 + 8]

        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    vaild_data = vaild_data.astype('float32')
    test_data = test_data.astype('float32')

    return [
        (train_data, train_label),
        (vaild_data, vaild_label),
        (test_data, test_label)
    ]

def convolutional_layer(data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("conv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias', bias_size, initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')
    linear_output = tf.nn.relu(tf.add(conv, bias))
    # ksize = [1, height, width, 1], stride = [1, stride,stride, 1]
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    return pooling

def linear_layer(data, weights_size, biases_size):
    weights = tf.get_variable("weigths", weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_size, initializer=tf.random_normal_initializer())
    return tf.add(tf.matmul(data, weights), biases)

def convolutional_neural_network(data):
    # 根据类别个数定义最后输出层的神经元
    n_ouput_layer = 40

    # [filter_height, filter_width, in_channels, out_channels]
    kernel_shape1=[5, 5, 1, 32]
    kernel_shape2=[5, 5, 32, 64]
    full_conn_w_shape = [15 * 12 * 64, 1024]
    out_w_shape = [1024, n_ouput_layer]

    bias_shape1=[32]
    bias_shape2=[64]
    full_conn_b_shape = [1024]
    out_b_shape = [n_ouput_layer]

    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("conv_layer1") as layer1:
        layer1_output = convolutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=[1, 2, 2, 1]
        )
    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("conv_layer2") as layer2:
        layer2_output = convolutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=[1, 2, 2, 1]
        )
    with tf.variable_scope("full_connection") as full_layer3:
        # 讲卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        layer3_output = tf.nn.relu(
            linear_layer(
                data=layer2_output_flatten,
                weights_size=full_conn_w_shape,
                biases_size=full_conn_b_shape
            )
        )
        # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    with tf.variable_scope("output") as output_layer4:
        output = linear_layer(
            data=layer3_output,
            weights_size=out_w_shape,
            biases_size=out_b_shape
        )
    writer = tf.summary.FileWriter("./path", tf.get_default_graph())
    writer.close()
    
    return output;

def train_facedata(dataset, model_dir,model_path):

    batch_size = 40

    train_set_x = dataset[0][0]
    train_set_y = dataset[0][1]
    valid_set_x = dataset[1][0]
    valid_set_y = dataset[1][1]
    test_set_x = dataset[2][0]
    test_set_y = dataset[2][1]

    X = tf.placeholder(tf.float32, [batch_size, 57 * 47])
    Y = tf.placeholder(tf.float32, [batch_size, 40])

    predict = convolutional_neural_network(X)

    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

    # 用于保存训练的最佳模型
    saver = tf.train.Saver()
    #model_dir = './model'
    #model_path = model_dir + '/best.ckpt'
    with tf.Session() as session:
        # 若不存在模型数据，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            session.run(tf.global_variables_initializer())
            best_loss = float('Inf')
            for epoch in range(20):
                epoch_loss = 0
                for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                    x = train_set_x[i * batch_size: (i + 1) * batch_size]
                    y = train_set_y[i * batch_size: (i + 1) * batch_size]
                    _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                    epoch_loss += cost

                print(epoch, ' : ', epoch_loss)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                        print("create the directory: %s" % model_dir)
                    save_path = saver.save(session, model_path)
                    print("Model saved in file: %s" % save_path)

        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        return test_pred


def DistillPic():
    # 提取图片并显示
    w_box = 400
    h_box = 400

    dataset_path = "./GraySets.jpg"

    img_open = Image.open(dataset_path) 
    #获取图像的原始大小
    w, h = img_open.size
    print(w) 
    print(h)
    img_open_resized = Resize(w, h, w_box, h_box, img_open)
    img=ImageTk.PhotoImage(img_open_resized)  
    # l1.config(image=img)  
    # l1.image=img #keep a reference

    result.set(str(random.randint(0,39)))
    c1.delete('all')
    c1.create_image(0, 0, image = img)    

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




dataset_path = "olivettifaces.gif"

data = load_data(dataset_path)
model_dir = './model'
model_path = model_dir + '/best.ckpt'

# 返回测试集结果y
test_result = train_facedata(data, model_dir, model_path)
np.savetxt("test_result.txt", test_result);

 
