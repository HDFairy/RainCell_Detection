import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import keras
import pickle
import PIL.Image
import random
from keras.applications.resnet50 import ResNet50
import time
from tqdm import tqdm
import cv2

def findfile(base_path, filter):
    Imglist = []
    for maindir, subdir, filename_list in os.walk(base_path):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in filename_list:
            filepath = os.path.join(maindir, filename)  # 合并成一个完整路径
            ext = os.path.splitext(filepath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
            if ext in filter:
                Imglist.append(filepath)
        for dir in subdir:
            findfile(dir, filter)

    return Imglist

def FileScaning(train_RC_path, train_NoRC_path, test_RC_path, test_NoRC_path):
    # train
    filter = '.tiff'
    train_imglist_RC = findfile(train_RC_path, filter)
    train_label_RC = np.empty((len(train_imglist_RC), 2))
    count = 0
    for i in train_imglist_RC:
        train_label_RC[count] = np.array((0, 1))
        count += 1

    train_imglist_NoRC = findfile(train_NoRC_path, filter)
    train_label_NoRC = np.empty((len(train_imglist_NoRC), 2))
    count = 0
    for j in train_imglist_NoRC:
        train_label_NoRC[count] = np.array((1, 0))
        count += 1

    # test
    test_imglist_RC = findfile(test_RC_path, filter)
    test_label_RC = np.empty((len(test_imglist_RC), 2))
    count = 0
    for i in test_imglist_RC:
        test_label_RC[count] = np.array((0, 1))
        count += 1

    test_imglist_NoRC = findfile(test_NoRC_path, filter)
    test_label_NoRC = np.empty((len(test_imglist_NoRC), 2))
    count = 0
    for j in test_imglist_NoRC:
        test_label_NoRC[count] = np.array((1, 0))
        count += 1

    return train_imglist_RC, train_imglist_NoRC, train_label_RC, train_label_NoRC, test_imglist_RC, test_imglist_NoRC, test_label_RC, test_label_NoRC

def rotateImg(img_raw):
    rows = img_raw.shape[1]
    cols = img_raw.shape[2]
    result_img = []
    for i in range(10):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), i * 5, 1)
        dest_img = cv2.warpAffine(img_raw, M, (rows, cols))
        result_img.append(dest_img)

        M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -i * 5, 1)
        dest_img2 = cv2.warpAffine(img_raw, M2, (rows, cols))
        result_img.append(dest_img2)
    return result_img

def DataSet(train_imglist_RC, train_imglist_NoRC, train_label_RC, train_label_NoRC, test_imglist_RC, test_imglist_NoRC, test_label_RC, test_label_NoRC):

    sample_train_RC = []
    sample_train_NoRC = []
    label_train_RC = []
    label_train_NoRC = []

    sample_test_RC = []
    sample_test_NoRC = []
    label_test_RC = []
    label_test_NoRC = []

    # 划分训练集
    N1 = len(train_imglist_RC)
    for i in range(N1):
        sample_train_RC.append(train_imglist_RC[i])
        label_train_RC.append(train_label_RC[i])

    N2 = len(train_imglist_NoRC)
    for i in range(N2):
        sample_train_NoRC.append(train_imglist_NoRC[i])
        label_train_NoRC.append(train_label_NoRC[i])

    N3 = len(test_imglist_RC)
    for i in range(N3):
        sample_test_RC.append(test_imglist_RC[i])
        label_test_RC.append(test_label_RC[i])

    N4 = len(test_imglist_NoRC)
    for i in range(N4):
        sample_test_NoRC.append(test_imglist_NoRC[i])
        label_test_NoRC.append(test_label_NoRC[i])

    return sample_train_RC, sample_train_NoRC, label_train_RC, label_train_NoRC, sample_test_RC , sample_test_NoRC, label_test_RC, label_test_NoRC

def datasplit(train_RC_path, train_NoRC_path, test_RC_path, test_NoRC_path):

    # 读取数据 将其 路径与label存入csv文件，便于后续取用
    train_imglist_RC, train_imglist_NoRC, train_label_RC, train_label_NoRC, test_imglist_RC, test_imglist_NoRC, test_label_RC, test_label_NoRC \
        = FileScaning(train_RC_path, train_NoRC_path, test_RC_path, test_NoRC_path)
    sample_train_RC, sample_train_NoRC, label_train_RC, label_train_NoRC, sample_test_RC, sample_test_NoRC, label_test_RC, label_test_NoRC \
        = DataSet(train_imglist_RC, train_imglist_NoRC, train_label_RC, train_label_NoRC, test_imglist_RC, test_imglist_NoRC, test_label_RC, test_label_NoRC)

    train_RC_sample, train_RC_label = extractFeature(sample_train_RC, label_train_RC)
    train_NoRC_sample, train_NoRC_label = extractFeature(sample_train_NoRC, label_train_NoRC)
    test_RC_sample, test_RC_label = extractFeature(sample_test_RC, label_test_RC)
    test_NoRC_sample, test_NoRC_label = extractFeature(sample_test_NoRC, label_test_NoRC)

    return train_RC_sample, train_RC_label, train_NoRC_sample, train_NoRC_label, test_RC_sample, test_RC_label, test_NoRC_sample, test_NoRC_label


print('------------------------------- Convolution Calculation with RseNet50--------------------------------------')
from keras.models import Sequential, Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
import cv2

def extractFeature(sample, label, resize_format=(224, 224), colorType=cv2.IMREAD_COLOR,
                   resize_interpolation=cv2.INTER_NEAREST, needGenImg=False):   ####IMREAD_COLOR   IMREAD_GRAYSCALE

    model = ResNet50(include_top=True, weights='./Parameter File/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    dense_result = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)  ####分类层的前一层


    Input_train = []  # 存储训练集图像
    Label_train = []
    print('Loading training dataset')
    for count in tqdm(range(len(sample))):
        Label_train.append(label[count])
        Img_path = sample[count]
        Img = cv2.imread(Img_path, colorType)
        Img_formated = cv2.resize(Img, resize_format, interpolation=resize_interpolation)
        Img_formated = np.expand_dims(Img_formated, axis=0)
        Img_flat = dense_result.predict(Img_formated)
        Img_flat = Img_flat.ravel() # 将数组维度变为一维
        # 训练的输入为图像，输出为分类，（0，1）是TC，（1，0）是NonTC
        Input_train.append(Img_flat)

        if needGenImg:
            genImgs = rotateImg(Img_formated)  ####通过旋转图片进行数据集增广
            for genImg in genImgs:
                Img_flat = Img_formated.ravel()
                Input_train.append(Img_flat)
                Label_train.append(label[count])

    return np.double(Input_train), np.double(Label_train)

