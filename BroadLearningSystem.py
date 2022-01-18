# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python.
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper,
   please contact the authors of related paper.
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time

'''
#输出训练/测试准确率
'''
def show_accuracy(predictLabel,Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))

'''
激活函数
'''
def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))

def linear(data):
    return data

def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

def relu(data):
    return np.maximum(data, 0)

def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

'''
参数压缩
'''
def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z

'''
参数稀疏化
'''
def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

import numpy as np

def BroadLearningSystem(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    # preprocessing.scale 标准化处理，与 zscore 作用一致
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)  # , with_mean = '0') #处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_Accuracy = np.zeros([1, L + 1])
    train_Recall = np.zeros([1, L + 1])
    train_Precision = np.zeros([1, L + 1])
    test_Accuracy = np.zeros([1, L + 1])
    test_Recall = np.zeros([1, L + 1])
    test_Precision = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    for i in range(N2):
        # mapped feature
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        # 压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        # 存储每个窗口的系数化权重
        Beta1OfEachWindow.append(betaOfEachWindow)        # 每个窗口的输出 T1
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow
    # 生成强化层 enhancement nodes
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = sigmoid(tempOfOutputOfEnhanceLayer * parameterOfShrink)  ####激活函数tansig relu pinv sigmoid
    # 生成最终输入 mapped nodes + enhancement nodes
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start
    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    Y_train = train_y.argmax(axis=1)
    Y_predict = OutputOfTrain.argmax(axis=1)
    train_TP = 0
    train_FP = 0
    train_TN = 0
    train_FN = 0
    for j in range(np.shape(Y_train)[0]):
        if (Y_predict[j] == 1 and Y_train[j] == 1):
            train_TP += 1
        if (Y_predict[j] == 0 and Y_train[j] == 1):
            train_FN += 1
        if (Y_predict[j] == 0 and Y_train[j] == 0):
            train_TN += 1
        if (Y_predict[j] == 1 and Y_train[j] == 0):
            train_FP += 1
    train_Accuracy = float((train_TP + train_TN) / (train_TP + train_FP + train_TN + train_FN))
    train_Recall = float(train_TP / (train_TP + train_FN))
    train_Precision = float(train_TP / (train_TP + train_FP))
    train_time = time.time()-time_start
    print('train_Accuracy     train_Recall      train_Precision')
    print('%f%% %f%% %f%%' % (train_Accuracy * 100, train_Recall * 100, train_Precision * 100))

    ##训练参数的保存
    np.save("./BLS-model/Beta1OfEachWindow.npy", Beta1OfEachWindow)  ##
    np.save("./BLS-model/minOfEachWindow.npy", minOfEachWindow)  ##
    np.save("./BLS-model/distOfMaxAndMin.npy", distOfMaxAndMin)  ##
    np.save("./BLS-model/weightOfEnhanceLayer.npy", weightOfEnhanceLayer)  ##
    np.save("./BLS-model/parameterOfShrink.npy", parameterOfShrink)  ##
    np.save("./BLS-model/OutputWeight.npy", OutputWeight)  ##
    # return train_Accuracy,train_time
    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)  # ,with_mean = True,with_std = True) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()  # 测试计时开始
    #  映射层 mapped feature
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin
    #  强化层 enhancement nodes
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = sigmoid(
        tempOfOutputOfEnhanceLayerTest * parameterOfShrink)  ####激活函数 tansig、relu pinv sigmoid
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    Plabel = OutputOfTest.argmax(axis=1)  ##返回最大值的下标
    test_time = time.time() - time_start
    Y_test = test_y.argmax(axis=1)
    Y_test_predict = OutputOfTest.argmax(axis=1)
    test_TP = 0
    test_FP = 0
    test_TN = 0
    test_FN = 0
    for j in range(np.shape(Y_test)[0]):
        if (Y_test_predict[j] == 1 and Y_test[j] == 1):
            test_TP += 1
        if (Y_test_predict[j] == 0 and Y_test[j] == 1):
            test_FN += 1
        if (Y_test_predict[j] == 0 and Y_test[j] == 0):
            test_TN += 1
        if (Y_test_predict[j] == 1 and Y_test[j] == 0):
            test_FP += 1
    test_Accuracy = float((test_TP + test_TN) / (test_TP + test_FP + test_TN + test_FN))
    test_Recall = float(test_TP / (test_TP + train_FN))
    test_Precision = float(test_TP / (test_TP + test_FP))
    print('test_Accuracy     test_Recall      test_Precision')
    print('%f%% %f%% %f%%' % (test_Accuracy * 100, test_Recall * 100, test_Precision * 100))

    return train_Accuracy, train_Recall, train_Precision, test_Accuracy, test_Recall, test_Precision, test_time, train_time


def BLStest(test_x, test_y, s, c, N1, N2, N3):
    ###加载训练参数，训练模型中包括6个矩阵
    Beta1OfEachWindow = np.load("./BLS-model/Beta1OfEachWindow.npy")
    minOfEachWindow = np.load("./BLS-model/minOfEachWindow.npy")
    distOfMaxAndMin = np.load("./BLS-model/distOfMaxAndMin.npy")
    weightOfEnhanceLayer = np.load("./BLS-model/weightOfEnhanceLayer.npy")
    parameterOfShrink = np.load("./BLS-model/parameterOfShrink.npy")
    OutputWeight = np.load("./BLS-model/OutputWeight.npy")

    #####
    L = 0
    ymin = 0
    ymax = 1
    test_Accuracyuracy = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    #####
    test_x = preprocessing.scale(test_x,
                                 axis=1)  # ,with_mean = True,with_std = True) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()  # 测试计时开始
    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin
    #  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = sigmoid(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)  ####sigmoid   tansig
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    print(OutputOfTest)  ############测试输出的结果
    Plabel = OutputOfTest.argmax(axis=1)  ##返回最大值的下标
    print(Plabel)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_Accuracyuracy[0][0] = testAcc
    test_time[0][0] = testTime

    return test_Accuracyuracy, test_time, Plabel


def BLStestone(test_x, s, c, N1, N2, N3):
    ###加载训练参数，训练模型中包括留个矩阵
    Beta1OfEachWindow = np.load("./BLS-model/Beta1OfEachWindow.npy")
    minOfEachWindow = np.load("./BLS-model/minOfEachWindow.npy")
    distOfMaxAndMin = np.load("./BLS-model/distOfMaxAndMin.npy")
    weightOfEnhanceLayer = np.load("./BLS-model/weightOfEnhanceLayer.npy")
    parameterOfShrink = np.load("./BLS-model/parameterOfShrink.npy")
    OutputWeight = np.load("./BLS-model/OutputWeight.npy")

    #####
    L = 0
    ymin = 0
    ymax = 1
    test_Accuracyuracy = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    #####
    test_x = preprocessing.scale(test_x,
                                 axis=1)  # ,with_mean = True,with_std = True) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()  # 测试计时开始
    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin
    #  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = sigmoid(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)  ####激活函数sigmoid  tansig
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    # print(OutputOfTest)############测试输出的结果
    # Plabel = OutputOfTest.argmax(axis=1)  ##返回最大值的下标
    Plabel = OutputOfTest
    # print(OutputOfTest)
    # print(Plabel)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start

    # testAcc = show_accuracy(OutputOfTest,test_y)
    # print('Testing accurate is' ,testAcc * 100,'%')
    # print('Testing time is ',testTime,'s')
    # test_Accuracyuracy[0][0] = testAcc
    # test_time[0][0] = testTime

    return test_time, Plabel


def bls_train_inputenhance(train_x, train_y, train_xf, train_yf, test_x, test_y, s, C, N1, N2, N3, l, m, m2):
    #
    # %Incremental Learning Process of the proposed broad learning system: for
    # %increment of input patterns
    # %Input:
    # %---train_x,test_x : the training data and learning data in the beginning of
    # %the incremental learning
    # %---train_y,test_y : the label
    # %---train_yf,train_xf: the whole training samples of the learning system
    # %---We: the randomly generated coefficients of feature nodes
    # %---wh:the randomly generated coefficients of enhancement nodes
    # %----s: the shrinkage parameter for enhancement nodes
    # %----C: the regularization parameter for sparse regularization
    # %----N1: the number of feature nodes  per window
    # %----N2: the number of windows of feature nodes
    # %----N3: the number of enhancements nodes
    # % ---m:number of added input patterns per incremental step
    # %----m2:number of added enhancement nodes per incremental step
    # % ----l: steps of incremental learning
    #
    # %output:
    # %---------Testing_time1:Accumulative Testing Times
    # %---------Training_time1:Accumulative Training Time
    u = 0
    ymax = 1
    ymin = 0
    train_err = np.zeros([1, l + 1])
    test_err = np.zeros([1, l + 1])
    train_time = np.zeros([1, l + 1])
    test_time = np.zeros([1, l + 1])
    l2 = []
    '''feature nodes'''
    time_start = time.time()
    train_x = preprocessing.scale(train_x, axis=1)
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
    y = np.zeros([train_x.shape[0], N2 * N1])
    beta11 = list()
    minOfEachWindow = []
    distMaxAndMin = []
    for i in range(N2):
        random.seed(i + u)
        we = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        A1 = H1.dot(we)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler2.transform(A1)
        beta1 = sparse_bls(A1, H1).T
        beta11.append(beta1)
        T1 = H1.dot(beta1)
        minOfEachWindow.append(T1.min(axis=0))
        distMaxAndMin.append(T1.max(axis=0) - T1.min(axis=0))
        T1 = (ymax - ymin) * (T1 - minOfEachWindow[i]) / distMaxAndMin[i] - ymin
        y[:, N1 * i:N1 * (i + 1)] = T1

    '''
    enhancement nodes
    '''
    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])]);
    Wh = list()
    if N1 * N2 >= N3:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    Wh.append(wh)
    T2 = H2.dot(wh)
    l2.append(s / np.max(T2))
    T2 = tansig(T2 * l2[0])
    T3 = np.hstack([y, T2])
    beta = pinv(T3, C)
    beta2 = beta.dot(train_y)
    Training_time = time.time() - time_start
    train_time[0][0] = Training_time
    print('Training has been finished!')
    print('The Total Training Time is : ', Training_time, ' seconds')
    '''
    %%%%%%%%%%%%%%%%% Training Accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    xx = T3.dot(beta2)

    TrainingAccuracy = show_accuracy(xx, train_y)
    print('Training Accuracy is : ', TrainingAccuracy * 100, ' %')
    train_err[0][0] = TrainingAccuracy
    '''
    %%%%%%%%%%%%%%%%%%%%%% Testing Process %%%%%%%%%%%%%%%%%%%
    '''
    time_start = time.time()
    test_x = preprocessing.scale(test_x, axis=1)
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])

    yy1 = np.zeros([test_x.shape[0], N2 * N1])

    for i in range(N2):
        beta1 = beta11[i]
        TT1 = HH1.dot(beta1)
        TT1 = (ymax - ymin) * (TT1 - minOfEachWindow[i]) / distMaxAndMin[i] - ymin
        yy1[:, N1 * i:N1 * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = tansig(HH2.dot(wh) * l2[0])
    TT3 = np.hstack([yy1, TT2])
    x = TT3.dot(beta2)
    TestingAccuracy = show_accuracy(x, test_y)
    '''        
    %%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    Testing_time = time.time() - time_start
    test_time[0][0] = Testing_time
    test_err[0][0] = TestingAccuracy
    print('Testing has been finished!')
    print('The Total Testing Time is : ', Testing_time, ' seconds')
    print('Testing Accuracy is : ', TestingAccuracy * 100, ' %')
    '''
    %%%%%%%%%%%%% incremental training steps %%%%%%%%%%%%%%%%%%%增强的过程
    '''

    for e in range(l):
        time_start = time.time()
        '''
        WARNING: If data comes from a single dataset, the following 'train_xx' and 'train_y1' should be reset!
        '''
        train_xx = preprocessing.scale(train_xf[(10000 + (e) * m):(10000 + (e + 1) * m), :], axis=1)
        train_y1 = train_yf[0:10000 + (e + 1) * m, :]
        Hx1 = np.hstack([train_xx, 0.1 * np.ones([train_xx.shape[0], 1])])
        yx = np.zeros([train_xx.shape[0], N1 * N2])
        for i in range(N2):
            beta1 = beta11[i]
            Tx1 = Hx1.dot(beta1)
            Tx1 = (ymax - ymin) * (Tx1 - minOfEachWindow[i]) / distMaxAndMin[i] - ymin
            yx[:, N1 * i:N1 * (i + 1)] = Tx1
        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0], 1])])
        tx22 = np.zeros([Hx2.shape[0], 0])
        for o in range(e + 1):
            wh = Wh[o]
            tx2 = Hx2.dot(wh)
            tx2 = tansig(tx2 * l2[o])
            tx22 = np.hstack([tx22, tx2])

        tx2x = np.hstack([yx, tx22])
        betat = pinv(tx2x, C)
        beta = np.hstack([beta, betat])
        T3 = np.vstack([T3, tx2x])
        y = np.vstack([y, yx])
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        if N1 * N2 >= m2:
            #            random.seed(100+e)
            wh1 = LA.orth(2 * random.randn(N2 * N1 + 1, m2) - 1)
        else:
            #            random.seed(100+e)
            wh1 = LA.orth(2 * random.randn(N2 * N1 + 1, m2).T - 1).T

        Wh.append(wh1)
        t2 = H2.dot(wh1)
        l2.append(s / np.max(t2))
        t2 = tansig(t2 * l2[e + 1])
        T3_temp = np.hstack([T3, t2])
        d = beta.dot(t2)
        c = t2 - T3.dot(d)
        if c.all() == 0:
            w = d.shape[1]
            b = np.mat(np.eye(w) + d.T.dot(d)).I.dot(d.T.dot(beta))
        else:
            b = pinv(c, C)
        beta = np.vstack([(beta - d.dot(b)), b])
        beta2 = beta.dot(train_y1)
        T3 = T3_temp
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time
        xx = T3.dot(beta2)
        TrainingAccuracy = show_accuracy(xx, train_y1)
        train_err[0][e + 1] = TrainingAccuracy
        print('Training Accuracy is : ', TrainingAccuracy * 100, ' %')
        #    %%%%%%%%%%%%%incremental testing steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        time_start = time.time()
        wh = Wh[e + 1]
        tt2 = tansig(HH2.dot(wh) * l2[e + 1])
        TT3 = np.hstack([TT3, tt2])
        x = TT3.dot(beta2)
        TestingAccuracy = show_accuracy(x, test_y)
        Testing_time = time.time() - time_start
        test_time[0][e + 1] = Testing_time
        test_err[0][e + 1] = TestingAccuracy
        print('Testing has been finished!')
        print('The Total Testing Time is : ', Testing_time, ' seconds')
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %')
    return test_err, test_time, train_err, train_time