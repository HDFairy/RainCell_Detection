from BroadLearningSystem import *
from Functions import *
from numpy import random


'''
增加强化层节点

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''
def BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0
    ymax = 1  # 数据收缩上限
    ymin = 0  # 数据收缩下限
    train_x = preprocessing.scale(train_x, axis=1)  # 处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    #    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_Accuracy = np.zeros([1, L + 1])
    train_Recall = np.zeros([1, L + 1])
    train_Precision = np.zeros([1, L + 1])
    test_Accuracy = np.zeros([1, L + 1])
    test_Recall = np.zeros([1, L + 1])
    test_Precision = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1  # 生成每个窗口的权重系数，最后一行为偏差
        #        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)  # 生成每个窗口的特征
        # 压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

        # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    print('---------------- initinal training -------------------')
    Y_train = train_y.argmax(axis=1)
    Y_predict = OutputOfTrain.argmax(axis=1)
    train_11 = 0
    train_10 = 0
    train_00 = 0
    train_01 = 0
    total_train = 0
    for j in range(np.shape(OutputOfTrain)[0]):
        if (Y_predict[j] == Y_train[j]):
            total_train += 1
        if (Y_predict[j] == 1 and Y_train[j] == 1):
            train_11 += 1
        if (Y_predict[j] == 0 and Y_train[j] == 1):
            train_10 += 1
        if (Y_predict[j] == 0 and Y_train[j] == 0):
            train_00 += 1
        if (Y_predict[j] == 1 and Y_train[j] == 0):
            train_01 += 1
    train_acc_total = float(total_train / np.shape(OutputOfTrain)[0])
    train_HR = float(train_11 / (train_11 + train_10))
    train_FAR = float(train_01 / (train_00 + train_01))
    train_Accuracy[0][0] = train_acc_total
    train_Recall[0][0] = train_HR
    train_Precision[0][0] = train_FAR
    print('train_acc_total     train_HR      train_FAR')
    print('%f%% %f%% %f%%' % (train_acc_total * 100, train_HR * 100, train_FAR * 100))

    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)  # 处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
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
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    print('---------------- initinal testing -------------------')
    Y_test = test_y.argmax(axis=1)
    Y_test_predict = OutputOfTest.argmax(axis=1)
    test_11 = 0
    test_10 = 0
    test_00 = 0
    test_01 = 0
    total_test = 0
    for j in range(np.shape(OutputOfTest)[0]):
        if (Y_test_predict[j] == Y_test[j]):
            total_test += 1
        if (Y_test_predict[j] == 1 and Y_test[j] == 1):
            test_11 += 1
        if (Y_test_predict[j] == 0 and Y_test[j] == 1):
            test_10 += 1
        if (Y_test_predict[j] == 0 and Y_test[j] == 0):
            test_00 += 1
        if (Y_test_predict[j] == 1 and Y_test[j] == 0):
            test_01 += 1
    test_acc_total = float(total_test / np.shape(OutputOfTest)[0])
    test_HR = float(test_11 / (test_11 + test_10))
    test_FAR = float(test_01 / (test_00 + test_01))
    test_Accuracy[0][0] = test_acc_total
    test_Recall[0][0] = test_HR
    test_Precision[0][0] = test_FAR
    print('test_acc_total     test_HR      test_FAR')
    print('%f%% %f%% %f%%' % (test_acc_total * 100, test_HR * 100, test_FAR * 100))
    '''
        增量增加强化节点
    '''
    print('---------------- Adding enhancement nodes -------------------')
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M).T - 1).T

        #        WeightOfEnhanceLayerAdd[e,:,:] = weightOfEnhanceLayerAdd
        #        weightOfEnhanceLayerAdd = weightOfEnhanceLayer[:,N3+e*M:N3+(e+1)*M]
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)

        # 增量增加节点的 测试过程
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)

        Y_test = test_y.argmax(axis=1)
        Y_test_predict = OutputOfTest1.argmax(axis=1)
        test_11 = 0
        test_10 = 0
        test_00 = 0
        test_01 = 0
        total_test = 0
        for j in range(np.shape(OutputOfTest1)[0]):
            if (Y_test_predict[j] == Y_test[j]):
                total_test += 1
            if (Y_test_predict[j] == 1 and Y_test[j] == 1):
                test_11 += 1
            if (Y_test_predict[j] == 0 and Y_test[j] == 1):
                test_10 += 1
            if (Y_test_predict[j] == 0 and Y_test[j] == 0):
                test_00 += 1
            if (Y_test_predict[j] == 1 and Y_test[j] == 0):
                test_01 += 1
        test_acc_total = float(total_test / np.shape(OutputOfTest1)[0])
        test_HR = float(test_11 / (test_11 + test_10))
        test_FAR = float(test_01 / (test_00 + test_01))

        test_Accuracy[0][0] = test_acc_total
        test_Recall[0][0] = test_HR
        test_Precision[0][0] = test_FAR

        print('test_acc_total     test_HR      test_FAR')
        print('%f%% %f%% %f%%' % (test_acc_total * 100, test_HR * 100, test_FAR * 100))

    return test_Accuracy, test_Recall, test_Precision


'''
增加映射层和强化层节点

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''
def BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3):
    u = 0
    ymax = 1
    ymin = 0
    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
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
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1;  # 生成每个窗口的权重系数，最后一行为偏差
        #        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)  # 生成每个窗口的特征
        # 压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.mean(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow
        # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    # 生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain, c)
    OutputWeight = pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start
    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayerTrain, OutputWeight)
    print('---------------- initinal training -------------------')
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
    train_Accuracy[0][0] = float((train_TP + train_TN) / (train_TP + train_FP + train_TN + train_FN))
    train_Recall[0][0] = float(train_TP / (train_TP + train_FN))
    train_Precision[0][0] = float(train_TP / (train_TP + train_FP))
    train_time[0][0] = time.time() - time_start
    print('train_Accuracy     train_Recall      train_Precision')
    print('%f%% %f%% %f%%' % (train_Accuracy[0][0] * 100, train_Recall[0][0] * 100, train_Precision[0][0] * 100))

    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)  # 处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
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
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    print('---------------- initinal testing -------------------')
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
    test_Accuracy[0][0] = float((test_TP + test_TN) / (test_TP + test_FP + test_TN + test_FN))
    test_Recall[0][0] = float(test_TP / (test_TP + train_FN))
    test_Precision[0][0] = float(test_TP / (test_TP + test_FP))
    print('test_Accuracy     test_Recall      test_Precision')
    print('%f%% %f%% %f%%' % (test_Accuracy[0][0] * 100, test_Recall[0][0] * 100, test_Precision[0][0] * 100))
    '''
        增加Mapping feature 和 相应的强化节点
    '''
    print('---------------- Adding mapping feature nodes and enhancement nodes -------------------')
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e + N2 + u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1] + 1, M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)
        # FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        # 压缩每个窗口特征到[-1，1]
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)

        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append(np.max(TempOfFeatureOutput, axis=0) - np.min(TempOfFeatureOutput, axis=0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput, axis=0))
        outputOfNewWindow = (TempOfFeatureOutput - minOfEachWindow[N2 + e]) / distOfMaxAndMin[N2 + e]
        # 新的映射层整体输出
        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer, outputOfNewWindow])
        # 新增加映射窗口的输出带偏置
        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0], 1))])
        # 新映射窗口对应的强化层节点，M2列
        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]) - 1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]).T - 1).T
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)

        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)

        parameter1 = s / np.max(tempOfNewFeatureEhanceNodes)
        # 与新增的Feature Mapping 节点对应的强化节点输出
        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2 * N1 + (e+1) * M1 >= M3:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3) - 1)
        else:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3).T - 1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)
        # 整体映射层输出带偏置
        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s / np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)
        OutputOfTotalNewAddNodes = np.hstack(
            [outputOfNewWindow, outputOfNewFeatureEhanceNodes, OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain, OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)

        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w) - D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeight = pinvOfInput.dot(train_y)
        InputOfOutputLayerTrain = tempOfInputOfLastLayes

        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e + 1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        Y_train = train_y.argmax(axis=1)
        Y_predict = predictLabel.argmax(axis=1)
        train_TP = 0
        train_FP = 0
        train_TN = 0
        train_FN = 0
        for j in range(np.shape(Y_train)[0]):
            if (Y_predict[j] == 1 and Y_train[j] == 1):
                train_TP += 1
            if (Y_predict[j] == 0 and Y_train[j] == 1):
                train_FP += 1
            if (Y_predict[j] == 0 and Y_train[j] == 0):
                train_TN += 1
            if (Y_predict[j] == 1 and Y_train[j] == 0):
                train_FN += 1
        train_Accuracy[0][e + 1] = float((train_TP + train_TN) / (train_TP + train_FP + train_TN + train_FN))
        train_Recall[0][e + 1] = float(train_TP / (train_TP + train_FN))
        train_Precision[0][e + 1] = float(train_TP / (train_TP + train_FP))
        print('train_Accuracy     train_Recall      train_Precision')
        print('%f%% %f%% %f%%' % (
        train_Accuracy[0][e + 1] * 100, train_Recall[0][e + 1] * 100, train_Precision[0][e + 1] * 100))
        # 测试过程
        # 先生成新映射窗口输出
        time_start = time.time()
        WeightOfNewMapping = Beta1OfEachWindow[N2 + e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping)
        # TT1
        outputOfNewWindowTest = (ymax - ymin) * (outputOfNewWindowTest - minOfEachWindow[N2 + e]) / distOfMaxAndMin[
            N2 + e] - ymin
        ## 整体映射层输出
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, outputOfNewWindowTest])
        # HH2
        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones([OutputOfFeatureMappingLayerTest.shape[0], 1])])
        # hh2
        NewInputOfEnhanceLayerWithBiasTest = np.hstack(
            [outputOfNewWindowTest, 0.1 * np.ones([outputOfNewWindowTest.shape[0], 1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        # tt22
        OutputOfRelateEnhanceNodes = tansig(
            NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        #
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        # tt2
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes) * parameter2)

        InputOfOutputLayerTest = np.hstack(
            [InputOfOutputLayerTest, outputOfNewWindowTest, OutputOfRelateEnhanceNodes, OutputOfNewEnhanceNodes])

        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        Y_test = test_y.argmax(axis=1)
        Y_test_predict = predictLabel.argmax(axis=1)
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
        test_Accuracy[0][e + 1] = float((test_TP + test_TN) / (test_TP + test_FP + test_TN + test_FN))
        test_Recall[0][e + 1] = float(test_TP / (test_TP + test_FN))
        test_Precision[0][e + 1] = float(test_TP / (test_TP + test_FP))
        print('test_Accuracy     test_Recall      test_Precision    test_TP ')
        print('%f%% %f%% %f%%' % (
        test_Accuracy[0][e + 1] * 100, test_Recall[0][e + 1] * 100, test_Precision[0][e + 1] * 100))

    return test_Accuracy, test_Recall, test_Precision

def Add_EnhanceNodes(s, c, N1, N2, N3, L, M):
    N1 = 2  # # of nodes belong to each window 5
    N2 = 2  # # of windows -------Feature mapping layer
    N3 = 2  # # of enhancement nodes -----Enhance layer
    M = 2
    L = 10
    s = 0.8  # shrink coefficient
    c = 2 ** -30  # Regularization coefficient

    print('------------------------------ Nodes Incremental learning ---------------------------------')
    train_x = np.load("./ExtractData/Input_train.npy")
    train_y = np.load("./ExtractData/Label_train.npy")
    test_x = np.load("./ExtractData/Input_test.npy")
    test_y = np.load("./ExtractData/Label_test.npy")

    print('------------------------------ Add enhancement node ---------------------------------------')
    test_Accuracy, test_Recall, test_Precision = BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M)

def Add_FeatureEnhanceNodes(s, c, N1, N2, N3, L, M):

    M1 = 1 # 增加的mapping feature的个数
    M2 = 1 # 与mapping feature 相对应的增强节点个数
    M3 = 2 # 新增加的强化节点
    L = 40

    print('------------------------------ Nodes Incremental learning ---------------------------------')
    train_x = np.load("./ExtractData/Input_train.npy")
    train_y = np.load("./ExtractData/Label_train.npy")
    test_x = np.load("./ExtractData/Input_test.npy")
    test_y = np.load("./ExtractData/Label_test.npy")

    print('------------------------------ Add mapping feature nodes and enhancement nodes ---------------------------------------')
    test_Accuracy, test_Recall, test_Precision = BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3)