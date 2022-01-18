from BroadLearningSystem import *
from Functions import *
from numpy import random
from Add_Nodes import *

'''
增加训练数据
'''
def bls_train_input(train_RC_sample, train_RC_label, train_NoRC_sample, train_NoRC_label,Input_test, Label_test, s, c, N1, N2, N3, l):
    # %---initial_train_x,initial_train_y : the training data and learning data in the beginning 刚开始时候的输入数据
    # incremental learning 增量学习
    # %---N_step_train,N_step_test : 每一步增加的输入样本及其标签
    # %---Input_test,Label_test: 每一步使用同一个测试集数据
    # %---We: the randomly generated coefficients of feature nodes随机产生的特征节点系数
    # %---wh:the randomly generated coefficients of enhancement nodes随机产生的增强节点系数
    # %----s: the shrinkage parameter for enhancement nodes增强节点的收缩系数
    # %----C: the regularization parameter for sparse regularization正则化系数
    # %----N1: the number of feature nodes  per window每一个窗口内的特征参数
    # %----N2: the number of windows of feature nodes窗口数
    # %----N3: the number of enhancements nodes增强节点数
    # % ---m:number of added input patterns per increment step每个增量步骤添加的输入样本数量
    # % ---l: steps of incremental learning###增量步骤
    u = 0  # random seed
    ymin = 0
    ymax = 1
    train_Accuracy = np.zeros([1, l + 1])
    train_Recall = np.zeros([1, l + 1])
    train_Precision = np.zeros([1, l + 1])
    test_Accuracy = np.zeros([1, l + 1])
    test_Recall = np.zeros([1, l + 1])
    test_Precision = np.zeros([1, l + 1])
    train_time = np.zeros([1, l + 1])
    test_time = np.zeros([1, l + 1])
    minOfEachWindow = []
    distMaxAndMin = []
    beta11 = list()
    Wh = list()

    Number_train = np.zeros([1, l + 1])
    '''
    feature nodes
    '''
    len_train_RC = len(train_RC_sample)
    M1 = int(len_train_RC * 0.1) + 1 # 每一次增加的RC样本
    len_initial_train_RC = int(len_train_RC * 0.7) + 1
    len_train_NoRC = len(train_NoRC_sample)
    M2 = int(len_train_NoRC * 0.1) + 1 # # 每一次增加的NoRC样本
    len_initial_train_NoRC = int(len_train_NoRC * 0.7) + 1

    initial_train_x = np.vstack((train_RC_sample[0:len_initial_train_RC, :], train_NoRC_sample[0:len_initial_train_NoRC, :]))
    initial_train_y = np.vstack((train_RC_label[0:len_initial_train_RC, :], train_NoRC_label[0:len_initial_train_NoRC, :]))

    # BroadLearningSystem(initial_train_x, initial_train_y, Input_test, Label_test, s, c, N1, N2, N3)

    time_start = time.time()
    train_x = preprocessing.scale(initial_train_x, axis=1)  # 训练数据进行预处理
    # 将输入 X 与偏差 beta 堆叠成一个矩阵 实现 X * W + beta
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])  ###产生一个随机权重矩阵，后边添加一列（偏置）为了便于计算
    y = np.zeros([train_x.shape[0], N2 * N1])  ####
    for i in range(N2):
        random.seed(i + u)
        we = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        A1 = H1.dot(we)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler2.transform(A1)
        beta1 = sparse_bls(A1, H1).T
        beta11.append(beta1)    # 生成mapped feature 的权重

        T1 = H1.dot(beta1)
        minOfEachWindow.append(T1.min(axis=0))
        distMaxAndMin.append(T1.max(axis=0) - T1.min(axis=0))
        T1 = (T1 - minOfEachWindow[i]) / distMaxAndMin[i]
        y[:, N1 * i:N1 * (i + 1)] = T1
        del T1
        del A1
        del we
    '''
    enhancement nodes
    '''
    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
    if N1 * N2 >= N3:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    Wh.append(wh)
    T2 = H2.dot(wh)
    parameter = s / np.max(T2)
    T2 = sigmoid(T2 * parameter)  ## tansig   sigmoid
    T3 = np.hstack([y, T2])
    beta = pinv(T3, c)
    beta2 = beta.dot(initial_train_y)
    Training_time = time.time() - time_start
    train_time[0][0] = Training_time
    print('Training has been finished!')
    print('The initial Training Time is : ', Training_time, ' seconds')
    xx = T3.dot(beta2)

    print('---------------- initinal training -------------------')
    Y_train = initial_train_y.argmax(axis=1)
    Y_predict = xx.argmax(axis=1)
    train_TP = 0
    train_FP = 0
    train_TN = 0
    train_FN = 0
    for j in range(np.shape(xx)[0]):
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
    Number_train[0][0] = len(initial_train_x)
    print('train_Accuracy     train_Recall      train_Precision')
    print('%f%% %f%% %f%%' % (train_Accuracy[0][0] * 100, train_Recall[0][0] * 100, train_Precision[0][0] * 100))

    print('---------------- initinal testing -------------------')
    test_x = preprocessing.scale(Input_test, axis=1)
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
    yy1 = np.zeros([test_x.shape[0], N2 * N1])
    for i in range(N2):
        beta1 = beta11[i]
        TT1 = HH1.dot(beta1)
        TT1 = (ymax - ymin) * (TT1 - minOfEachWindow[i]) / distMaxAndMin[i] - ymin
        yy1[:, N1 * i:N1 * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = sigmoid(HH2.dot(wh) * parameter)  #######tansig  sigmoid
    TT3 = np.hstack([yy1, TT2])
    x = TT3.dot(beta2)
    Y_test = Label_test.argmax(axis=1)
    Y_test_predict = x.argmax(axis=1)
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
    incremental training steps,前边都是比较常规的计算，后边是数据增加时候的计算内容
    '''
    print('--------------- Incremental Learning -----------------')
    train_y = initial_train_y
    for e in range(l):
        time_start1 = time.time()
        '''
        WARNING: If data comes from a single dataset, the following 'train_x_add' and 'train_y_add' should be reset!
        '''
        train_x_RC_add = train_RC_sample[(len_initial_train_RC+(e)*M1):(len_initial_train_RC+(e+1)*M1), :]
        train_x_NoRC_add = train_NoRC_sample[(len_initial_train_NoRC+(e)*M2):(len_initial_train_NoRC+(e+1)*M2), :]
        train_x_add = preprocessing.scale(np.vstack((train_x_RC_add, train_x_NoRC_add)), axis=1)
        train_y_RC_add = train_RC_label[(len_initial_train_RC+(e)*M1):(len_initial_train_RC+(e+1)*M1), :]
        train_y_NoRC_add = train_NoRC_label[(len_initial_train_NoRC+(e)*M2):(len_initial_train_NoRC+(e+1)*M2), :]
        train_y_add = np.vstack((train_y_RC_add, train_y_NoRC_add))
        train_y = np.vstack((train_y, train_y_add))

        Hx1 = np.hstack([train_x_add, 0.1 * np.ones([train_x_add.shape[0], 1])])
        yx = np.zeros([train_x_add.shape[0], N1 * N2])
        for i in range(N2):
            beta1 = beta11[i]
            Tx1 = Hx1.dot(beta1)
            Tx1 = (ymax - ymin) * (Tx1 - minOfEachWindow[i]) / distMaxAndMin[i] - ymin
            yx[:, N1 * i:N1 * (i + 1)] = Tx1

        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0], 1])])
        wh = Wh[0]
        t2 = sigmoid(Hx2.dot(wh) * parameter)  # tansig  sigmoid
        t3 = np.hstack([yx, t2])
        betat = pinv(t3, c)
        beta = np.hstack([beta, betat])
        beta2 = np.dot(beta, train_y)
        T3 = np.vstack([T3, t3])

        Training_time = time.time() - time_start1
        train_time[0][e + 1] = Training_time

        xx = T3.dot(beta2)
        Y_train = train_y.argmax(axis=1)
        Y_predict = xx.argmax(axis=1)
        train_TP = 0
        train_FP = 0
        train_TN = 0
        train_FN = 0
        for j in range(np.shape(xx)[0]):
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
        Number_train[0][e + 1] = len(train_y)
        print('train_Accuracy     train_Recall      train_Precision')
        print('%f%% %f%% %f%%' % (train_Accuracy[0][e+1] * 100, train_Recall[0][e+1] * 100, train_Precision[0][e+1] * 100))
        '''
        incremental testing steps
        '''
        time_start2 = time.time()
        x = TT3.dot(beta2)
        Y_test = Label_test.argmax(axis=1)
        Y_test_predict = x.argmax(axis=1)
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
        print('%f%% %f%% %f%%' % (test_Accuracy[0][e+1] * 100, test_Recall[0][e+1] * 100, test_Precision[0][e+1] * 100))
        Testing_time = time.time() - time_start2
        test_time[0][e + 1] = Testing_time

    return train_Accuracy, train_Recall, train_Precision, test_Accuracy, test_Recall, test_Precision, test_time, train_time, Number_train

from BayesianOptimization import *
def Add_InputData(s, c, l):
    print('------------------------------ Incremental learning ---------------------------------')
    train_RC_sample = np.load("./ExtractData/train_RC_sample.npy")
    train_RC_label = np.load("./ExtractData/train_RC_label.npy")
    train_NoRC_sample = np.load("./ExtractData/train_NoRC_sample.npy")
    train_NoRC_label = np.load("./ExtractData/train_NoRC_label.npy")
    test_RC_sample = np.load("./ExtractData/test_RC_sample.npy")
    test_RC_label = np.load("./ExtractData/test_RC_label.npy")
    test_NoRC_sample = np.load("./ExtractData/test_NoRC_sample.npy")
    test_NoRC_label = np.load("./ExtractData/test_NoRC_label.npy")

    Input_train = np.load("./ExtractData/Input_train.npy")
    Label_train = np.load("./ExtractData/Label_train.npy")
    Input_test = np.load("./ExtractData/Input_test.npy")
    Label_test = np.load("./ExtractData/Label_test.npy")

    # time_start = time.time()
    # train_RC_path = ('B:/RainCellDetection/DataSet/train/1')
    # train_NoRC_path = ('B:/RainCellDetection/DataSet/train/0')
    # test_RC_path = ('B:/RainCellDetection/DataSet/test/1')
    # test_NoRC_path = ('B:/RainCellDetection/DataSet/test/0')
    #
    # train_RC_sample, train_RC_label, train_NoRC_sample, train_NoRC_label, test_RC_sample, test_RC_label, test_NoRC_sample, test_NoRC_label \
    #     = datasplit(train_RC_path, train_NoRC_path, test_RC_path, test_NoRC_path)
    #
    # Input_train = np.vstack((train_RC_sample, train_NoRC_sample))
    # Label_train = np.vstack((train_RC_label, train_NoRC_label))
    # Input_test = np.vstack((test_RC_sample, test_NoRC_sample))
    # Label_test = np.vstack((test_RC_label, test_NoRC_label))

    # print('--------------------------- BayesianOptimization ------------------------------------')
    # time_start = time.time()
    # len_train_RC = len(train_RC_sample)
    # len_initial_train_RC = int(len_train_RC * 0.7)
    # len_train_NoRC = len(train_NoRC_sample)
    # len_initial_train_NoRC = int(len_train_NoRC * 0.7)
    # #
    # initial_train_x = np.vstack(
    #     (train_RC_sample[0:len_initial_train_RC, :], train_NoRC_sample[0:len_initial_train_NoRC, :]))
    # initial_train_y = np.vstack(
    #     (train_RC_label[0:len_initial_train_RC, :], train_NoRC_label[0:len_initial_train_NoRC, :]))
    # #
    # #
    # # #
    # #
    # best = BayesianOptimization(s, c, initial_train_x, initial_train_y, Input_test, Label_test)
    # N1 = int(best['N1']) # 87
    # N2 = int(best['N2']) # 84
    # N3 = int(best['N3']) # 2580
    # print(N1, N2, N3)
    # BroadLearningSystem(initial_train_x, initial_train_y, Input_test, Label_test, s, c, N1, N2, N3)
    # #
    # time_end = time.time()
    # timecost = time_end - time_start
    # print(timecost)

    # N1 = 12
    # N2 = 10
    # N3 = 938
    # BroadLearningSystem(initial_train_x, initial_train_y, Input_test, Label_test, s, c, N1, N2, N3)
    # time_end = time.time()
    # timecost = time_end - time_start
    # print(timecost)

    N1 = 12
    N2 = 10
    N3 = 938
    l = 3
    train_Accuracy, train_Recall, train_Precision, test_Accuracy, test_Recall, test_Precision, test_time, train_time, Number_train \
        = bls_train_input(train_RC_sample,train_RC_label,train_NoRC_sample,train_NoRC_label,Input_test,Label_test, s, c, N1, N2, N3, l)

    print(test_Accuracy)
    print(test_Recall)
    print(test_Precision)
    print(Number_train)
    print(train_time)

