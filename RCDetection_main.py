# -*- coding: UTF-8 -*-

# Main program
from BroadLearningSystem import *
from Functions import *
from BayesianOptimization import *
from OriginalBLS import *
from Add_InputData import *
from Add_Nodes import *
from Add_Data_Nodes import *

if __name__ == '__main__':
    # 设置参数
    N1 = 12  # 16
    N2 = 10  # # of windows -------Feature mapping layer 82
    N3 = 938  # # of enhancement nodes -----Enhance layer 2328
    l = 7  # # of incremental steps
    M = 200
    L = 1
    s = 0.8  # shrink coefficient
    c = 2 ** -30  # Regularization coefficient
    #
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
    #
    # # 保存提取的数据
    # np.save("./ExtractData/Input_train.npy", Input_train)
    # np.save("./ExtractData/Label_train.npy", Label_train)
    # np.save("./ExtractData/Input_test.npy", Input_test)
    # np.save("./ExtractData/Label_test.npy", Label_test)
    # #
    # np.save("./ExtractData/train_RC_sample.npy", train_RC_sample)
    # np.save("./ExtractData/train_RC_label.npy", train_RC_label)
    # np.save("./ExtractData/train_NoRC_sample.npy", train_NoRC_sample)
    # np.save("./ExtractData/train_NoRC_label.npy", train_NoRC_label)
    # np.save("./ExtractData/test_RC_sample.npy", test_RC_sample)
    # np.save("./ExtractData/test_RC_label.npy", test_RC_label)
    # np.save("./ExtractData/test_NoRC_sample.npy", test_NoRC_sample)
    # np.save("./ExtractData/test_NoRC_label.npy", test_NoRC_label)

    # start = time.time()
    # 确定模型参数 并保存该次训练/测试数据集划分结果
    # Input_train = np.load("./ExtractData/Input_train.npy")
    # Label_train = np.load("./ExtractData/Label_train.npy")
    # Input_test = np.load("./ExtractData/Input_test.npy")
    # Label_test = np.load("./ExtractData/Label_test.npy")
    #
    # BayesianOptimization(s, c, Input_train, Label_train, Input_test, Label_test)

    # 简单BLS训练与测试
    Original_BLS(s, c, N1, N2, N3)

    # 增加数据进行增量学习
    # Add_InputData(s, c, l)

    # 增加增强节点数目
    # Add_EnhanceNodes(s, c, N1, N2, N3, L, M)

    # 增加mapping feature和相应的增强节点
    # Add_FeatureEnhanceNodes(s, c, N1, N2, N3, L, M)

    # 增加数据并增加节点
    # Incremental_Learning(s, c, N1, N2, N3, l, M, L)
