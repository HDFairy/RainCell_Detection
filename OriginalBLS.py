from BroadLearningSystem import *
from Functions import *

def Original_BLS(s, C, N1, N2, N3):

    time_start = time.time()
    train_RC_path = ('B:/RainCellDetection/DataSet/train/1')
    train_NoRC_path = ('B:/RainCellDetection/DataSet/train/0')
    test_RC_path = ('B:/RainCellDetection/DataSet/test/1')
    test_NoRC_path = ('B:/RainCellDetection/DataSet/test/0')

    train_RC_sample, train_RC_label, train_NoRC_sample, train_NoRC_label, test_RC_sample, test_RC_label, test_NoRC_sample, test_NoRC_label \
        = datasplit(train_RC_path, train_NoRC_path, test_RC_path, test_NoRC_path)

    Input_train = np.vstack((train_RC_sample, train_NoRC_sample))
    Label_train = np.vstack((train_RC_label, train_NoRC_label))
    Input_test = np.vstack((test_RC_sample, test_NoRC_sample))
    Label_test = np.vstack((test_RC_label, test_NoRC_label))
    #
    # # 保存提取的数据
    # np.save("./ExtractData/Input_train.npy", Input_train)
    # np.save("./ExtractData/Label_train.npy", Label_train)
    # np.save("./ExtractData/Input_test.npy", Input_test)
    # np.save("./ExtractData/Label_test.npy", Label_test)

    # Input_train = np.load("./ExtractData/Input_train.npy")
    # Label_train = np.load("./ExtractData/Label_train.npy")
    # Input_test = np.load("./ExtractData/Input_test.npy")
    # Label_test = np.load("./ExtractData/Label_test.npy")

    print('------------------- Broad Learning System ---------------------------')
    BroadLearningSystem(Input_train, Label_train, Input_test, Label_test, s, C, N1, N2, N3)
    time_end = time.time()
    timecost = time_end-time_start
    print(timecost)

