from BroadLearningSystem import *

def BayesianOptimization(s, c, Input_train, Label_train, Input_test, Label_test):

    # 第一次读取数据 并将路径与label存入csv文件，后续使用直接读取存储的文件
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

    # Input_train = np.load("./ExtractData/Input_train.npy")
    # Label_train = np.load("./ExtractData/Label_train.npy")
    # Input_test = np.load("./ExtractData/Input_test.npy")
    # Label_test = np.load("./ExtractData/Label_test.npy")

    from bayes_opt import BayesianOptimization
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    import hyperopt.pyll.stochastic
    print('--------------------BayesianOptimization------------------------')
    def BLSopt(argsDict):    #
        N1 = int(argsDict['N1'])
        N2 = int(argsDict['N2'])
        N3 = int(argsDict['N3'])
        train_Accuracy, train_Recall, train_Precision, test_Accuracy, test_Recall, test_Precision, test_time, train_time\
            = BroadLearningSystem(Input_train, Label_train, Input_test, Label_test, s, c, N1, N2, N3)
        return -test_Accuracy
    spaceBL = {
        'N1': hp.quniform('N1', 1,20,1),
        'N2': hp.quniform('N2', 1,50,2),
        'N3': hp.quniform('N3', 1,2000,2)
    }
    trials = Trials()
    best = fmin(BLSopt, space=spaceBL, algo=tpe.suggest, max_evals=10, trials=trials)
    print('best:', best)

    return best