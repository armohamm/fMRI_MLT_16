import scipy.io as lod
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 100, 50
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


def corr_feature_sel(dict_mat):
    # dict_mat = lod.loadmat("/home/prince/Documents/acads_7thsem/MLT/fMRI/fmri_words.mat")
    # dict_mat = lod.loadmat("fmri_words.mat")
    X_train = dict_mat['X_train']
    Y_train = dict_mat['Y_train']
    X_test = dict_mat['X_test']
    Y_test = dict_mat['Y_test']
    word_features_std = dict_mat['word_features_std']
    # print type(X_train), type(Y_train), X_train.shape, Y_train.shape

    # count = np.zeros(shape=(21764, 1))
    # z = 10000
    # z = feature_num
    word_feature = a = np.empty((300, 218))
    # print "b;"
    for p in range(300):
        # print p
        # mapping y value with their feature
        word_feature[p, :] = word_features_std[Y_train[p] - 1, :]  # 0th row has feature vector for y_train =1

    c_mat = np.zeros(shape=(21764, 218))
    for i in range(218):
        for j in range(21764):
            b_mat = np.corrcoef(X_train[:, j], word_feature[:, i])
            c_mat[j, i] = b_mat[0, 1]
            # print i,j
    # print "before sorting ", c_mat[1:10,1]
    # # c_mat = -np.sort(-c_mat, axis=0)  # sorting in descending order in column
    I = np.ndarray.argsort(-c_mat, axis=0)  # gives the actual index of the value after sorting
    # print "After sorting index",I[1:10, 1]

    # #     [~,I] = sort(C,'descend')
    # #
    return I


def get_feature(X_train, X_test, feature_num, index_mat):
    z = feature_num
    I = index_mat
    count = np.zeros(shape=(21764, 1))
    for r in range(218):
        for q in range(z):
            # print I[q, r] , count[I[q, r], 0]
            # count[I[q, r], 0] = count[I[q, r], 0] + 1
            count[I[q, r], 0] += 1
            # print count[I[q, r], 1]
    # print "count = ", count
    #
    train_new = np.zeros(shape=(300, z))
    test_new = np.zeros(shape=(60, z))
    I_new = np.ndarray.argsort(-count, axis=0)
    for k in range(300):
        for l in range(z):
            train_new[k, l] = X_train[k, I_new[l]]

    for k in range(60):
        for l in range(z):
            test_new[k, l] = X_test[k, I_new[l]]

    # print train_new[1:10,1:10]

    return train_new, test_new
