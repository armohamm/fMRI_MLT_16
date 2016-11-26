import scipy.io as lod
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt

import featr_sel_rank_spedup as fea_sel


dict_mat = lod.loadmat("/home/prince/Documents/acads_7thsem/MLT/fMRI/fmri_words.mat")
# dict_mat = lod.loadmat("fmri_words.mat")
X_train = dict_mat['X_train']
Y_train = dict_mat['Y_train']
X_test = dict_mat['X_test']
Y_test = dict_mat['Y_test']
word_features_std = dict_mat['word_features_std']
print type(X_train), type(Y_train), X_train.shape, Y_train.shape



def ridge_regression(data, y_val, alpha ):
    # Fit the model
    ridgereg = linear_model.Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data, y_val)
    W = ridgereg.coef_

    return W

def accuracy( X_test , Y_test , W):
    correct_count = 60
    y_feat_test_pred = np.mat(X_test) * np.mat(W.transpose())
    for i in range(60):
        dist_ = np.zeros(2)
        dist_[0] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[Y_test[i][0] - 1, :])
        dist_[1] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[Y_test[i][1] - 1, :])
        min_ind = np.argmin(dist_)
        # y_test_pred[i] = Y_test[i, min_ind]
        correct_count -= min_ind
        # print(y_test_pred[i])
    print("correct count: " + str(correct_count))
    print("accuracy: " + str(correct_count * 100 / 60.0) + "%")
    return (correct_count * 100 / 60.0)

def plot_graph(x , y , xlabel , ylabel ,title ,alpha):

    # 3rd param is size
    plt.scatter(x, y, 100, 'crimson', '*')
    # labeling coordinates for the point
    # for xy in zip(x, y):  # <--
    #     plt.annotate('(%d, %.2f)' % xy, xy=xy, textcoords='data', fontsize=8)  #
    # Show grid in the plot
    # plt.grid()
    # # Sets x-axis
    # plt.xlabel(xlabel)
    # # Sets y-axis
    # plt.ylabel(ylabel)
    # # Sets title
    # plt.title(title)
    # # Finally, display the plot
    # plt.show()
    #saving
    plt.savefig('corr_alpha_%g'%alpha)
    return 0

# mapping the y_value to the 218 feature
ans = word_features_std[np.ix_(((Y_train.transpose())[0]) - 1, np.arange(218))]

# clf = linear_model.Ridge(alpha=0.1)
# clf.fit(X_train, ans)
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20 , 50 , 100 , 102,104,109,115,124,1000]
# alpha_ridge = [20]
feature_num = [7000 ,8000, 9000 , 10000,12000, 13000 , 14000,15000,17000 , 19000]
# acc = []
print "cal correlation..."
index_mat = fea_sel.corr_feature_sel(dict_mat)

X_train_actual = X_train
X_test_actual = X_test
# file_1 = open("file.txt", 'w')
for alpha in alpha_ridge:
    acc = []
    # graph_plot_dict = {}
    for fea_num in feature_num:
        t = time.time()
        # calling feature selection function from feature_sel_ranking.py file
        X_train , X_test = fea_sel.get_feature(X_train_actual, X_test_actual,fea_num, index_mat)
        print X_test.shape , X_train.shape
        W = ridge_regression(X_train, ans,alpha)
        # err = cross_validation(X_train, ans)
        # print " (clf.coef_.shape) = ", (clf.coef_.shape)
        # W = clf.coef_

        y_feat_test_pred = np.mat(X_test) * np.mat(W.transpose())
        acc = acc + [accuracy(X_test,Y_test,W)]

        # calculating time taken
        t_diff = time.time() - t
        print " t_diff = ", (t_diff)
        print "alpha = ",alpha,"\n","acc = ",acc  , "\n" , "no. of feature = " , fea_num

    # print>>file_1, acc
# plt.plot(alpha_ridge,acc,'*')
#     plot_graph(feature_num, acc, "FEATURES (SELECTED USING CORRELATION RANKING)", "ACCURACY",
#                    "ACCURACY VS FEATURES FOR ALPHA = %g" % alpha ,alpha)
    # here %g is used so that 1e-15 is printed instead of 0.0000 in float

    print "\n#########################\n"
# plt.plot(feature_num,acc,'*')
# plt.show()
# import test
# acc =  [85.0, 86.666666666666671, 85.0, 80.0, 76.666666666666671, 75.0, 71.666666666666671, 70.0, 71.666666666666671, 71.666666666666671]
# feature_num = [7000 ,8000, 9000 , 10000,12000, 13000 , 14000,15000,17000 , 19000]
# test.plot_graph(feature_num, acc, "FEATURES (SELECTED USING CORRELATION RANKING)", "ACCURACY",
#                    "ACCURACY VS FEATURES FOR ALPHA = " )
# plt.scatter(feature_num, acc, 100, 'crimson', '*')
# plt.show()