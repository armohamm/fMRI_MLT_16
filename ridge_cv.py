import scipy.io as lod
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
import scipy.io as lod
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 100, 50
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

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
    # ridgereg = linear_model.Lasso(alpha=alpha) # for lasso
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
    return str(correct_count * 100 / 60.0)


def err_predict(X_test, Y_test, W):
    y_feat_test_pred = np.mat(X_test) * np.mat(W.transpose())
    (test_count, dimen) = X_test.shape  # getting no. of test data
    correct_count = 0
    for i in range(test_count):
        err = np.linalg.norm(y_feat_test_pred[i, :] - Y_test[i, :])
        print("err " + str(err))
    return err

def plot_graph(alpha_ridge_dict):
    x = alpha_err_dict.keys()
    y =  alpha_err_dict.values()
    # 3rd param is size
    plt.scatter(x, y, 100, 'crimson', '*')
    # labeling coordinates for the point
    for xy in zip(x, y):  # <--
        plt.annotate('(%d, %.2f)' % xy, xy=xy, textcoords='data', fontsize=8)  #

    # Show grid in the plot
    plt.grid()
    # Sets x-axis
    plt.xlabel('VALUE OF ALPHA')
    # Sets y-axis
    plt.ylabel('AVERAGE ERROR ')
    # Sets title
    plt.title("Cross_validation_for RIDGE REGRESSION")
    # Finally, display the plot
    plt.show()
    return
# mapping the y_value to the 218 feature
ans = word_features_std[np.ix_(((Y_train.transpose())[0]) - 1, np.arange(218))]

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20 , 50 , 100 , 102,104,109,115,124,1000]

# alpha_ridge = [20]
acc = []
alpha_err_dict = {}
for alpha in alpha_ridge:
    t = time.time()
    # #####################################################################################################
    # making loo model
    loo = LeaveOneOut()
    # print loo.get_n_splits(X_train)
    err_alpha = 0 # to calculate total eerr and sum it over all split
    for train_index, test_index in loo.split(X_train):
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = ans[train_index], ans[test_index]
        #  training the model with splitted data
        W = ridge_regression(X_train1, y_train1, alpha)
        #  # find the err for the validating data (this cas has no of validating data = 1 in each loop)
        err_alpha += err_predict(X_test1, y_test1, W)
    print "err_ALPHA  = ", err_alpha / 300
    alpha_err_dict[alpha] = err_alpha / 300
    # ####################################################################################################
    # Kfold model for cross validation
    # n_splits = 200
    # kf = KFold(n_splits=n_splits)
    # KFold(n_splits=2, random_state=None, shuffle=False)
    # err_alpha_K = 0  # to calculate total eerr and sum it over all split
    # for train_index, test_index in kf.split(X_train):
    #     X_train1, X_test1 = X_train[train_index], X_train[test_index]
    #     y_train1, y_test1 = ans[train_index], ans[test_index]
    #     #  training the model with splitted data
    #     W = ridge_regression(X_train1, y_train1, alpha)
    #     #  # find the err for the validating data (this cas has no of validating data = 1 in each loop)
    #     err_alpha_K += err_predict(X_test1, y_test1, W)
    # print "err_ALPHA  = ", err_alpha_K / n_splits , "alpha = " , alpha
    # alpha_err_dict[alpha] = err_alpha_K / n_splits
    # calculating time taken
    t_diff = time.time() - t
    print " t_diff = ", (t_diff)
print alpha_ridge,"\n",alpha_err_dict

plot_graph(alpha_ridge)
