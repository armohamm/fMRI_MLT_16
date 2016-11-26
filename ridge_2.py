import scipy.io as lod
# this uses ridge model from scikit learn to learn from the data
import scipy.io as lod
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 100, 50

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
    return str(correct_count * 100 / 60.0)

from sklearn.model_selection import cross_val_score

# mapping the y_value to the 218 feature
ans = word_features_std[np.ix_(((Y_train.transpose())[0]) - 1, np.arange(218))]

# clf = linear_model.Ridge(alpha=0.1)
# clf.fit(X_train, ans)
# alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20 , 50 , 100 , 102,104,109,115,124,1000]
alpha_ridge = [20]
acc = []
for alpha in alpha_ridge:
    t = time.time()
    W = ridge_regression(X_train, ans,alpha)
    # err = cross_validation(X_train, ans)
    # print " (clf.coef_.shape) = ", (clf.coef_.shape)
    # W = clf.coef_

    # y_feat_test_pred = np.mat(X_test) * np.mat(W.transpose())
    acc = acc + [accuracy(X_test,Y_test,W)]

    # calculating time taken
    t_diff = time.time() - t
    print " t_diff = ", (t_diff)
print alpha_ridge,"\n",acc
plt.plot(alpha_ridge,acc,'*')
plt.show()