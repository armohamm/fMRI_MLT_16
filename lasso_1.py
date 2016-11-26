from sklearn import linear_model
import time
import numpy as np
import scipy.io

mat = scipy.io.loadmat("/home/prince/Documents/acads_7thsem/MLT/fMRI/fmri_words.mat")

x_train = mat['X_train']
y_train = mat['Y_train']
x_test = mat['X_test']
y_test = mat['Y_test']
word_features_std=mat['word_features_std']

t=time.time()
ans=word_features_std[np.ix_(((y_train.transpose())[0])-1, np.arange(218))]
t_diff = time.time()-t
print " t_diff) = ",(t_diff)

clf = linear_model.Lasso(alpha=0.1)
# y=ans[:, 0]
# clf.fit(x_train,y)
clf.fit(x_train,ans)
print " (clf.coef_.shape) = ",(clf.coef_.shape)
print "(np.count_nonzero(clf.coef_)) = " , (np.count_nonzero(clf.coef_))
W=clf.coef_
y_feat_test_pred = np.mat(x_test)*np.mat(W.transpose())



correct_count=60
for i in range(60):
    dist_ = np.zeros(2)
    dist_[0] = np.linalg.norm(y_feat_test_pred[i,:] - word_features_std[y_test[i][0]-1,:] )
    dist_[1] = np.linalg.norm(y_feat_test_pred[i,:] - word_features_std[y_test[i][1]-1,:] )
    min_ind = np.argmin(dist_)
    y_test_pred[i] = y_test[i, min_ind]
    correct_count -= min_ind
    print(y_test_pred[i])
print("correct count: " + str(correct_count))
print("accuracy: "+ str(correct_count*100/60.0) + "%")
