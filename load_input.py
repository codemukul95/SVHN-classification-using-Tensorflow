import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.misc import *
from tflearn.data_utils import shuffle, to_categorical

#Dataset location
train_location = '/home/robolab/Documents/Mukul Arora/SVHN/dataset/train_32x32.mat'
test_location = '/home/robolab/Documents/Mukul Arora/SVHN/dataset/test_32x32.mat'




def load_train_data():
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in xrange(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in xrange(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = to_categorical(Y_train,10)
    return (X_train,Y_train)

def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in xrange(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in xrange(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = to_categorical(Y_test,10)
    return (X_test,Y_test)