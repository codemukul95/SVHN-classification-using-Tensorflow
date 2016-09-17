import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io as sio
import cv2

train=sio.loadmat('train_32x32.mat')

X=np.asarray(train['X'])
numimages=X.shape[3]

Xtrain=[]
for i in xrange(numimages):
    Xtrain.append(X[:,:,:,i])
Xtrain=np.asarray(Xtrain)
#print(Xtrain.shape)
#print(train['y'][1000])

Ytrain=np.asarray(train['y'])
#print(Ytrain.shape)

#Converting to one hot representation
Ytr=[]
for el in Ytrain:
    temp=np.zeros(10)
    if el==10:
        temp[0]=1
    elif el==1:
        temp[1]=1
    elif el==2:
        temp[2]=1
    elif el==3:
        temp[3]=1
    elif el==4:
        temp[4]=1
    elif el==5:
        temp[5]=1
    elif el==6:
        temp[6]=1
    elif el==7:
        temp[7]=1
    elif el==8:
        temp[8]=1
    elif el==9:
        temp[9]=1
    Ytr.append(temp)

Ytrain=np.asarray(Ytr)
print(Ytrain.shape)

np.save("/home/eeg/Documents/mukul/svhn/Xtrain",Xtrain)
np.save("/home/eeg/Documents/mukul/svhn/Ytrain",Ytrain)

test=sio.loadmat('test_32x32.mat')

X=np.asarray(test['X'])
numimages=X.shape[3]

Xtest=[]
for i in xrange(numimages):
    Xtest.append(X[:,:,:,i])
Xtest=np.asarray(Xtest)
#print(Xtrain.shape)
#print(train['y'][1000])

Ytest=np.asarray(test['y'])
#print(Ytrain.shape)

#Converting to one hot representation
Yts=[]
for el in Ytest:
    temp=np.zeros(10)
    if el==10:
        temp[0]=1
    elif el==1:
        temp[1]=1
    elif el==2:
        temp[2]=1
    elif el==3:
        temp[3]=1
    elif el==4:
        temp[4]=1
    elif el==5:
        temp[5]=1
    elif el==6:
        temp[6]=1
    elif el==7:
        temp[7]=1
    elif el==8:
        temp[8]=1
    elif el==9:
        temp[9]=1
    Yts.append(temp)

Ytest=np.asarray(Yts)
print(Ytest.shape)
np.save("/home/eeg/Documents/mukul/svhn/Xtest",Xtest)
np.save("/home/eeg/Documents/mukul/svhn/Ytest",Ytest)
