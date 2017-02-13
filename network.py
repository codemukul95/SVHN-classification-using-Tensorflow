import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from load_input import load_train_data


X_train, Y_train = load_train_data()
X_train, Y_train = shuffle(X_train, Y_train)
print('shuffle done')

X_val = X_train[2000:4000]
Y_val = Y_train[2000:4000]
network = input_data(shape=[None, 32, 32, 3])

network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier')
network = batch_normalization(network)

network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier')      
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier')    
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier')
network = max_pool_2d(network, 2)
network = batch_normalization(network)


network = conv_2d(network, 64, 3, activation='relu', weights_init='xavier')
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = fully_connected(network, 256, activation='relu', weights_init='xavier')
network = dropout(network, 0.25)


network = fully_connected(network, 10, activation='softmax', weights_init='xavier')


network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X_train, Y_train, n_epoch=10, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=100,
          snapshot_epoch=True,
          run_id='svhn_1')

model.save("svhn_1.tfl")
print("Network trained and saved as svhn_1.tfl!")