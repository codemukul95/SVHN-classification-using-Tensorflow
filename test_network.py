import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from load_input import load_test_data


X_test, Y_test = load_test_data()
X_test, Y_test = shuffle(X_test, Y_test)

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

model.load("svhn_1.tfl")

total_samples = len(X_test)
correct_predict = 0.
for i in xrange(len(X_test)):
	prediction = model.predict([X_test[i]])
	digit = np.argmax(prediction)
	label = np.argmax(Y_test[i])
	if(digit == label):
		correct_predict += 1

print(correct_predict/total_samples)