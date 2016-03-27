import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io as sio
import cv2

import svhnInput

data=svhnInput.read_data_sets()

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 32*32*3])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-3)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-3)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=False)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,32,32,3])

h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_norm1=tf.nn.local_response_normalization(h_pool1,4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')


W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
h_norm2=tf.nn.local_response_normalization(h_conv2,4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
h_pool2 = max_pool_2x2(h_norm2)


W_fc1 = weight_variable([8 * 8* 128,500])
b_fc1 = bias_variable([500])

W_fc2 = weight_variable([500,200])
b_fc2 = bias_variable([200])


h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*128])

h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

h_fc2=tf.nn.sigmoid(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)

W_fc3 = weight_variable([200, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(5000):
  batch = data.train.next_batch(100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
