import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io as sio
import cv2



class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 3  #rgb image
      images =images.reshape(images.shape[0],
                              images.shape[1] *images.shape[2]*images.shape[3])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        #images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)

      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]



def read_data_sets(fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  datasets = DataSets()
  X_train=np.load('Xtrain.npy')
  Y_train=np.load('Ytrain.npy')
  X_test=np.load('Xtest.npy')
  Y_test=np.load('Ytest.npy')

  print(X_train.shape)
  print(Y_test.shape)
  X_val=X_test[2000:3000]
  Y_val=Y_test[2000:3000]



  datasets.train=DataSet(X_train,Y_train,dtype=dtype)
  datasets.validation=DataSet(X_val,Y_val,dtype=dtype)
  datasets.test=DataSet(X_test,Y_test,dtype=dtype)


  return datasets
