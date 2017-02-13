# SVHN-classification-using-Tensorflow
Attempt to implement classification of SVHN Dataset using Tensorflow

Dataset URL: http://ufldl.stanford.edu/housenumbers/

Libraries Used: NumPy, Scipy, Tensorflow and TfLearn

MNIST example tensorflow: https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#deep-mnist-for-experts

CIFAR-10 example tensorflow: https://www.tensorflow.org/versions/0.6.0/tutorials/deep_cnn/index.html

Description of the files: <br>
1. load_data.py: Reads the dataset '.mat' files from the directory and returns them in the form of Numpy arrays. The labels are converted in one_hot_encoded format to ease the process of training
<br>
2. network.py: The CNN architecture is built to train the model.
<br>
3. test_network.py: The training set is tested against the trained weights, to check the testing accuracy value.
