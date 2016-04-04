# SVHN-classification-using-Tensorflow
Attempt to implement classification of SVHN Dataset using Tensorflow

Dataset URL: http://ufldl.stanford.edu/housenumbers/

Libraries Used: NumPy, Scipy and Tensorflow

MNIST example tensorflow: https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#deep-mnist-for-experts

CIFAR-10 example tensorflow: https://www.tensorflow.org/versions/0.6.0/tutorials/deep_cnn/index.html

Description of the files:
1. checkSVHN.py: If the '.mat' files are saved in the working directory it will extract them in the prooper format and save as NumPy arrays.
2. svhnInput.py: Loads the numpy arrays from the working directory and models the dataset for the CNN architecture.
   NOTE: Validation dataset has been arbitrarily chosen by me.
3. svhnTest.py: Makes the architecture for training the data and testing the test data to find accuracy.
