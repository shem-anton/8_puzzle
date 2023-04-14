import numpy as np
import tensorflow as tf


class MNISTModel(tf.Module):
    """A simple tensorflow model to classify MNIST digits.

    Consists of two convolutional blocks with 3x3 convolution kernels with
    stride 1 and valid padding combined with RELU activation and 2x2 max
    pooling. The classifier is a single Dense layer with softmax activation.
    Input shape is 28x28, output shape is 10, data format is NHWC.

    Args:
      first_filters: Number of filters in the first convolutional layer.
      second_filters: Number of filters in the second convolutional layer.
    """

    def __init__(self, first_filters: int, second_filters: int):
        self.layer1_kernels = tf.Variable(
            self.he_uniform(shape=[3, 3, 1, first_filters],
                            n_in=28 * 28))
        self.layer2_kernels = tf.Variable(
            self.he_uniform(shape=[3, 3, first_filters, second_filters],
                            n_in=13 * 13 * first_filters))
        
        self.dense_weights = tf.Variable(
            self.he_uniform(shape=[5 * 5 * second_filters, 10],
                            n_in=5 * 5 * second_filters))
        self.dense_biases = tf.Variable(tf.zeros(10))

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Calculate the network on input tensor."""
        # First convolution, RELU and max pooling
        x = tf.nn.conv2d(x,
                         filters=self.layer1_kernels,
                         strides=[1,1,1,1],
                         padding="VALID")
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2,2), strides=(2,2), padding="VALID")

        # Second convolution, RELU and max pooling
        x = tf.nn.conv2d(x,
                         filters=self.layer2_kernels,
                         strides=[1,1,1,1],
                         padding="VALID")
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2,2), strides=(2,2), padding="VALID")

        # Flatten, apply dense layer and biases
        x = tf.reshape(x, (x.shape[0], tf.size(x) / x.shape[0]))
        x = tf.matmul(x, self.dense_weights)
        x = x + self.dense_biases

        # Softmax activation function
        x = tf.nn.softmax(x)
        return x

    def he_uniform(self, shape: list, n_in: int) -> tf.Tensor:
        """Randomly generate tensor with unifor He initializaion.
        
        Args:
          shape: shape of the generated tensor.
          n_in: the number of input units in the weight tensor.
        """
        limit = np.sqrt(6. / n_in)
        return tf.random.uniform(shape=shape, minval=-limit, maxval=limit)
