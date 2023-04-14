import os

import numpy as np
import pandas as pd
import tensorflow as tf


def format_image(image: np.ndarray) -> tf.Tensor:
    """Reformat input image to model compatible format.

    Normalize the image, invert white and black and reshape to 28x28 tensor.

    Args:
        image: human friendly image (large, black on white) represented as
        numpy array.

    Returns:
        The same image reformatted to be compatible with the MNIST model.
    """
    image = image / image.max()
    image = 1 - image
    image_tensor = tf.constant(image)
    image_tensor = tf.expand_dims(image_tensor, axis=2)
    image_tensor = tf.image.resize(images=image_tensor, size=[28, 28])
    return tf.expand_dims(image_tensor, axis=0)


class MNISTDataset:
    """Class to handle the training dataset.

    Loads the images and labels from a csv source, serves the training dataset
    to the model.

    Args:
      data_directory: directory with the source csv file and training images.
      filename: csv source name relative to the data directory.
      batch_size: batch size of the dataset.
    """

    def __init__(self, data_directory: str, filename: str, batch_size: int):
        train_path = os.path.join(data_directory, filename)
        train_df = pd.read_csv(train_path)

        images = train_df["image_paths"].to_numpy()
        images = data_directory + images
        labels = train_df["labels"].to_numpy()
        self.batch_size = batch_size
        self.size = labels.shape[0]

        self.dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        self.dataset = self.dataset.map(self.load_image)

    def get_train_data(self) -> tf.data.Dataset:
        """Shuffle and serve the dataset in batches.

        Returns:
          The shuffled and batched training dataset.
        """
        dataset = self.dataset.shuffle(buffer_size=self.size)
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def load_image(self,
                   file_path: tf.Tensor,
                   label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Map data from (filename, integer label) to model compatible data.

        Load png image from the filename and one-hot encode the label.

        Args:
          file_path: string tensor with image filename.
          label: numerical tensor with image label.

        Returns:
          3D Tensor with loaded image in HWC format and 10-dimensional one-hot
          encoded label.
        """
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.one_hot(label, depth=10)
        return image, label
