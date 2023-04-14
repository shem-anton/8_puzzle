import numpy as np
import tensorflow as tf

from dataset import MNISTDataset

dataset = MNISTDataset(data_directory="data/", filename="train.csv", batch_size=32)

image = np.ones((150, 150)) * 255

tensor = dataset.format_image(image)
print(tensor)
print(tensor.shape)