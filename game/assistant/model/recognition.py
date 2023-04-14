import numpy as np
import tensorflow as tf

from assistant.model.tf_model import MNISTModel
from assistant.model.dataset import format_image
from assistant.model.dataset import MNISTDataset
 

class DigitRecognitionModel:
    """MNIST digit recognition model."""

    def __init__(self):
        self.model = MNISTModel(first_filters=8, second_filters=8)
        self.dataset = MNISTDataset(data_directory="data/",
                                    filename="train.csv",
                                    batch_size=32)
        self.learning_rate = 0.001
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def train_epoch(self) -> float:
        """Train the model once on a full training dataset.
        
        Returns:
          Total training loss.
        """
        epoch_loss = 0
        for x_batch, y_batch in self.dataset.get_train_data():
            # Record the model operations on gradient tape
            with tf.GradientTape() as tape:
                batch_loss = self.loss(self.model(x_batch), y_batch)

            # Compute the gradients of categorical cross entropy with respect
            # to model weights
            grads = tape.gradient(batch_loss, self.model.variables)
            # Make a gradient descent step
            for g,v in zip(grads, self.model.variables):
                v.assign_sub(self.learning_rate * g)
            
            epoch_loss += batch_loss.numpy()
        return epoch_loss

    def recognize(self, image: np.ndarray) -> int:
        """Run model inference on an input image to recognize the digit.

        Args:
          image: input image in a human friendly format.

        Returns:
          The predicted digit, recognized by the model.
        """
        # There is no need to recognize empty image
        if image.min() == 255:
            return 0

        model_input = format_image(image)
        prediction = self.model(model_input)
        argmax = tf.math.argmax(prediction, axis=-1)
        return int(argmax.numpy())
