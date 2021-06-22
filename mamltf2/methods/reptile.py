import tensorflow as tf
from mamltf2.model import Model
from mamltf2.tftools import TensorflowTools


class RegressionReptile(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = self.model.get_weights()

    @tf.function
    def taskLoss(self, batch):
        """Computes the loss for one task given one batch of inputs and correspondings labels
        """
        y_train, x_train = batch

        with tf.GradientTape() as taskTape:
            loss = self.mse(y_train, self.model(
                tf.reshape(x_train, (-1, 1))))

        grads = taskTape.gradient(loss, self.model.trainable_weights)
        fastWeights = self.fastWeights.compute(grads)

        self.fastWeights.apply([weights + 0.1 * (fastWeights[i] - weights)
                                for (i, weights) in enumerate(self.model.trainable_weights)])
        
        return loss

    @tf.function
    def update(self, batch):
        """Implements the meta-update step for a bunch of tasks.

        @batch: Tuple of training and test data for the update step. Can be directly passed through to 
        the task updates.
        """
        return tf.reduce_sum(
            tf.map_fn(self.taskLoss, elems=batch, fn_output_signature=tf.float32))

    def trainBatch(self, nSamples, nTasks, nBatch):
        return super().trainBatch(nSamples, nTasks, nBatch, alsoSampleTest=False)
