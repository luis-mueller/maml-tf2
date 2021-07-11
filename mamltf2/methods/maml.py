import tensorflow as tf
from mamltf2.model import Model
from mamltf2.optimizer import FastWeights

class MAML(Model):
    def __init__(self, *args, **kwargs):
        """ Implementation of MAML (Finn et al. 2017).
        """
        super().__init__(*args, **kwargs)

        self.optimizer = tf.keras.optimizers.Adam(self.outerLearningRate)
        self.fastWeights = FastWeights(self.model, self.innerLearningRate)

    @tf.function
    def taskLoss(self, batch):
        """Computes the loss for one task given one batch of inputs and correspondings labels
        """
        y_train, x_train, y_test, x_test = batch

        with tf.GradientTape() as taskTape:
            loss = self.lossfn(y_train, self.model(
                tf.reshape(x_train, (-1, 1))))

        grads = taskTape.gradient(loss, self.model.trainable_weights)
        weights = self.fastWeights.computeUpdate(zip(grads, self.model.trainable_weights))

        return self.lossfn(y_test, self.fastWeights(weights, x_test))

    @tf.function
    def update(self, batch):
        """Implements the meta-update step for a bunch of tasks.

        @batch: Tuple of training and test data for the update step. Can be directly passed through to 
        the task updates.
        """
        with tf.GradientTape() as metaTape:
            loss = tf.reduce_sum(
                tf.map_fn(self.taskLoss, elems=batch, fn_output_signature=tf.float32))

        self.optimizer.minimize(loss, self.model.trainable_variables, tape=metaTape)
        return loss
