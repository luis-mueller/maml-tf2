import tensorflow as tf
from mamltf2.model import Model


class RegressionMAML(Model):
    @tf.function
    def step(self, grads, x):
        """'Fast weights': Implements a single SGD step on the current meta model with a fixed step-size. It seems that tensorflow 
        is not able to differentiate through an optimizer's steps, hence this implementation. In the meta-validation step this can 
        be replaced by a proper tf.keras.optimizers.SGD instance.
        """
        k = 0
        y = tf.reshape(x, (-1, 1))
        for j in range(len(self.model.layers)):
            kernel = self.model.layers[j].kernel - 0.01 * grads[k]
            bias = self.model.layers[j].bias - 0.01 * grads[k+1]
            y = self.model.layers[j].activation(y @ kernel + bias)
            k += 2
        return y

    @tf.function
    def taskLoss(self, batch):
        """Computes the loss for one task given one batch of inputs and correspondings labels
        """
        y_train, x_train, y_test, x_test = batch

        with tf.GradientTape() as taskTape:
            loss = self.mse(y_train, self.model(
                tf.reshape(x_train, (-1, 1))))

        grads = taskTape.gradient(loss, self.weights)
        return self.mse(y_test, self.step(grads, x_test))

    @tf.function
    def update(self, batch):
        """Implements the meta-update step for a bunch of tasks.

        @batch: Tuple of training and test data for the update step. Can be directly passed through to 
        the task updates.
        """
        with tf.GradientTape() as metaTape:
            loss = tf.reduce_sum(
                tf.map_fn(self.taskLoss, elems=batch, fn_output_signature=tf.float32))

        self.optimizer.minimize(loss, self.weights, tape=metaTape)
        return loss
