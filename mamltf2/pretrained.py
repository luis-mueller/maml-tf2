import tensorflow as tf
from mamltf2.model import Model


class PretrainedModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = tf.keras.optimizers.SGD(self.innerLearningRate)


    @tf.function
    def computeTaskLosses(self, batch):
        """Computes a tensor of losses across a collection of tasks.
        """
        def taskLoss(batch):
            y_train, x_train = batch
            return self.lossfn(y_train, self.model(tf.reshape(x_train, (-1, 1))))

        return tf.map_fn(taskLoss, elems=batch, fn_output_signature=tf.float32)

    @tf.function
    def update(self, batch):
        """Compute the loss over a training batch and take an optimizer step.
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(self.computeTaskLosses(batch))

        self.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss

    def trainBatch(self, nSamples, nTasks, nBatch):
        return super().trainBatch(nSamples, nTasks, nBatch, alsoSampleTest=False)
