import tensorflow as tf
from mamltf2.model import Model


class RegressionReptile(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nInnerSteps = 5
        self.interpolationRate = 0.1
        self.mysgd = tf.keras.optimizers.SGD(0.02)

    @tf.function
    def interpolate(self, source, target):
        """Linearly interpolate between source and target.
        """
        return target + (source - target) * self.interpolationRate

    @tf.function
    def copyWeightsApply(self, source, target, fn=lambda s, t: s):
        """Assign weights from source to target. Apply a transform fn to source and target weights first.
        """
        for j in range(len(source.trainable_weights)):
            target.trainable_weights[j].assign(
                fn(source.trainable_weights[j], target.trainable_weights[j]))

    @tf.function
    def taskLoss(self, batch):
        """Computes the loss for one task given one batch of inputs and correspondings labels
        """
        y, x = batch
        self.copyWeightsApply(self.model, self.modelCopy)

        self.interpolationRate *= 0.9999
        #adam = tf.keras.optimizers.Adam(0.001)

        for _ in range(self.nInnerSteps):
            with tf.GradientTape() as taskTape:
                loss = self.mse(y, self.modelCopy(tf.reshape(x, (-1, 1))))

            self.mysgd.minimize(
                loss, self.modelCopy.trainable_variables, tape=taskTape)

        self.copyWeightsApply(self.modelCopy, self.model, self.interpolate)
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
