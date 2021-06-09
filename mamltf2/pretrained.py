import tensorflow as tf
from mamltf2.model import Model


class PretrainedModel(Model):
    @tf.function
    def computeTaskLosses(self, trainingLabels, trainingInputs):
        """Computes a tensor of losses across a collection of tasks.
        """
        def taskLoss(batch):
            y_train, x_train = batch
            return self.mse(y_train, self.model(tf.reshape(x_train, (-1, 1))))

        batch = (trainingLabels, trainingInputs)
        return tf.map_fn(taskLoss, elems=batch, fn_output_signature=tf.float32)

    @tf.function
    def update(self, trainingLabels, trainingInputs):
        """Compute the loss over a training batch and take an optimizer step.
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(self.computeTaskLosses(
                trainingLabels, trainingInputs))

        self.optimizer.minimize(loss, self.weights, tape=tape)
        return loss

    def trainBatch(self, nSamples, nTasks, nBatch):
        """Utility method to train an entire episode/epoch in one go, sampling the entire data at once.
        This should speed up training significantly, since this way the inner update function can be compiled.
        """
        batch = self.taskDistribution.sampleTaskBatches(
            nSamples, nTasks, nBatch, alsoSampleTest=False)

        return float(tf.reduce_mean(tf.map_fn(lambda batch: self.update(
                *batch), elems=batch, fn_output_signature=tf.float32) / nTasks))
