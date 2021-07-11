import tensorflow as tf
from mamltf2.model import Model

class Reptile(Model):
    def __init__(self, *args, nInnerSteps = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.nInnerSteps = nInnerSteps
        self.interpolationRateInitial = self.outerLearningRate
        self.sgd = tf.keras.optimizers.SGD(self.innerLearningRate)
        self.modelCopy = tf.keras.models.clone_model(self.model)

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
    def updateInterpolationRate(self):
        self.interpolationRate = self.interpolationRateInitial * (1 - self.nIteration / self.nIterations)
        self.nIteration += 1

    @tf.function
    def taskLoss(self, batch):
        """Computes the loss for one task given one batch of inputs and correspondings labels
        """
        y, x = batch
        self.copyWeightsApply(self.model, self.modelCopy)

        self.updateInterpolationRate()

        for _ in range(self.nInnerSteps):
            with tf.GradientTape() as taskTape:
                loss = self.lossfn(y, self.modelCopy(tf.reshape(x, (-1, 1))))

            self.sgd.minimize(
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
        self.nIterations = nBatch 
        self.nIteration = 0
        self.innerLearningRate = self.interpolationRateInitial

        return super().trainBatch(nSamples, nTasks, nBatch, alsoSampleTest=False)
