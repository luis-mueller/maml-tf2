import tensorflow as tf
from mamltf2.deployable import DeployableModel


class Model(DeployableModel):
    def __init__(self, model, taskDistribution, lossfn='mse', outerLearningRate=0.001, innerLearningRate=0.01):
        """Super class for models in this library: Initializes from disk or from a tensorflow layers model.
        """
        super().__init__(model)

        self.outerLearningRate = outerLearningRate
        self.innerLearningRate = innerLearningRate

        self.taskDistribution = taskDistribution
        self.lossfn = tf.keras.losses.MeanSquaredError() if lossfn == 'mse' else lossfn

    def fit(self, y, x, nSteps=1):
        """For meta-validation: Take a fixed number of gradient steps for the loss given y and x.
        """
        clone = self.__deepCloneModel(self.model)
        optimizer = tf.keras.optimizers.SGD(self.innerLearningRate)
        for _ in range(nSteps):
            with tf.GradientTape() as tape:
                loss = self.lossfn(y, clone(x))

            optimizer.minimize(loss, clone.trainable_weights, tape=tape)
        return clone

    def trainBatch(self, nSamples, nTasks, nBatch, alsoSampleTest=True):
        """Utility method to train an entire episode/epoch in one go, sampling the entire data at once.
        This should speed up training significantly, since this way the inner update function can be compiled.
        """
        batch = self.taskDistribution.sampleTaskBatches(
            nSamples, nTasks, nBatch, alsoSampleTest=alsoSampleTest)

        return float(tf.reduce_mean(tf.map_fn(lambda batch: self.update(
            batch), elems=batch, fn_output_signature=tf.float32) / nTasks))

    def __deepCloneModel(self, model):
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        return clone
