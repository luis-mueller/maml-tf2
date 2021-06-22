from mamltf2.optimizer import FastWeights
import tensorflow as tf
from mamltf2.tftools import TensorflowTools
from mamltf2.optimizer import FastWeights

class Model:
    def __init__(self, model, taskDistribution, learningRate=0.001):
        """Super class for models in this library: Initializes from disk or from a tensorflow layers model.
        """
        self.model = TensorflowTools.loadModelFromContext(model)
        self.weights = self.model.trainable_variables
        self.taskDistribution = taskDistribution

        self.optimizer = tf.keras.optimizers.Adam(learningRate)
        self.sgd = tf.keras.optimizers.SGD(learningRate)
        self.fastWeights = FastWeights(self.model, 0.01)
        self.mse = tf.keras.losses.MeanSquaredError()

    def saveKeras(self, path):
        """Save the model as a keras model.
        """
        TensorflowTools.saveKeras(self.model, path)

    def steps(self, y, x, nSteps=1):
        """For meta-validation: Take a fixed number of gradient steps for the loss given y and x.
        """
        clone = TensorflowTools.deepCloneModel(self.model)
        self.fitClone(clone, y, x, nSteps)
        return clone

    def fitClone(self, clone, y, x, nSteps):
        """Does the actual model fitting with a simple step-size.
        """
        for _ in range(nSteps):
            with tf.GradientTape() as taskTape:
                loss = self.mse(y, clone(x))

            tf.keras.optimizers.SGD(0.01).minimize(loss, clone.trainable_weights, tape = taskTape)

    def trainBatch(self, nSamples, nTasks, nBatch, alsoSampleTest=True):
        """Utility method to train an entire episode/epoch in one go, sampling the entire data at once.
        This should speed up training significantly, since this way the inner update function can be compiled.
        """
        batch = self.taskDistribution.sampleTaskBatches(
            nSamples, nTasks, nBatch, alsoSampleTest=alsoSampleTest)

        return float(tf.reduce_mean(tf.map_fn(lambda batch: self.update(
            batch), elems=batch, fn_output_signature=tf.float32) / nTasks))
