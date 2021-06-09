import tensorflow as tf 
from mamltf2.tftools import TensorflowTools

class Model:
    def __init__(self, model, taskDistribution, learningRate=0.001):
        self.model = TensorflowTools.loadModelFromContext(model)
        self.weights = self.model.trainable_variables
        self.taskDistribution = taskDistribution

        self.optimizer = tf.keras.optimizers.Adam(learningRate)
        self.mse = tf.keras.losses.MeanSquaredError()

    def saveKeras(self, path):
        TensorflowTools.saveKeras(self.model, path)

    def steps(self, y, x, nSteps=1):
        clone = TensorflowTools.deepCloneModel(self.model)
        for _ in range(nSteps):
            with tf.GradientTape() as taskTape:
                loss = self.mse(y, clone(x))

            grads = taskTape.gradient(loss, clone.trainable_weights)
            tf.keras.optimizers.SGD(0.01).apply_gradients(
                zip(grads, clone.trainable_weights))
        return clone