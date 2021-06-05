import tensorflow as tf
from tftools import TensorflowTools


class RegressionMAML:
    def __init__(self, model, taskDistribution, metaLearningRate=0.001):
        self.metaModel = TensorflowTools.loadModelFromContext(model)
        self.weights = self.metaModel.trainable_variables
        self.taskDistribution = taskDistribution

        self.metaOptimizer = tf.keras.optimizers.Adam(metaLearningRate)
        self.mse = tf.keras.losses.MeanSquaredError()

    def steps(self, y, x, nSteps=1):
        clone = TensorflowTools.deepCloneModel(self.metaModel)
        for _ in range(nSteps):
            with tf.GradientTape() as taskTape:
                loss = self.mse(y, clone(x))

            grads = taskTape.gradient(loss, clone.trainable_weights)
            tf.keras.optimizers.SGD(0.01).apply_gradients(
                zip(grads, clone.trainable_weights))
        return clone
    
    def saveKeras(self, path):
        TensorflowTools.saveKeras(self.metaModel, path)

    @tf.function
    def step(self, grads, x):
        k = 0
        y = tf.reshape(x, (-1, 1))
        for j in range(len(self.metaModel.layers)):
            kernel = self.metaModel.layers[j].kernel - 0.01 * grads[k]
            bias = self.metaModel.layers[j].bias - 0.01 * grads[k+1]
            y = self.metaModel.layers[j].activation(y @ kernel + bias)
            k += 2
        return y

    @tf.function
    def loss(self, y_train, x_train, y_test, x_test):
        def taskLoss(batch):
            ya, xa, yb, xb = batch
            with tf.GradientTape() as taskTape:
                loss = self.mse(ya, self.metaModel(tf.reshape(xa, (-1, 1))))

            grads = taskTape.gradient(loss, self.weights)
            return (self.mse(yb, self.step(grads, xb)), 0, 0, 0)

        with tf.GradientTape() as metaTape:
            batch = (y_train, x_train, y_test, x_test)
            losses, _, _, _ = tf.map_fn(taskLoss, elems=batch)
            loss = tf.reduce_sum(losses)

        grads = metaTape.gradient(loss, self.weights)
        self.metaOptimizer.apply_gradients(zip(grads, self.weights))
        return loss

    def trainBatch(self, nSamples, nTasks, nBatch):
        def metaLoss(batch):
            return (self.loss(*batch), 0, 0, 0)

        batch = self.taskDistribution.sampleTaskBatches(
            nSamples, nTasks, nBatch)

        losses, _, _, _ = tf.map_fn(metaLoss, elems=batch)

        return float(tf.reduce_mean(losses / nTasks))
        
