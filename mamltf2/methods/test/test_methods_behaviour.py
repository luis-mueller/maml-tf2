import unittest
import tensorflow as tf
from mamltf2 import MAML, FirstOrderMAML, Reptile, SinusoidRegressionTaskDistribution

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestMethodBehaviour(unittest.TestCase):
    def setUp(self):
        """Tests convergence behaviour over a few epochs in compiled mode to 
        see if expected thresholds are crossed. This is no guaranteed measure but 
        the methods all roughly behave a certain way when called with the same params.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(40, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.taskDistribution = SinusoidRegressionTaskDistribution()

    def doTraining(self, methodName, model, nEpochs, nSamples, nTasks, nBatch):
        print("#### %s ####:" % methodName)
        for epoch in range(nEpochs):
            meanBatchLoss = model.trainBatch(nSamples, nTasks, nBatch)

            print(
                "Meta loss @ epoch %d, method %s: %.4f"
                % (epoch, methodName, float(meanBatchLoss))
            )
        return meanBatchLoss

    def test_maml_convbehav(self):
        model = MAML(self.model, self.taskDistribution, lossfn='mse',
                     outerLearningRate=0.001, innerLearningRate=0.01)
        
        loss = self.doTraining('maml', model, 3, 10, 5, 1000)
        self.assertLessEqual(loss, 1.2)

    def test_fomaml_convbehav(self):
        model = FirstOrderMAML(self.model, self.taskDistribution, lossfn='mse',
                     outerLearningRate=0.001, innerLearningRate=0.01)
        
        loss = self.doTraining('fomaml', model, 3, 10, 5, 1000)
        self.assertLessEqual(loss, 1.5)

    def test_reptile_convbehav(self):
        model = Reptile(self.model, self.taskDistribution, lossfn='mse',
                     outerLearningRate=0.1, innerLearningRate=0.02)
        
        loss = self.doTraining('reptile', model, 4, 10, 1, 1000)
        self.assertLessEqual(loss, 1.0)
