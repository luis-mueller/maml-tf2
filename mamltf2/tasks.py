import numpy as np
import tensorflow as tf

class SinusoidRegressionTask:
    def __init__(self, amplitude, phase):
        self.amplitude = amplitude
        self.phase = phase

    def sampleFromTask(self, nSamples, sampleLower = -5.0, sampleUpper = 5.0):
        t = tf.random.uniform((nSamples, 1), sampleLower, sampleUpper)
        return self(t), t
    
    def __call__(self, t):
        return self.amplitude * np.sin(t - self.phase)

class SinusoidRegressionTaskDistribution:
    def sampleTask(self):
        A = tf.random.uniform((1,), 0.1, 5.0)
        p = tf.random.uniform((1,), 0, np.pi)
        return SinusoidRegressionTask(A, p)

    def sampleTaskBatches(self, nSamples, nTasks, nBatch, sampleLower = -5.0, sampleUpper = 5.0, alsoSampleTest = True):
        A = tf.random.uniform((nTasks, 1), 0.1, 5.0)
        p = tf.random.uniform((nTasks, 1), 0, np.pi)
        t = tf.random.uniform((nBatch, nTasks, nSamples), sampleLower, sampleUpper)
        if alsoSampleTest:
            u = tf.random.uniform((nBatch, nTasks, nSamples), sampleLower, sampleUpper)
            return A * tf.sin(t - p), t, A * tf.sin(u - p), u
        return A * tf.sin(t - p), t




