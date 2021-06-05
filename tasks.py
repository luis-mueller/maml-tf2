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

    def sampleTaskBatch(self, nSamples, nTasks):
        A = tf.random.uniform((nTasks,), 0.1, 5.0)
        p = tf.random.uniform((nTasks,), 0, np.pi)
        t = tf.random.uniform((nSamples, nTasks), -5.0, 5.0)
        u = tf.random.uniform((nSamples, nTasks), -5.0, 5.0)
        return tf.transpose(A * tf.sin(t - p)), tf.transpose(t), tf.transpose(A * tf.sin(u - p)), tf.transpose(u)

    def sampleTaskBatches(self, nSamples, nTasks, nBatch):
        A = tf.random.uniform((nTasks, 1), 0.1, 5.0)
        p = tf.random.uniform((nTasks, 1), 0, np.pi)
        t = tf.random.uniform((nBatch, nTasks, nSamples), -5.0, 5.0)
        u = tf.random.uniform((nBatch, nTasks, nSamples), -5.0, 5.0)
        return A * tf.sin(t - p), t, A * tf.sin(u - p), u




