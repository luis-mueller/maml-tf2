import unittest
import tensorflow as tf
from random import randint
import numpy as np



from mamltf2.methods.imaml import conjugate_gradient


def sample_vector(dims=10):
    return np.random.normal(size=(dims,))

def sample_factor():
    return (-1 if randint(0,1) else 1) * np.exp(sample_vector(1))

def apply_random_elementary_operation(A):
    operation_type = randint(0,2)
    dims = A.shape[0]

    if operation_type == 0:  # row switching
        # sample source and target
        i = randint(0, dims-1)
        j = randint(0, dims-2)

        if j >=i: j+=1

        # switch rows
        A[[i, j]] = A[[j, i]]

    elif operation_type == 1:  # row multiplication
        i = randint(0, dims-1)  # sample row
        A[i, :] = A[i, :] * sample_factor()  # multiply

    else:  # row addition
        # sample source and target
        i = randint(0, dims-1)
        j = randint(0, dims-2)
        if j >=i: j+=1
        A[i] = A[i] + sample_vector(1) * A[j]

    return A

def sample_invertible_matrix(dims=10, n_ops=10):
    A = np.identity(dims)
    for _ in range(n_ops):
        A = apply_random_elementary_operation(A)
    return A


class TestConjugateGradient(unittest.TestCase):
    def test_solve(self):
        for _ in range(20):
            A = np.array([[1, 0], [0,2]], dtype=float)
            target = np.array([1, 1], dtype=float)

            #A = sample_invertible_matrix(dims=2))
            A = tf.convert_to_tensor(A)

            #target = sample_vector(dims=2)
            target = [tf.convert_to_tensor(target)]

            def operator(v):
                return [tf.tensordot(A, v[0], 1)]

            b = operator(target)

            x = conjugate_gradient(operator, b)

            error = tf.sqrt(tf.reduce_sum(
                [tf.square(a - b) for a, b, in zip(target, x)]
            ))

            self.assertTrue(error <= 1e-5)


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



if __name__ == '__main__':
    unittest.main()
    print("Done")
