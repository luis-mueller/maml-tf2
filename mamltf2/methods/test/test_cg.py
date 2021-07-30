import unittest
import tensorflow as tf
from random import randint
import numpy as np



from mamltf2.methods.imaml import conjugate_gradient, line_search


@tf.function
def operator(v, A):
    return [tf.tensordot(A, v[0], ([-1], [-1]))]

class TestConjugateGradient(unittest.TestCase):
    def test_solve(self):
        for dims in range(2,10):

            # generate a random positive definite matrix
            A = np.random.rand(dims, dims)
            A = 0.5*(A+A.T)
            A = A + dims*np.identity(dims)


            A = tf.convert_to_tensor(A, dtype=tf.float32)

            target = np.random.normal(0, 1, size=(dims,))
            target = [tf.convert_to_tensor(target, dtype=tf.float32)]

            b = operator(target, A)


            x = conjugate_gradient(
                operator, b, operator_kwargs=dict(A=A),
                tolerance=1e-03, max_iterations=dims)

            error = tf.sqrt(tf.reduce_sum(
                [tf.square(a - b) for a, b, in zip(target, x)]
            ))

            error = tf.keras.backend.eval(error)

            self.assertTrue(error <= 1e-3)


class TestLineSearch(unittest.TestCase):
    def test_solve(self):
        for dims in range(2,5):

            # generate a random positive definite matrix
            A = np.random.rand(dims, dims)
            A = 0.5*(A+A.T)
            A = A + dims*np.identity(dims)


            A = tf.convert_to_tensor(A, dtype=tf.float32)

            target = np.random.normal(0, 1, size=(dims,))
            target = [tf.convert_to_tensor(target, dtype=tf.float32)]

            b = operator(target, A)


            x = line_search(
                operator, b, operator_kwargs=dict(A=A),
                max_iterations=20)

            error = tf.sqrt(tf.reduce_sum(
                [tf.square(a - b) for a, b, in zip(target, x)]
            ))

            error = tf.keras.backend.eval(error)

            self.assertTrue(error <= 1e-3)


if __name__ == '__main__':
    unittest.main()
    print("Done")
