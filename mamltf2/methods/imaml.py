import tensorflow as tf
from mamltf2.model import Model


@tf.function
def add_mul(a, b, s=1):
    """Computes a + s*b for lists of tensors a and b and scalar s."""
    return [ai + s*bi for ai, bi in zip(a, b)]

@tf.function
def dot(a, b):
    return tf.reduce_sum([tf.reduce_sum(tf.multiply(ai, bi)) for ai, bi in zip(a, b)])


def conjugate_gradient(operator, b, tolerance=1e-05, max_iterations=20):
    """ Implements basic conjugate gradient method.
        @operator: function, should return the product of A and v
        @b: target value (right-hand side)
    """

    x = [tf.zeros_like(bi) for bi in b]  # starting point


    r = [bi - ox_i for bi, ox_i in zip(b, operator(x))]  # residual
    p = [tf.identity(ri) for ri in r]  # search direction
    r_dot_r = dot(r, r)

    for _ in tf.range(max_iterations):
        Ap = operator(p)
        alpha = r_dot_r / dot(p, Ap)

        # update x and residual
        #x = x + p * alpha
        x = add_mul(x, p, alpha)

        # r = r + Ap * alpha
        r = add_mul(r, Ap, alpha)

        r_dot_r_new = dot(r, r)

        # stopping condition
        if tf.less(r_dot_r_new, tolerance): break

        # prepare for next iteration
        beta = r_dot_r_new / r_dot_r

        # update search direction
        # p = r + p * beta
        p = add_mul(r, p, beta)

        r_dot_r = r_dot_r_new  # shortcut for next iteration

    return x




class IMAML(Model):
    def __init__(self, *args, nInnerSteps = 5, regularizationCoeffiecient=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.nInnerSteps = nInnerSteps
        self.innerOptimizer = tf.keras.optimizers.SGD(self.innerLearningRate)
        self.outerOptimizer = tf.keras.optimizers.SGD(self.outerLearningRate)
        self.modelCopy = tf.keras.models.clone_model(self.model)
        self.regularizer = tf.keras.regularizers.L2(regularizationCoeffiecient)
        self.regularizationCoeffiecient = regularizationCoeffiecient

    @tf.function
    def regularizationLoss(self):
        loss = 0
        for metaParam, taskParam in zip(self.model.trainable_weights,
                                        self.modelCopy.trainable_weights):
            loss += self.regularizer(taskParam-metaParam)
        return loss

    @tf.function
    def initializeTaskParams(self):
        for metaParam, taskParam in zip(self.model.trainable_weights,
                                        self.modelCopy.trainable_weights):
            taskParam.assign(metaParam)

    @tf.function
    def calcLoss(self, x, y):
        y_pred = self.modelCopy(tf.reshape(x, (-1, 1)))
        loss = self.lossfn(y, y_pred)

        return loss

    @tf.function
    def trainOnSamples(self, x, y):
        for _ in range(self.nInnerSteps):
            with tf.GradientTape() as taskTape:
                loss = self.calcLoss(x, y)
                regularizedLoss = loss + self.regularizationLoss()

            self.innerOptimizer.minimize(
                regularizedLoss, self.modelCopy.trainable_variables, tape=taskTape)

        return loss

    def create_operator(self, x, y):
        def operator(v):
            return v + [h/self.regularizationCoeffiecient for h in self.hessian_vector_product(x, y, v)]
        return operator

    @tf.function
    def hessian_vector_product(self, x, y, v):
        variables = self.modelCopy.trainable_variables

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = self.calcLoss(x, y)
            grads = inner_tape.gradient(loss, variables)
        return outer_tape.gradient(grads, variables, output_gradients=v)

    #@tf.function
    def update(self, batch):
        """Implements the meta-update step for a bunch of tasks.

        @batch: Tuple of training and test data for the update step. Can be directly passed through to
        the task updates.
        """

        metaGradients = list()
        lossSum = 0

        for y_train, x_train, y_test, x_test in zip(*(tf.unstack(x) for x in batch)):
            # copy meta parameter to fast adapting model
            self.initializeTaskParams()

            self.trainOnSamples(x_train, y_train)

            with tf.GradientTape() as tape:
                test_loss = self.calcLoss(x_test, y_test)
            test_loss_gradient = tape.gradient(test_loss, self.modelCopy.trainable_variables)


            print(test_loss)

            lossSum += test_loss

            metaGradients.append(conjugate_gradient(
                operator=self.create_operator(x_train, y_train),
                b=test_loss_gradient
            ))

        metaGradient = [tf.reduce_mean(grads, axis=0) for grads in zip(*metaGradients)]

        # Apply gradient
        self.outerOptimizer.apply_gradients(
            list(zip(metaGradient, self.model.trainable_weights))
        )

        return lossSum
