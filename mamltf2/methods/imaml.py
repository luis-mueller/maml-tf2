import tensorflow as tf
from mamltf2.model import Model

@tf.function
def add_mul(a, b, s=1):
    return [a[i] + s*b[i] for i in range(len(a))]

@tf.function
def dot(a, b):
    return tf.reduce_sum([tf.reduce_sum(tf.multiply(ai, bi)) for ai, bi in zip(a, b)])


def conjugate_gradient(operator, b, tolerance=1e-05, max_iterations=20, operator_kwargs={}):
    """ Implements basic conjugate gradient method.
        @operator: function, should return the product of A and v
        @b: target value (right-hand side)
    """

    x = tf.zeros_like(b)  # starting point
    r = add_mul(b, operator(x, **operator_kwargs), -1.0)  # residual

    p = [tf.identity(ri) for ri in r]  # search direction

    r_dot_r = dot(r, r)

    for i in tf.range(max_iterations):

        Ap = operator(p, **operator_kwargs)

        alpha = r_dot_r / dot(p, Ap)

        # update x and residual
        #x = x + alpha*p
        x = add_mul(x, p, alpha)

        #r = r - alpha*Ap
        r = add_mul(r, Ap, -alpha)

        r_dot_r_new = dot(r, r)

        # stopping condition
        #if tf.less(r_dot_r_new, tolerance): break

        # prepare for next iteration
        beta = r_dot_r_new / r_dot_r

        # update search direction
        #p = r + beta * p
        p = add_mul(r, p, beta)

        r_dot_r = r_dot_r_new  # shortcut for next iteration

    return x



def line_search(operator, b, max_iterations=20, operator_kwargs={}):
    # starting point
    x = [tf.zeros_like(bi) for bi in b]

    Ax = operator(x, **operator_kwargs)

    # residual
    r = add_mul(b, Ax, -1.0)

    for i in range(max_iterations):
        r_dot_r = dot(r, r)
        Ar = operator(r, **operator_kwargs)
        rAr = dot(r, Ar)
        if tf.less(rAr, 1e-7): break
        alpha = r_dot_r / rAr

        #x = x + alpha*r
        x = add_mul(x, r, alpha)

        #r = r - alpha*Ar
        r = add_mul(r, Ar, -alpha)

    return x



class IMAML(Model):
    def __init__(self, *args,
                 nInnerSteps = 5,
                 regularizationCoeffiecient=2,
                 solverSteps=20,
                 outerLearningRate = 0.01,
                 innerLearningRate = 0.001,
                 **kwargs):

        super().__init__(*args, outerLearningRate=outerLearningRate, innerLearningRate=innerLearningRate, **kwargs)
        self.nInnerSteps = nInnerSteps
        self.solverSteps = solverSteps
        self.innerOptimizer = tf.keras.optimizers.SGD(self.innerLearningRate)
        self.outerOptimizer = tf.keras.optimizers.SGD(self.outerLearningRate)

        self.model.theta = [
            tf.Variable(tf.zeros_like(variable), trainable=False, name="theta")
            for variable in self.model.trainable_weights
        ]

        self.regularizer = tf.keras.regularizers.L2(regularizationCoeffiecient)
        self.regularizationCoeffiecient = regularizationCoeffiecient

    @tf.function
    def regularizationLoss(self):
        return self.regularizationCoeffiecient * tf.reduce_sum([
            tf.reduce_sum((phi - theta)**2)
            for phi, theta in zip(self.model.trainable_weights, self.model.theta)
        ])

    def theta_to_phi(self):
        """ Initialize parameters with theta."""
        for i in range(len(self.model.trainable_weights)):
            self.model.trainable_weights[i].assign(self.model.theta[i])


    def theta_from_phi(self):
        """ Set theta from current model parameters. This is useful for applying
            an optimizer to theta."""
        for i in range(len(self.model.trainable_weights)):
            self.model.theta[i].assign(self.model.trainable_weights[i])


    @tf.function
    def calcLoss(self, x, y):
        y_pred = self.model(tf.reshape(x, (-1, 1)))
        loss = self.lossfn(y, y_pred)
        return loss

    @tf.function
    def trainOnSamples(self, x, y):
        for _ in range(self.nInnerSteps):
            with tf.GradientTape() as taskTape:
                loss = self.calcLoss(x, y)
                regularizedLoss = loss + self.regularizationLoss()

            self.innerOptimizer.minimize(
                regularizedLoss, self.model.trainable_weights, tape=taskTape)

        return loss

    @tf.function
    def cg_operator(self, v, x, y):
        return add_mul(v, self.hessian_vector_product(x, y, v), 1/self.regularizationCoeffiecient)

    @tf.function
    def hessian_vector_product(self, x, y, v):
        vars = self.model.trainable_weights

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                #loss = self.regularizationLoss()
                loss = self.calcLoss(x, y)
            grads = inner_tape.gradient(loss, vars)
        return outer_tape.gradient(grads, vars, output_gradients=v)

    def update(self, batch):
        """Implements the meta-update step for a bunch of tasks.

        @batch: Tuple of training and test data for the update step. Can be directly passed through to
        the task updates.
        """

        self.theta_from_phi()

        metaGradient = list()
        lossSum = 0

        for y_train, x_train, y_test, x_test in zip(*(tf.unstack(x) for x in batch)):
            # copy meta parameter to fast adapting model
            self.theta_to_phi()


            self.trainOnSamples(x_train, y_train)

            with tf.GradientTape() as tape:
                test_loss = self.calcLoss(x_test, y_test)
            test_loss_gradient = tape.gradient(test_loss, self.model.trainable_weights)

            lossSum += test_loss

            #grad = conjugate_gradient(
            grad = line_search(
                operator=self.cg_operator,
                operator_kwargs=dict(x=x_train, y=y_train),
                b=test_loss_gradient,
                max_iterations=self.solverSteps
            )

            metaGradient.append(grad)


        metaGradient = [tf.reduce_mean(grads, axis=0) for grads in zip(*metaGradient)]


        # Apply gradient

        self.theta_to_phi()

        self.outerOptimizer.apply_gradients(
            list(zip(metaGradient, self.model.trainable_weights))
        )


        return lossSum
