
import tensorflow as tf 

class FastWeights:
    def __init__(self, model, lr = 0.01):
        """For the MAML implementation we need an optimizer that can apply SGD using the weights of one 
        neural net to another neural net. This optimizer can apply gradients and compute a forward pass in one 
        go,to compute updated forward pass as easy as possible. Always coupled to one specific neural net that 
        it 'models'.
        """
        self.lr = lr
        self.model = model

    @tf.function 
    def __call__(self, weights, input):
        output = tf.reshape(input, (-1, 1))
        for j in range(len(self.model.layers)):
            kernel, bias = weights[j * 2], weights[j * 2 + 1]
            output = self.model.layers[j].activation(output @ kernel + bias)
        return output

    @tf.function 
    def computeUpdate(self, grads_and_vars):
        return [ variable - self.lr * grad for grad, variable in grads_and_vars]