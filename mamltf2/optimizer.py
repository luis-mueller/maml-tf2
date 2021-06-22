
import tensorflow as tf 

class FastWeights:
    def __init__(self, complement, lr = 0.01):
        """For the MAML implementation we need an optimizer that can apply SGD using the weights of one 
        neural net to another neural net. This optimizer can apply gradients and compute a forward pass in one 
        go,to compute updated forward pass as easy as possible. Always coupled to one specific neural net that 
        it 'complements'.
        """
        self.lr = lr
        self.complement = complement
    
    @tf.function
    def __call__(self, grads, input, nSteps = 1):
        """Compute fast weights and apply them for a forward pass. It seems that tensorflow 
        is not able to differentiate through an optimizer's steps, hence this implementation. In the meta-validation step this can 
        be replaced by a proper tf.keras.optimizers.SGD instance.
        """
        k = 0
        output = tf.reshape(input, (-1, 1))
        for j in range(len(self.complement.layers)):
            kernel = self.complement.layers[j].kernel
            bias = self.complement.layers[j].bias

            for _ in range(nSteps):
                kernel = kernel - self.lr * grads[k]
                bias = bias - self.lr * grads[k+1]

            output = self.complement.layers[j].activation(output @ kernel + bias)
            k += 2
        return output