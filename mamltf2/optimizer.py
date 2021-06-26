
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
    def __call__(self, grads, input, nSteps = 1):
        """Compute fast weights and apply them for a forward pass. It seems that tensorflow 
        is not able to differentiate through an optimizer's steps, hence this implementation. In the meta-validation step this can 
        be replaced by a proper tf.keras.optimizers.SGD instance.
        
        k = 0
        output = tf.reshape(input, (-1, 1))
        for j in range(len(self.model.layers)):
            kernel = self.model.layers[j].kernel
            bias = self.model.layers[j].bias

            for _ in range(nSteps):
                kernel = kernel - self.lr * grads[k]
                bias = bias - self.lr * grads[k+1]

            output = self.model.layers[j].activation(output @ kernel + bias)
            k += 2
        return output
        """
        output = tf.reshape(input, (-1, 1))
        for j in range(len(self.model.layers)):
            kernel, bias = self.__computeKernelAndBias(j, grads, nSteps)
            output = self.model.layers[j].activation(output @ kernel + bias)
        return output

    @tf.function
    def compute(self, grads, nSteps = 1):
        """Computes fast weights and returns them in an list.
        """
        nLayers = len(self.model.layers)
        return [ weights for j in range(nLayers) for weights in self.__computeKernelAndBias(j, grads, nSteps) ]

    @tf.function
    def apply(self, weights):
        """Apply a set of weights to the model, layer by layer.
        """
        for j in range(len(self.model.trainable_weights)):
            
            self.model.trainable_weights[j].assign(weights[j])

    @tf.function
    def __computeKernelAndBias(self, layerIndex, grads, nSteps):
        """Computes fast weights for a number of steps for one layer.
        """
        kernel = self.model.trainable_weights[layerIndex * 2]#self.model.layers[layerIndex].kernel
        bias = self.model.trainable_weights[layerIndex * 2 + 1]#self.model.layers[layerIndex].bias

        for _ in range(nSteps):
            kernel = kernel - self.lr * grads[layerIndex * 2]
            bias = bias - self.lr * grads[layerIndex * 2 + 1]

        return kernel, bias