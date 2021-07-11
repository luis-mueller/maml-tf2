import tensorflowjs as tfjs

class DeployableModel: 
    def __init__(self, model):
        """Implements a model that has I/O interfaces and can be deployed/integrated to a pipeline.
        Merely a helper class to implement a separation of concerns an dto provide an interface for extending 
        I/O functionality.
        """
        self.model = self.__loadModelFromContext(model)

    def saveKeras(self, path):
        """Save the model as a keras model.
        """
        tfjs.converters.save_keras_model(self.model, path)
    
    def __loadModelFromContext(self, model):
        if isinstance(model, str):
            return tfjs.converters.load_keras_model(model)
        return model
    
