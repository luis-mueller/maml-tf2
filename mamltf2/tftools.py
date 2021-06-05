import tensorflow as tf 
import tensorflowjs as tfjs

class TensorflowTools:
    def deepCloneModel(model):
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        return clone
    
    def loadModelFromContext(model):
        if isinstance(model, str):
            return tfjs.converters.load_keras_model(model)
        return model
    
    def saveKeras(model, path):
        tfjs.converters.save_keras_model(model, path)
