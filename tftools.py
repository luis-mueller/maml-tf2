import tensorflow as tf 

class TensorflowTools:
    def deepCloneModel(model):
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        return clone
