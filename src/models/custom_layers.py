import tensorflow as tf
from tensorflow.keras.layers import Layer

class AbsoluteDifference(Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

    def get_config(self):
        return super().get_config()