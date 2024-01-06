import tensorflow as tf

class ClassifierNetwork(tf.keras.Model):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        self.sequence = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
    def call(self, X: tf.Tensor) -> tf.Tensor:
        y_hat = self.sequence(X)
        return y_hat
