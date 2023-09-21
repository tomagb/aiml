import tensorflow as tf
import numpy as np

x = tf.ones(shape=(2,1))
# print(x)

r = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
# print(r)

vx = tf.Variable(initial_value=x)
# print(vx)

vr = tf.Variable(initial_value=r)
# print(vr)

input_var = tf.Variable(initial_value = 3.)

with tf.GradientTape() as tape:
    result = tf.square(input_var)
    gradient = tape.gradient(result, input_var)

    # print(gradient)


# 3.6.1

class SimpleDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

denseLayer = tf.keras.layers.Dense(32, activation="relu")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32)
])

print(model)
