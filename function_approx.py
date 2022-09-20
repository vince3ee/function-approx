import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_seed(0)

inputs = np.arange(-3,3,0.2)
a = inputs.reshape((1,len(inputs)))

x = np.arange(-3,3,0.3)
y = x ** 2


class Mymodel:
    def __init__(self, input , target):
        self.input = input

        self.target = target

        self.initializer1 = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed= 1 )

        self.model = tf.keras.Sequential([
            layers.Dense(self.input.shape[1], activation='sigmoid', kernel_initializer = self.initializer1),
            layers.Dense(self.input.shape[1], activation='sigmoid', kernel_initializer = self.initializer1 ),
            layers.Dense(self.input.shape[1], activation='sigmoid', kernel_initializer=self.initializer1),
            layers.Dense(self.input.shape[1], activation='relu', kernel_initializer=self.initializer1),
            layers.Dense(self.input.shape[1], activation='relu', kernel_initializer = self.initializer1),
            layers.Dense(self.input.shape[1], activation='relu', kernel_initializer = self.initializer1),
            layers.Dense(self.input.shape[1] , kernel_initializer = self.initializer1 )
        ])

        self.output  = self.forward_pass()

    def forward_pass(self):
        return self.model(self.input)

    def my_loss_fn(self, y_true, y_pred):
        self.squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(self.squared_difference, axis=-1)  # Note the `axis=-1`

    def train(self):
        self.model.compile(optimizer='RMSprop', loss=self.my_loss_fn)
        self.model.fit(self.output, self.target, epochs=2500, initial_epoch=2000)

        self.output = self.model(self.input)

if __name__ == '__main__':

    model = Mymodel(a, target= a ** 2 )

    model.train()

    plt.scatter(model.input, model.output)
    plt.plot(x,y, 'm')
    plt.show()