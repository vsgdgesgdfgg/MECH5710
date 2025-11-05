import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class CNN(keras.Model):
    def __init__(self, *, params):
        super().__init__()
        self.cnn_layers = [keras.layers.Conv2D(**params['cnn'][i]) for i in range(len(params['cnn']))]
        self.pool_layers = [keras.layers.MaxPool2D(**params['pool'][i]) for i in range(len(params['pool']))]
        self.flatten = keras.layers.Flatten()
        self.output_dense = keras.layers.Dense(**params['output_dense'])
        self.output_act = keras.layers.Activation(**params['output_act'])

    def build(self, input_shape):
        input_shape = input_shape[0]
        for i in range(len(self.cnn_layers)):
            self.cnn_layers[i].build(input_shape)
            input_shape = self.cnn_layers[i].compute_output_shape(input_shape)
            self.pool_layers[i].build(input_shape)
            input_shape = self.pool_layers[i].compute_output_shape(input_shape)
        self.flatten.build(input_shape)
        input_shape = self.flatten.compute_output_shape(input_shape)
        self.output_dense.build(input_shape)
        self.built = True

    def call(self, x, training=False):
        for i in range(len(self.cnn_layers)):
            x = self.cnn_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.flatten(x)
        x = self.output_dense(x)
        x = self.output_act(x)
        return x

    # @tf.function
    # def train_step(self, data):
    #     x, y = data
    #     with tf.GradientTape(persistent=True) as tape:
    #         for l in self.layers:
    #             x = l(x)
    #         loss = self.loss(y, x)
    #         loss = tf.reduce_mean(loss)
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     return {'loss': loss}
    #
    # @tf.function
    # def test_step(self, data):
    #     x, y = data
    #     for l in self.layers:
    #         x = l(x)
    #     loss = self.loss(y, x)
    #     loss = tf.reduce_mean(loss)
    #     return {'loss': loss}


# params = {'params': {
#                         'cnn': [{'filters': 32, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
#                                  'activation': 'relu'},
#                                 {'filters': 128, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
#                                  'activation': 'relu'},
#                                 {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
#                                  'activation': 'relu'},
#                                 ],
#                         'pool': [{'pool_size': (4, 4)},  # (64, 64, 32)
#                                  {'pool_size': (4, 4), },  # (16, 16, 128)
#                                  {'pool_size': (4, 4)},  # (4, 4, 256)
#                                  ],
#                         'output_dense': {'units': 10},
#                         'output_act': {'activation': 'softmax'},
#                     }}
#
# import numpy as np
# model = CNN(**params)
# model.build(input_shape=[(None, 256, 256, 3)])
# model.call(np.zeros((1, 256, 256, 3)))
# model.summary()
# model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy())
# model.save('./model.keras')

