import tensorflow as tf
from utils.para_sym_network import HyperEQL, StackedEQL

class Encoder:
    def __init__(self, kernel_size, kernel_stride, pool_size, filter_num_list, dense_dim_list, latent_dim):
        self.conv_list = [tf.keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, strides=kernel_stride, padding='same') for filter_num in filter_num_list]
        self.pool_list = [tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same') for _ in filter_num_list]
        self.flatten = tf.keras.layers.Flatten()
        self.dense_list = [tf.keras.layers.Dense(units=dense_dim, activation=tf.nn.relu) for dense_dim in dense_dim_list]
        self.output_weight = tf.keras.layers.Dense(units=latent_dim)
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, h):
        for conv_layer, pool_layer in zip(self.conv_list, self.pool_list):
            h = conv_layer(h)
            h = pool_layer(h)
        h = self.flatten(h)
        for dense_layer in self.dense_list:
            h = dense_layer(h)
        h = self.output_weight(h)
        h = self.batch_norm(h)

        return h
        

class ParameterizedConvNet:
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1, kernel_size=5, kernel_stride=1, pool_size=2,
                 filter_num_list=[32, 64], dense_dim_list=[128, 16], latent_dim=1):
        self.encoder = Encoder(kernel_size, kernel_stride, pool_size, filter_num_list, dense_dim_list, latent_dim)
        self.parameterized_net = HyperEQL(symbolic_depth, funcs, initial_weights, initial_bias, variable, init_stddev)
    
    def __call__(self, t_batch, h, sample=True, reuse_u=False):
        l = tf.reshape(h, [-1, h.shape[3], 1])
        l = self.encoder(l)
        l = tf.reshape(l, [tf.shape(h)[0], tf.shape(h)[1], h.shape[2]])
        y = self.parameterized_net(t_batch, l, sample, reuse_u)

        return l, y

    def get_loss(self):
        return self.parameterized_net.get_loss()
    
    def get_weights(self, t_batch):
        return self.parameterized_net.get_weights(t_batch)


class StackedConvNet:
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1, numW=64, kernel_size=5, kernel_stride=1, pool_size=2,
                 filter_num_list=[32, 64], dense_dim_list=[128, 16], latent_dim=1):
        self.encoder = Encoder(kernel_size, kernel_stride, pool_size, filter_num_list, dense_dim_list, latent_dim)
        self.stacked_net = StackedEQL(symbolic_depth, funcs, initial_weights, initial_bias, variable, init_stddev, numW)
    
    def __call__(self, h, sample=True, reuse_u=False):
        l = tf.reshape(h, [-1, h.shape[3], 1])
        l = self.encoder(l)
        l = tf.reshape(l, [tf.shape(h)[0], tf.shape(h)[1], h.shape[2]])
        y = self.stacked_net(l, sample, reuse_u)

        return l, y

    def get_loss(self):
        return self.stacked_net.get_loss()
    
    def get_weights(self):
        return self.stacked_net.get_weights()