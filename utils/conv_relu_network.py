import tensorflow as tf
from utils.conv_sym_network import Encoder


class ReLUNet:
    def __init__(self, dense_dim_list=[64, 256, 1024, 128], output_dim=1):
        self.dense_list = [tf.keras.layers.Dense(units=dense_dim, activation=tf.nn.relu) for dense_dim in dense_dim_list]
        self.output_weight = tf.keras.layers.Dense(units=output_dim)
    
    def __call__(self, t_batch, h):
        t_batch = tf.expand_dims(t_batch, axis=2)
        t_batch_tiled = tf.tile(t_batch, [1, tf.shape(h)[1], 1])
        h = tf.concat([h, t_batch_tiled], axis=2)

        y = tf.reshape(h, [-1, h.shape[2]])

        for dense_layer in self.dense_list:
            y = dense_layer(y)
        y = self.output_weight(y)

        y = tf.reshape(y, [tf.shape(h)[0], tf.shape(h)[1], -1])

        return y


class ReLUConvNet:
    def __init__(self, relu_dense_dim_list=[64, 256, 1024, 128], output_dim=1, kernel_size=5, kernel_stride=1, pool_size=2,
                 filter_num_list=[32, 64], enc_dense_dim_list=[128, 16], latent_dim=1):
        self.encoder = Encoder(kernel_size, kernel_stride, pool_size, filter_num_list, enc_dense_dim_list, latent_dim)
        self.relu_net = ReLUNet(relu_dense_dim_list, output_dim)

    def __call__(self, t_batch, h):
        l = tf.reshape(h, [-1, h.shape[3], 1])
        l = self.encoder(l)
        l = tf.reshape(l, [tf.shape(h)[0], tf.shape(h)[1], h.shape[2]])
        y = self.relu_net(t_batch, l)

        return l, y