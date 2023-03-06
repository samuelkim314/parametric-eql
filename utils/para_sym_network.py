import tensorflow as tf
from utils.symbolic_network import SymbolicLayerL0, SymbolicNetL0


class SharedZLayerL0(SymbolicLayerL0):
    """A layer with multiple weight matrices for different time-steps but a shared z matrix"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1, bias=False, droprate_init=0.5, lamba=1., numW=2):
        super().__init__(funcs, initial_weight, variable,
                         init_stddev, bias, droprate_init, lamba)
        self.numW = numW

    def build(self, in_dim):
        with tf.name_scope("symbolic_layer"):
            self.in_dim = in_dim
            if self.W is None:
                self.W = tf.Variable(tf.random_normal(
                    (self.numW, in_dim, self.out_dim), stddev=self.init_stddev))
            if self.use_bias:
                self.bias = tf.Variable(
                    0.1*tf.ones((self.numW, 1, self.out_dim)))
            self.qz_log_alpha = tf.Variable(tf.random_normal((in_dim, self.out_dim),
                                                             mean=tf.log(
                                                                 1-self.droprate_init) - tf.log(self.droprate_init),
                                                             stddev=1e-2))

    def __call__(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        with tf.name_scope("symbolic_layer"):
            if self.W is None or self.qz_log_alpha is None:
                self.build(x.shape[2].value)

            if sample:
                h = tf.matmul(x, self.sample_weights(reuse_u=reuse_u))
            else:
                w = self.get_weight()
                h = tf.matmul(x, w)

            if self.use_bias:
                h = h + self.bias

            self.output = []
            # apply a different activation unit to each column of h
            in_i = 0    # input index
            out_i = 0   # output index
            # Apply functions with only a single input
            while out_i < self.n_single:
                self.output.append(self.funcs[out_i](h[:, :, in_i]))
                in_i += 1
                out_i += 1
            # Apply functions that take 2 inputs and produce 1 output
            while out_i < self.n_funcs:
                self.output.append(self.funcs[out_i](
                    h[:, :, in_i], h[:, :, in_i+1]))
                in_i += 2
                out_i += 1
            self.output = tf.stack(self.output, axis=2)
            return self.output


class SharedZNetL0(SymbolicNetL0):
    """A neural net with multiple weight matricies for different time-steps but a shared z matrix at each layer"""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1, numW=2):
        super().__init__(symbolic_depth, funcs, initial_weights,
                         initial_bias, variable, init_stddev)
        if initial_weights is not None:
            self.symbolic_layers = [SharedZLayerL0(funcs=funcs, initial_weight=initial_weights[i], variable=variable, numW=numW)
                                    for i in range(self.depth)]
            if not variable:
                self.output_weight = tf.Variable(initial_weights[-1])
            else:
                self.output_weight = initial_weights[-1]
        else:
            # Each layer initializes its own weights
            if isinstance(init_stddev, list):
                self.symbolic_layers = [SharedZLayerL0(funcs=funcs, init_stddev=init_stddev[i], numW=numW)
                                        for i in range(self.depth)]
            else:
                self.symbolic_layers = [SharedZLayerL0(funcs=funcs, init_stddev=init_stddev, numW=numW)
                                        for _ in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = tf.Variable(tf.random_uniform(
                shape=(numW, self.symbolic_layers[-1].n_funcs, 1)))

    def __call__(self, input, sample=True, reuse_u=False):
        self.shape = (int(input.shape[2]), 1)
        # connect output from previous layer to input of next layer
        h = input
        for i in range(self.depth):
            h = self.symbolic_layers[i](h, sample=sample, reuse_u=reuse_u)
        # Final output (no activation units) of network
        h = tf.matmul(h, self.output_weight)
        return h


class StackedEQL(SharedZNetL0):
    """SEQL implementation"""
    def get_loss(self):
        adj_time_loss = 0
        for i in range(self.depth-1):
            layer = self.symbolic_layers[i].W
            for j in range(layer.shape[0]-1):
                adj_time_loss += tf.norm(layer[j+1]-layer[j])
        return tf.reduce_sum([self.symbolic_layers[i].loss() for i in range(self.depth)]) + 1e-3 * adj_time_loss

    def __call__(self, input, sample=True, reuse_u=False):
        self.shape = (int(input.shape[2]), 1)     # Dimensionality of the input
        # connect output from previous layer to input of next layer
        h = input
        saved_h = [h]
        for i in range(self.depth):
            if i == self.depth-1:
                h = self.symbolic_layers[i](
                    h, sample=sample, reuse_u=reuse_u)
            else:
                h = self.symbolic_layers[i](
                    h, sample=sample, reuse_u=reuse_u)
                h = tf.concat([h, saved_h[-1]], axis=2)
            saved_h.append(h)
        # Final output (no activation units) of network
        h = tf.matmul(h, self.output_weight)
        return h


class WeightLayer:
    """Implementation for one layer of the meta-weight unit"""
    def __init__(self, in_dim, out_dim, use_bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.W = tf.Variable(tf.random_normal(
            shape=[self.in_dim, self.out_dim], stddev=1.5e-1))
        if self.use_bias:
            self.bias = tf.Variable(tf.random_normal(
                shape=[self.out_dim], stddev=3e-2))

    def __call__(self, x):
        h = tf.matmul(x, self.W)
        if self.use_bias:
            h = h + self.bias
        return h


class WeightNet:
    """Meta-weight unit implementation"""
    def __init__(self, mid_dims, out_dim, act_func=tf.nn.relu, use_bias=True):
        self.act_func = act_func

        self.dims = [1] + mid_dims + [out_dim]
        self.weight_layers = [WeightLayer(
            self.dims[i], self.dims[i+1], use_bias) for i in range(len(self.dims)-1)]

    def __call__(self, t):
        h = t

        for i in range(len(self.weight_layers)-1):
            h = self.weight_layers[i](h)
            h = self.act_func(h)

        h = self.weight_layers[-1](h)

        return h


class HyperEQLLayer(SymbolicLayerL0):
    """Implementation for one layer of the HEQL"""
    def build(self, in_dim):
        with tf.name_scope("symbolic_layer"):
            self.in_dim = in_dim
            if self.W is None:
                self.W = WeightNet([64, 64, 256], in_dim*self.out_dim)
            if self.use_bias:
                self.bias = tf.Variable(0.1*tf.ones((1, self.out_dim)))
            self.qz_log_alpha = tf.Variable(tf.random_normal((in_dim, self.out_dim),
                                                             mean=tf.log(
                                                                 1-self.droprate_init) - tf.log(self.droprate_init),
                                                             stddev=1e-2))

    def sample_weights(self, t_batch, reuse_u=False):
        z = self.quantile_concrete(self.sample_u(
            (self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = tf.clip_by_value(z, clip_value_min=0.0, clip_value_max=1.0)
        return mask * tf.reshape(self.W(t_batch), [-1, self.in_dim, self.out_dim])

    def get_weight(self, t_batch):
        """Deterministic value of weight based on mean of z"""
        return self.get_z_mean() * tf.reshape(self.W(t_batch), [-1, self.in_dim, self.out_dim])

    def __call__(self, t_batch, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        with tf.name_scope("symbolic_layer"):
            if self.W is None or self.qz_log_alpha is None:
                self.build(x.shape[2].value)

            if sample:
                h = tf.matmul(x, self.sample_weights(t_batch, reuse_u=reuse_u))
            else:
                w = self.get_weight(t_batch)
                h = tf.matmul(x, w)

            if self.use_bias:
                h = h + self.bias

            self.output = []
            # apply a different activation unit to each column of h
            in_i = 0    # input index
            out_i = 0   # output index
            # Apply functions with only a single input
            while out_i < self.n_single:
                self.output.append(self.funcs[out_i](h[:, :, in_i]))
                in_i += 1
                out_i += 1
            # Apply functions that take 2 inputs and produce 1 output
            while out_i < self.n_funcs:
                self.output.append(self.funcs[out_i](
                    h[:, :, in_i], h[:, :, in_i+1]))
                in_i += 2
                out_i += 1
            self.output = tf.stack(self.output, axis=2)
            return self.output


class HyperEQL(SymbolicNetL0):
    """HEQL implementation"""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1):
        super().__init__(symbolic_depth, funcs, initial_weights,
                         initial_bias, variable, init_stddev)

        self.symbolic_layers = [HyperEQLLayer(
            funcs=funcs) for i in range(self.depth)]
        self.output_weight = tf.Variable(tf.random_uniform(
            shape=(self.symbolic_layers[-1].n_funcs, 1), maxval=1))

    def __call__(self, t_batch, input, sample=True, reuse_u=False):
        self.shape = (int(input.shape[2]), 1)     # Dimensionality of the input
        # connect output from previous layer to input of next layer
        h = input
        saved_h = [h]
        for i in range(self.depth):
            if i == self.depth-1:
                h = self.symbolic_layers[i](
                    t_batch, h, sample=sample, reuse_u=reuse_u)
            else:
                h = self.symbolic_layers[i](
                    t_batch, h, sample=sample, reuse_u=reuse_u)
                # saved_h.append(h)
                h = tf.concat([h, saved_h[-1]], axis=2)
            saved_h.append(h)
        # Final output (no activation units) of network
        h = tf.matmul(h, self.output_weight)
        return h

    def get_loss(self):
        return tf.reduce_sum([self.symbolic_layers[i].loss() for i in range(self.depth)])

    def get_weights(self, t_batch):
        return self.get_symbolic_weights(t_batch) + [self.get_output_weight()]

    def get_symbolic_weights(self, t_batch):
        return [self.symbolic_layers[i].get_weight(t_batch) for i in range(self.depth)]
