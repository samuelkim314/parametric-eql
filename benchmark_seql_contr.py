"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import time
import pickle
import tensorflow as tf
import numpy as np
import os
from utils import functions, pretty_print
from utils.para_sym_network import StackedEQL
from utils.schedules import get_lr, get_rw
from generate_data.contr import gen_lin_func_data, gen_f1_func_data, gen_sin_func_data
from inspect import signature
import benchmark
import argparse


N_TRAIN = 512       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-3, 3)    # Domain of dataset
N_TEST = 256        # Size of test dataset
DOMAIN_TEST = (-5, 5)
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x", "y", "z", "a"]

numW = 128
TIME_RANGE = (0, 3)

X_MINI_BATCH_SIZE = 16

RES_DIRECTORY = "benchmark-seql-contr/"

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.5
init_sd_last = 0.5
init_sd_middle = 0.5


class Benchmark(benchmark.Benchmark):
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""

    def __init__(self, results_dir, numW=2, n_layers=2, reg_weight=1e-2, learning_rate=1e-2,
                 n_epochs1=20001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            # *[functions.Exp()] * 2,
            # *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]

        self.numW = numW
        self.n_layers = n_layers            # Number of hidden layers
        self.reg_weight = reg_weight     # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 200    # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)

    def benchmark(self, func, func_name, trials, train_data, test_data):
        """Benchmark the EQL network on data generated by the given function. Print the results ordered by test error.

        Arguments:
            func: lambda function to generate dataset
            func_name: string that describes the function - this will be the directory name
            trials: number of trials to train from scratch. Will save the results for each trial.
        """

        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        # Create a new sub-directory just for the specific function
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Save train and test data
        x_train, y_train, t_batch_train = train_data
        np.save("results/" + RES_DIRECTORY + func_name + "/x_train.npy", x_train)
        np.save("results/" + RES_DIRECTORY + func_name + "/y_train.npy", y_train)
        np.save("results/" + RES_DIRECTORY + func_name + "/t_batch_train.npy", t_batch_train)

        x_test, y_test, t_batch_test = test_data
        np.save("results/" + RES_DIRECTORY + func_name + "/x_test.npy", x_test)
        np.save("results/" + RES_DIRECTORY + func_name + "/y_test.npy", y_test)
        np.save("results/" + RES_DIRECTORY + func_name + "/t_batch_test.npy", t_batch_test)

        # Train network!
        expr_list, error_test_list = self.train(
            func, train_data, test_data, func_name, trials, func_dir)

        # Sort the results by test error (increasing) and print them to file
        # This allows us to easily count how many times it fit correctly.
        # List of (error, expr)
        error_expr_sorted = sorted(zip(error_test_list, expr_list))
        # Separating out the errors
        error_test_sorted = [x for x, _ in error_expr_sorted]
        # Separating out the expr
        expr_list_sorted = [x for _, x in error_expr_sorted]

        fi = open(os.path.join(self.results_dir, 'eq_summary.txt'), 'a')
        fi.write("\n{}\n".format(func_name))
        for i in range(trials):
            fi.write("[%f]\t\t%s\n" %
                     (error_test_sorted[i], str(expr_list_sorted[i])))
        fi.close()

    def train(self, func, train_data, test_data, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        # Number of input arguments to the function
        x_dim = len(signature(func).parameters)
        # Generate training data and test data
        x, y, t_batch = train_data
        x_test, y_test, t_batch_test = test_data

        # finding values for normalization
        x_max_np = np.amax(x, axis=(0,1), keepdims=True)
        x_min_np = np.amin(x, axis=(0,1), keepdims=True)
        x_scale_np = np.maximum(x_max_np, -1*x_min_np)
        x = x / x_scale_np
        x_test = x_test / x_scale_np

        y_max_np = np.amax(y, axis=1, keepdims=True)
        y_min_np = np.amin(y, axis=1, keepdims=True)
        y_range_np = y_max_np-y_min_np

        for i in range(y_range_np.shape[0]):
            if y_range_np[i][0][0] == 0:
                y_range_np[i] = 1

        y_max = tf.cast(tf.reduce_max(y, axis=1, keepdims=True), tf.float32)
        y_min = tf.cast(tf.reduce_min(y, axis=1, keepdims=True), tf.float32)
        y_range = y_max-y_min

        y_range = tf.convert_to_tensor(y_range_np, dtype=tf.float32)

        # Setting up the symbolic regression network
        x_placeholder = tf.placeholder(
            shape=(None, None, x_dim), dtype=tf.float32)
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
        sym = StackedEQL(self.n_layers, funcs=self.activation_funcs, numW=numW)
        y_hat = sym(x_placeholder)

        # Label and errors
        y_placeholder = tf.placeholder(
            shape=(numW, None, 1), dtype=tf.float32)
        y_max_placeholder = tf.placeholder(
            shape=(numW, 1, 1), dtype=tf.float32)
        y_min_placeholder = tf.placeholder(
            shape=(numW, 1, 1), dtype=tf.float32)
        y_range_placeholder = tf.placeholder(
            shape=(numW, 1, 1), dtype=tf.float32)
        error = tf.losses.mean_squared_error(
            labels=(y_placeholder-y_min_placeholder)/y_range_placeholder, predictions=y_hat)
        error_test = tf.losses.mean_squared_error(
            labels=(y_test-y_min)/y_range, predictions=y_hat)

        reg_weight = tf.placeholder(tf.float32)
        reg_loss = sym.get_loss()
        loss = error + reg_weight * reg_loss

        # Training
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # gradients, variables = zip(*opt.compute_gradients(loss))
        # gradients_clipped, _ = tf.clip_by_global_norm(gradients, 1e-4)
        # train = opt.apply_gradients(zip(gradients_clipped, variables))
        train = opt.minimize(loss)

        loss_list = []  # Total loss (MSE + regularization)
        error_list = []     # MSE
        reg_list = []       # Regularization
        error_test_list = []    # Test error

        error_test_final = []
        eq_list = []

        # Only take GPU memory as needed - allows multiple jobs on a single GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for trial in range(trials):
                print("Training on function " + func_name + " Trial " +
                      str(trial+1) + " out of " + str(trials))

                trial_start_time = time.time()

                loss_val = np.nan
                # Restart training if loss goes to NaN (which happens when gradients blow up)
                while np.isnan(loss_val):
                    sess.run(tf.global_variables_initializer())
                    # 1st stage of training with oscillating regularization weight
                    for i in range(self.n_epochs1):
                        x_idx_list = np.arange(N_TRAIN)
                        np.random.shuffle(x_idx_list)

                        x_batches = []
                        y_batches = []

                        y_max_batches = []
                        y_min_batches = []
                        y_range_batches = []

                        for jx in range(N_TRAIN//X_MINI_BATCH_SIZE):
                            xind = x_idx_list[X_MINI_BATCH_SIZE *
                                                jx:X_MINI_BATCH_SIZE*(jx+1)]

                            x_batches.append(x[:, xind, :])
                            y_batches.append(y[:, xind, :])

                            y_max_batches.append(y_max_np)
                            y_min_batches.append(y_min_np)
                            y_range_batches.append(y_range_np)

                        for k in range(len(x_batches)):
                            feed_dict = {x_placeholder: x_batches[k], y_placeholder: y_batches[k],
                                         y_max_placeholder: y_max_batches[k], y_min_placeholder: y_min_batches[k], y_range_placeholder: y_range_batches[k], learning_rate: get_lr(self.learning_rate, i, self.n_epochs1), reg_weight: get_rw(self.reg_weight, i, self.n_epochs1)}
                            _ = sess.run(train, feed_dict=feed_dict)

                        if i % self.summary_step == 0:
                            loss_val, error_val, reg_val = sess.run(
                                (loss, error, reg_loss), feed_dict=feed_dict)
                            error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                            print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (
                                i, loss_val, error_test_val))
                            loss_list.append(loss_val)
                            error_list.append(error_val)
                            reg_list.append(reg_val)
                            error_test_list.append(error_test_val)
                            if np.isnan(loss_val):  # If loss goes to NaN, restart training
                                break

                trial_end_time = time.time()

                # Print the expressions
                weights = sess.run(sym.get_weights())
                expr = []
                for i in range(4):
                    expr_idx = (numW * i) // 4
                    cur_weights = [w[expr_idx]
                                   for w in weights]
                    expr += [y_range[expr_idx, 0, 0].eval() * pretty_print.network(
                        cur_weights, self.activation_funcs, var_names[:x_dim]) + y_min[expr_idx, 0, 0].eval()]
                print(expr)
                print("training time:", trial_end_time-trial_start_time)

                # Save results
                trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)

                results = {
                    "weights": weights,
                    "loss_list": loss_list,
                    "error_list": error_list,
                    "reg_list": reg_list,
                    "error_test": error_test_list,
                    "expr": expr,
                    "training_time": trial_end_time-trial_start_time
                }

                with open(trial_file, "wb+") as f:
                    pickle.dump(results, f)

                # Save data
                y_hat_norm = sess.run(y_hat, feed_dict={x_placeholder: x_test})
                y_hat_unnorm = y_hat_norm*y_range + y_min
                np.save("results/" + RES_DIRECTORY + func_name + "/y_hat_" + str(trial) + ".npy", y_hat_unnorm.eval())

                error_test_final.append(error_test_list[-1])
                eq_list.append((expr, trial))

        return eq_list, error_test_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str,
                        default='results/' + RES_DIRECTORY)
    parser.add_argument("--numW", type=int, default=numW,
                        help="Number of time-steps, numW")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float,
                        default=3e-3, help='Regularization weight, lambda') # 3e-3
    parser.add_argument('--learning-rate', type=float,
                        default=1e-2, help='Base learning rate for training') # 1e-2
    parser.add_argument("--n-epochs1", type=int, default=1201,
                        help="Number of epochs to train the first stage")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json

    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    NUM_TRIALS = 40

    x, y, t_batch = gen_lin_func_data(
        N_TRAIN, numW, lambda x: x, range_min=DOMAIN[0], range_max=DOMAIN[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    x_test, y_test, t_batch_test = gen_lin_func_data(
        N_TEST, numW, lambda x: x, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    bench.benchmark(lambda x: x, func_name="t*x", trials=NUM_TRIALS, train_data=(x, y, t_batch),
                    test_data=(x_test, y_test, t_batch_test))

    x, y, t_batch = gen_f1_func_data(
        N_TRAIN, numW, lambda x: x, lambda x: 3*np.sin(x), range_min=DOMAIN[0], range_max=DOMAIN[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    x_test, y_test, t_batch_test = gen_f1_func_data(
        N_TEST, numW, lambda x: x, lambda x: 3*np.sin(x), range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    bench.benchmark(lambda x: x**2+x, func_name="t*x**2+3*sin(t)*x", trials=NUM_TRIALS, train_data=(x, y, t_batch),
                    test_data=(x_test, y_test, t_batch_test))

    x, y, t_batch = gen_f1_func_data(
        N_TRAIN, numW, lambda x: x, lambda x: 3*np.sign(x), range_min=DOMAIN[0], range_max=DOMAIN[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    x_test, y_test, t_batch_test = gen_f1_func_data(
        N_TEST, numW, lambda x: x, lambda x: 3*np.sign(x), range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    bench.benchmark(lambda x: x**2+x, func_name="t*x**2+3*sign(t)*x", trials=NUM_TRIALS, train_data=(x, y, t_batch),
                    test_data=(x_test, y_test, t_batch_test))

    x, y, t_batch = gen_sin_func_data(
        N_TRAIN, numW, lambda x: 1, lambda x: 1/2*(5+x), range_min=DOMAIN[0], range_max=DOMAIN[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    x_test, y_test, t_batch_test = gen_sin_func_data(
        N_TEST, numW, lambda x: 1, lambda x: 1/2*(5+x), range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    bench.benchmark(lambda x: np.sin(x), func_name="sin(0.5*(5+t)*x)", trials=NUM_TRIALS, train_data=(x, y, t_batch),
                    test_data=(x_test, y_test, t_batch_test))

    def t_fun(t):
        return np.piecewise(t, [t < 0, 0 <= t < 1.5, t >= 1.5],
                            [lambda t: 1/2*(5+t), lambda t: 2.5 - 0.5 * t, lambda t: t - 1.5 + 1.75])
    x, y, t_batch = gen_sin_func_data(
        N_TRAIN, numW, lambda x: 1, t_fun, range_min=DOMAIN[0], range_max=DOMAIN[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    x_test, y_test, t_batch_test = gen_sin_func_data(
        N_TEST, numW, lambda x: 1, t_fun, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1], time_min=TIME_RANGE[0], time_max=TIME_RANGE[1])
    bench.benchmark(lambda x: np.sin(x), func_name="sin(jagged_x)", trials=NUM_TRIALS, train_data=(x, y, t_batch),
                    test_data=(x_test, y_test, t_batch_test))
