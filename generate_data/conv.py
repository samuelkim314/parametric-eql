import numpy as np
from scipy.stats import norm
from generate_data.contr import gen_spring_force_func_data, gen_spring_potential_func_data


def point_to_conv(conv_mean, conv_std, conv_size, conv_min, conv_max):
    c = np.linspace(conv_min, conv_max, conv_size)
    return norm.pdf(c, loc=conv_mean, scale=conv_std)


def gen_spring_force_conv_data(N, numW, tfunc1, range_min, range_max, time_min, time_max, conv_std=0.1, conv_size=64, conv_min=None, conv_max=None, train=True):
    x, y, t_batch = gen_spring_force_func_data(18*N, numW, tfunc1, range_min, range_max, time_min, time_max)

    x_filtered_list = []
    y_filtered_list = []

    for i in range(len(t_batch)):
        if train:
            mask = (np.abs(x[i, :, 0] - x[i, :, 1]) <= 2/3 * (range_max - range_min))
        else:
            mask = np.full((x.shape[1]), True)
            # mask = (np.abs(x[i, :, 0] - x[i, :, 1]) > 2/3 * (range_max - range_min))
        x_filtered_list.append(x[i, :, :][mask][:N, :])
        y_filtered_list.append(y[i, :, :][mask][:N, :])

    x = np.stack(x_filtered_list, axis=0)
    y = np.stack(y_filtered_list, axis=0)

    if not conv_min:
        conv_min = range_min-0.1*(range_max-range_min)
    if not conv_max:
        conv_max = range_max+0.1*(range_max-range_min)

    x_conv_list = [[[point_to_conv(x[t, n, xidx], conv_std, conv_size, conv_min, conv_max) for xidx in range(2)] for n in range(N)] for t in range(numW)]
    x_conv = np.array(x_conv_list, dtype=np.float32)

    return x, x_conv, y, t_batch


def gen_spring_potential_conv_data(N, numW, tfunc1, range_min, range_max, time_min, time_max, conv_std=0.1, conv_size=64, conv_min=None, conv_max=None, train=True):
    x, y, t_batch = gen_spring_potential_func_data(2*N, numW, tfunc1, range_min, range_max, time_min, time_max)

    x_filtered_list = []
    y_filtered_list = []

    for i in range(len(t_batch)):
        if train:
            mask = (np.abs(x[i, :, 0] - x[i, :, 1]) <= 2/3 * (range_max - range_min))
        else:
            mask = np.full((x.shape[1]), True)
            # mask = (np.abs(x[i, :, 0] - x[i, :, 1]) > 2/3 * (range_max - range_min))
        x_filtered_list.append(x[i, :, :][mask][:N, :])
        y_filtered_list.append(y[i, :, :][mask][:N, :])

    x = np.stack(x_filtered_list, axis=0)
    y = np.stack(y_filtered_list, axis=0)
    
    if not conv_min:
        conv_min = range_min-0.1*(range_max-range_min)
    if not conv_max:
        conv_max = range_max+0.1*(range_max-range_min)

    x_conv_list = [[[point_to_conv(x[t, n, xidx], conv_std, conv_size, conv_min, conv_max) for xidx in range(2)] for n in range(N)] for t in range(numW)]
    x_conv = np.array(x_conv_list, dtype=np.float32)

    return x, x_conv, y, t_batch