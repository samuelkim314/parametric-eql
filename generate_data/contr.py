import numpy as np
from inspect import signature


def generate_data(func, N, range_min, range_max):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)     # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * np.random.random([N, x_dim]) + range_min
    y = np.random.normal([[func(*x_i)] for x_i in x], 0)
    return x, y


def gen_lin_func_data(N, numW, tfunc1, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x: tfunc1(t)*x, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_sq_func_data(N, numW, tfunc1, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x^2 at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x: tfunc1(t)*x**2, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_prod_func_data(N, numW, tfunc1, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x_1*x_2 at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x1, x2: tfunc1(t)*x1*x2, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_sin_func_data(N, numW, tfunc1, tfunc2, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*sin(f2(t)*x) at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x: tfunc1(t)*np.sin(tfunc2(t)*x), N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_exp_func_data(N, numW, tfunc1, tfunc2, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*exp(f2(t)*x) at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x: tfunc1(t)*np.exp(tfunc2(t)*x), N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_f1_func_data(N, numW, tfunc1, tfunc2, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x^2+f2(t)*x at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]

    data = [generate_data(lambda x: tfunc1(t)*x**2+tfunc2(t)*x, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_sum3_func_data(N, numW, tfunc1, tfunc2, tfunc3, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x_1+f2(t)*x_2+f3(t)*x_3 at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x1, x2, x3: tfunc1(t)*x1+tfunc2(t)*x2+tfunc3(t)*x3, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_f2_func_data(N, numW, tfunc1, tfunc2, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*x_1+f2(t)*x_2*x_3 at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x1, x2, x3: tfunc1(t)*x1+tfunc2(t)*x2*x3, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_spring_force_func_data(N, numW, tfunc1, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*(x_1-x_2) at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x1, x2: tfunc1(t)*(x1-x2), N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch


def gen_spring_potential_func_data(N, numW, tfunc1, range_min, range_max, time_min, time_max):
    """Generates functions of the form f1(t)*(x_1-x_2)^2 at various times t"""
    tdiff = (time_max-time_min) / (numW//2-1)
    times = [-time_max+i *
             tdiff for i in range(numW//2)] + [time_min+i*tdiff for i in range(numW//2)]
    tcoeff1 = [tfunc1(t) for t in times]

    data = [generate_data(lambda x1, x2: tfunc1(t)*(x1-x2)**2, N,
                      range_min, range_max) for t in times]

    x = np.stack([d[0] for d in data], axis=0)
    y = np.stack([d[1] for d in data], axis=0)

    t_batch = np.reshape(np.array(times, dtype=np.float32), [-1, 1])

    return x, y, t_batch