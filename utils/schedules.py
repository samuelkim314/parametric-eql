import numpy as np


def get_lr(base_lr, cur_epoch, total_epochs):
    ratio = cur_epoch / total_epochs
    if ratio <= 5/6:
        return (5 - 9.9 * np.abs(6/5 * ratio - 1/2)) * base_lr
    else:
        return 1e-4 * base_lr


def get_rw(base_rw, cur_epoch, total_epochs):
    ratio = cur_epoch / total_epochs
    if ratio <= 0.95 * 5/6:
        return (2 * 6/5 * ratio)**2 * base_rw
    elif ratio <= 5/6:
        return ((1.9**2 - 1) * 20 * (1 - 6/5 * ratio) + 1) * base_rw
    else:
        return base_rw
    # if ratio <= 5/6:
    #     return base_rw
    # else:
    #     return 0