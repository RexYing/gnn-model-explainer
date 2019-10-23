""" math_utils.py

    Math utilities.
"""

import torch

def exp_moving_avg(x, decay=0.9):
    '''Exponentially decaying moving average.
    '''
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

def tv_norm(input, tv_beta):
    '''Total variation norm
    '''
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad