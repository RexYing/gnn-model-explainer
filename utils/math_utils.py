""" math_utils.py

    Math utilities.
"""

def exp_moving_avg(x, decay=0.9):
    '''Exponentially decaying moving average.
    '''
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a