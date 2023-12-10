
import math
import EfficientGlobalOptimization as ego
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x[0]**2 * math.sin(5 * math.pi * x[0])**6 + x[1]**2 * math.sin(5 * math.pi * x[1])**6

if __name__ == '__main__':
    X_in = np.arange(0, 1, 1/12).reshape(-1, 1)
    X_in = np.hstack([X_in, X_in])

    y = [func(x) for x in X_in]

    metric_maker = ego.metric_optimizer(noise = 0, isotropy="diag", threads=8, bounds=[1e-3, 1e5])
    metric_maker.optimal_metric(X_in, y, plot=True)

