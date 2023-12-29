
import math
import EfficientGlobalOptimization as ego
import numpy as np
import random

def func(x):
    a = math.sin(x[0]) + math.cos(x[1])
    b = math.exp(-((x[0] - 2)**2 + (x[1] - 2)**2 + (x[2] - 2)**2))
    c = math.log(1 + x[2])
    return a * b * c

if __name__ == '__main__':
    random.seed(0)
    X_in = []

    for i in range(20):
        X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])

    y = [func(x) for x in X_in]

    metric_maker = ego.metric_optimizer(noise = 0, isotropy="diag", it=2, threads=3, bounds=[1e-3, 1e8])
    metric_maker.optimal_metric(X_in, y, plot=True)

