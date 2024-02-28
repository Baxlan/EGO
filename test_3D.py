
import math
import BayesianOptim as bo
import numpy as np
import random



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "lin", [0, 100]],
    ["var2", "real", "lin", [0, 100]],
    ["var3", "real", "lin", [0, 100]]]



def func(x):
    a = math.sin(x[0]) + math.cos(x[1])
    b = math.exp(-((x[0] - 2)**2 + (x[1] - 2)**2 + (x[2] - 2)**2))
    c = math.log(1 + x[2])
    return a * b * c

def func2(x):
    a = math.sin(x[0]) * math.cos(x[1]*0.1)
    b = x[0]**2 * math.sin(5 * math.pi * x[0])**6
    c = math.log(1 + x[2])
    return a * b * c

def func3(x):
    a = math.sin(x[1]*0.1)
    b = x[0]**2 * math.sin(0.1 * x[0])**4
    c = math.log(1 + 1/x[2])
    return a * b * c



if __name__ == '__main__':
    random.seed(0)
    X_in = []

    for i in range(10):
        X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
    X_in = np.array(X_in)

    y = [func3(x) for x in X_in]

    diffs = bo.make_diff_list(X_in)
    metric, lml = bo.optimal_metric(diffs, X_in, y, noise=0, bounds=[-12, 12], seed=32, threads=6)
    K = bo.make_kernel(diffs=diffs, noise=0, metric=metric)
    print(metric, flush=True)
    print(lml, flush=True)
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})
    print(K, flush=True)

    # test BayesianOptimiser class
    optimizer = bo.BayesianOptimizer("test", noise=0, data_info=data_info, seed=32, threads=6)
    optimizer.add_data(X_in, y)
    next_points = optimizer.next_points(n=10, a=10)
    print(next_points, flush=True)