
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
    term1 = math.sin(x[0]) * math.cos(x[1]*0.1)
    term2 = x[0]**2 * math.sin(5 * math.pi * x[0])**6
    term3 = math.log(1 + x[2])
    return term1 * term2 * term3

def func3(x):
    term1 = math.sin(x[1]*0.1)
    term2 = x[0]**2 * math.sin(0.1 * x[0])**4
    term3 = math.log(1 + 1/x[2])
    return term1 * term2 * term3



if __name__ == '__main__':
    random.seed(0)
    X_in = []

    for i in range(10):
        X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
    X_in = np.array(X_in)

    y = [func3(x) for x in X_in]

    metric = bo.max_metric(X_in)
    metric = bo.optimized_metric(X_in, y, noise=0, isotropy="aniso", initial=1e-2, method="gradient")

    K = bo.make_kernel(x=X_in, noise=0, metric=metric)
    print(metric)
    print(bo.log_marginal_likelihood(K, y))

    np.set_printoptions(formatter={'float':"{0:0.3f}".format})
    print(K)
