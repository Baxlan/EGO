
import math
import BayesianOptim as bo
import numpy as np
import random
import matplotlib.pyplot as plt



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "log", [1, 100]],
    ["var2", "real", "lin", [0.1, 100]],
    ["var3", "real", "lin", [0.1, 100]]]



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

    if True:
        X_in = bo.first_points(data_info, 20, 0)
    else:
        X_in = []
        for i in range(200):
            X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
        X_in = np.array(X_in)

    y = [func3(x) for x in X_in]

    # test BayesianOptimiser class
    optimizer = bo.BayesianOptimizer("test", noise=[0], data_info=data_info, constraints=[], iso="diag", seed=32, threads=7)
    optimizer.add_data(X_in, np.array(y).reshape(-1,1))
    next_points = optimizer.next_points(n=10, a=10)
    print(next_points, flush=True)

    X_plot = []
    for i in range(10000):
        #X_plot.append([i/10, 17.636, 93.36])
        X_plot.append([i/100, X_in[0, 1], X_in[0, 2]])
    X_plot = np.array(X_plot)

    model = [optimizer.kernel, y, optimizer.metric]
    y_pred, y_sigma = bo.predict(model, X_in, X_plot)

    a = 1
    plt.plot(X_plot[:, 0], y_pred)
    plt.fill(np.hstack([X_plot[:, 0], X_plot[:, 0][::-1]]), np.hstack([y_pred - a * y_sigma, (y_pred + a * y_sigma)[::-1]]), alpha = 0.5, fc = "b")

    plt.grid()
    plt.show()