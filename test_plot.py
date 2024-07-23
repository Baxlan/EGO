
import math
import BayesianOptim as bo
import numpy as np
import random
import matplotlib.pyplot as plt



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "log", [1, 100], 0],
    ["var2", "real", "lin", [0.1, 100], 0],
    ["var3", "real", "lin", [0.1, 100], 0]]


output_info = [
    ["output", "lin"]]


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
        X_in = bo.first_points(data_info, 30, 1)
        X_in = bo.postprocess_inputs(X_in, data_info)
    else:
        X_in = []
        for i in range(200):
            X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
        X_in = np.array(X_in)

    y = np.array([func3(x) for x in X_in]).reshape(-1, 1)

    # test BayesianOptimiser class
    optimizer = bo.BayesianOptimizer("test", noise=[0], data_info=data_info, constraints=[], iso="aniso", seed=32, threads=7)
    optimizer.add_data(X_in, y)
    a = 5


    # PLOT
    X_plot = []
    for i in range(10000):
        #X_plot.append([i/100, 93.777, 18.309])
        X_plot.append([36.073, 8.159, (i+1)/100])
    y_test = [func3(x) for x in X_plot]
    X_plot = np.array(X_plot)

    scaled_X_plot = bo.preprocess_inputs(X_plot, data_info)
    scaled_X_in = bo.preprocess_inputs(X_in, data_info)
    scaled_y = bo.preprocess_outputs(y)

    bo.parallelPlot(X_in, y, data_info, output_info)
    #bo.pairPlot(X_in, y, data_info, output_info)
