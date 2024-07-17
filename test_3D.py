
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
        X_in = bo.postprocess_inputs(X_in, data_info)
    else:
        X_in = []
        for i in range(200):
            X_in.append([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
        X_in = np.array(X_in)

    y = np.array([func3(x) for x in X_in]).reshape(-1, 1)
    print(np.hstack([X_in, y]), flush=True)

    # test BayesianOptimiser class
    optimizer = bo.BayesianOptimizer("test", noise=[0], data_info=data_info, constraints=[], iso="aniso", seed=32, threads=7)
    optimizer.add_data(X_in, y)
    next_points = optimizer.next_points(n=10, a=10)
    print(next_points, flush=True)



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

    a = 10
    model = [optimizer.kernel, scaled_y[:, 0], optimizer.metric]
    scaled_y_pred, scaled_y_sigma = bo.predict(model, scaled_X_in, scaled_X_plot)
    ei = bo.expected_improvement(scaled_y_pred, scaled_y_sigma, max(scaled_y_pred), a=a, epsilon=1e-13)
    y_pred, y_sigma = bo.postprocess_output(scaled_y_pred, y[:, 0], scaled_y_sigma)

    fig, ax1 = plt.subplots()
    ax1.plot(X_plot[:, 2], y_pred, label="Surrogate", c="blue")
    ax1.fill(np.hstack([X_plot[:, 2], X_plot[:, 2][::-1]]), np.hstack([y_pred - a * y_sigma, (y_pred + a * y_sigma)[::-1]]), alpha = 0.5, fc = "b")
    ax1.plot(X_plot[:, 2], y_test, label="True function", c="black")

    ax2 = ax1.twinx()
    ax2.plot(X_plot[:, 2], ei, label="Expected improvement", c="red")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.grid()
    plt.show()