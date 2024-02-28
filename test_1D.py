
import math
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "lin", [0, 1]]]



def func(x):
    return x**2 * math.sin(5 * math.pi * x)**6



if __name__ == '__main__':
    # evaluated points
    X_in = np.arange(0, 1.1, 0.1)
    y = [func(x) for x in X_in]
    X_in = X_in.reshape(-1, 1)

    X_out = np.arange(0, 1, 0.001).reshape(-1, 1)

    X_test = np.arange(0, 1, 0.005)
    y_test = [func(x) for x in X_test]


    diffs = bo.make_diff_list(X_in, data_info)
    metric, lml = bo.optimal_metric(diffs, X_in, y, noise=0, bounds=[-12, 12], seed=32, threads=6)

    print(metric, flush=True)
    print(lml, flush=True)
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})

    K = bo.make_kernel(diffs=diffs, noise=0, metric=metric)
    print(K, flush=True)



    pred, sigma = bo.predict(K, X_in, y, X_out, metric=metric)

    X_out = X_out.reshape(1, -1)[0]

    fig, ax1 = plt.subplots()
    ax1.plot(X_test, y_test, label="true function", c="black")
    ax1.scatter(X_in, y, label="first evaluations", c="blue")
    ax1.plot(X_out, pred, label="surrogate", c="orange")
    ax1.fill(np.hstack([X_out, X_out[::-1]]), np.hstack([pred - 1.9600 * sigma, (pred + 1.9600 * sigma)[::-1]]), alpha = 0.5, fc = "b")

    a = 5
    ei = bo.expected_improvement(pred, sigma, max(y), a=a, epsilon=1e-13)



    ax2 = ax1.twinx()
    ax2.plot(X_out, ei, label="expected improvement", c="cyan")

    next_pt = bo.next_points(K, X_in, y, data_info, n=10, seed=32, metric=metric, a=a, threads=6)
    X2_in = list(next_pt.values())[0][0]
    y2 = func(X2_in)
    ax1.scatter(X2_in, y2, label="next point", c="red", s=60)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()
    plt.show()