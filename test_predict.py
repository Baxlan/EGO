
import math
import EfficientGlobalOptimization as ego
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x**2 * math.sin(5 * math.pi * x)**6

if __name__ == '__main__':
    # evaluated points
    X_in = np.arange(0, 5, 1/8)
    y = [func(x) for x in X_in]
    X_in = X_in.reshape(-1, 1)

    X_out = np.arange(0, 5, 0.001).reshape(-1, 1)

    X_test = np.arange(0, 5, 0.001)
    y_test = [func(x) for x in X_test]



    model = ego.EGO(noise=0)
    mo = ego.metric_optimizer(noise=0)
    metric = mo.optimal_metric(X_in, y)
    print(metric)

    #pred, sigma = model.predict(X_in, y, X_out, [1/0.06**2])
    pred, sigma = model.predict(X_in, y, X_out, metric=metric)

    X_out = X_out.reshape(1, -1)[0]

    fig, ax1 = plt.subplots()
    ax1.plot(X_test, y_test, label="true function", c="black")
    ax1.scatter(X_in, y, label="first evaluations", c="blue")
    ax1.plot(X_out, pred, label="surrogate", c="orange")
    ax1.fill(np.hstack([X_out, X_out[::-1]]), np.hstack([pred - 1.9600 * sigma, (pred + 1.9600 * sigma)[::-1]]), alpha = 0.5, fc = "b")

    ei = model.expected_improvement(pred, sigma, max(y))


    ax2 = ax1.twinx()
    ax2.plot(X_out, ei, label="expected improvement", c="cyan")

    X2_in = X_out[np.where(ei == max(ei))][0]
    y2 = func(X2_in)

    ax1.scatter(X2_in, y2, label="second evaluation", c="red", s=60)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()
    plt.show()