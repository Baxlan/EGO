
import math
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "lin", [0, 1]]]

constraints = [
    ["title", [[">", 0.6, "<", 0.7], [">", 1]]]
]


def func(x):
    return x**2 * math.sin(5 * math.pi * x)**6

def func2(x):
    return 1/(x+1)


if __name__ == '__main__':
    # evaluated points
    X_in = np.arange(0, 1.1, 0.1)
    y = [func(x) for x in X_in]
    y_constr = [func2(x) for x in X_in]
    X_in = X_in.reshape(-1, 1)

    X_out = np.arange(0, 1, 0.001).reshape(-1, 1)

    X_test = np.arange(0, 1, 0.005)
    y_test = [func(x) for x in X_test]
    y_test_constr = [func2(x) for x in X_test]


    diffs = bo.make_diff_list(X_in)
    metric, lml = bo.optimal_metric(diffs, X_in, y, noise=0, bounds=[-12, 12], seed=32, threads=6)
    metric_constr, lml = bo.optimal_metric(diffs, X_in, y_constr, noise=0, bounds=[-12, 12], seed=32, threads=6)

    print(metric_constr, flush=True)
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})

    K = bo.make_kernel(diffs=diffs, noise=0, metric=metric)
    K_constr = bo.make_kernel(diffs=diffs, noise=0, metric=metric_constr)
    print(K_constr, flush=True)


    model = (K, y, metric)
    pred, sigma = bo.predict(model, X_in, X_out)
    model_constr = (K_constr, y_constr, metric_constr)
    pred_constr, sigma_constr = bo.predict(model_constr, X_in, X_out)

    X_out = X_out.reshape(1, -1)[0]

    a = 5

    fig, ax1 = plt.subplots()
    ax1.plot(X_test, y_test, label="true function", c="black")
    ax1.scatter(X_in, y, label="first evaluations", c="blue")
    ax1.plot(X_out, pred, label="surrogate", c="orange")
    ax1.fill(np.hstack([X_out, X_out[::-1]]), np.hstack([pred - a * sigma, (pred + a * sigma)[::-1]]), alpha = 0.5, fc = "b")

    ax1.plot(X_test, y_test_constr, c="black")
    ax1.scatter(X_in, y_constr, c="blue")
    ax1.plot(X_out, pred_constr, c="orange")
    ax1.fill(np.hstack([X_out, X_out[::-1]]), np.hstack([pred_constr - a * sigma_constr, (pred_constr + a * sigma_constr)[::-1]]), alpha = 0.5, fc = "b")

    ei = bo.expected_improvement(pred, sigma, max(y), a=a, epsilon=1e-13)

    for i in range(len(X_out)):
        if not bo.are_contraint_satifcation_probable(pred_constr[i], sigma_constr[i], a, constraints[0]):
            ei[i] = 0



    ax2 = ax1.twinx()
    ax2.plot(X_out, ei, label="expected improvement", c="cyan")

    next_pt = bo.next_points([model, model_constr], X_in, data_info, constraints=constraints, n=10, seed=32, a=a, threads=6)
    X2_in = list(next_pt.values())[0][0]
    y2 = func(X2_in)
    ax1.scatter(X2_in, y2, label="next point", c="red", s=60)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()
    plt.show()