
import math
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt



# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "lin", [0, 1]]]



def func(x):
    return 0.1*x**2 * math.sin(5 * math.pi * x)**6



if __name__ == '__main__':
    # evaluated points
    X_in = np.arange(0, 1.1, 0.1)
    y = np.array([func(x) for x in X_in]).reshape(-1, 1)
    X_in = X_in.reshape(-1, 1)

    X_out = np.arange(0, 1, 0.001).reshape(-1, 1)

    X_test = np.arange(0, 1, 0.005)
    y_test = [func(x) for x in X_test]

    scaled_X_in = bo.preprocess_inputs(X_in, data_info)
    scaled_y = bo.preprocess_outputs(y)
    scaled_X_out = bo.preprocess_inputs(X_out, data_info)

    diffs = bo.make_diff_list(scaled_X_in)
    metric, lml = bo.optimal_metric(diffs, scaled_X_in, scaled_y[:, 0], noise=0, bounds=[-12, 12], iso="iso", seed=32, threads=6)

    print(metric, flush=True)
    print(lml, flush=True)
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})

    K = bo.make_kernel(diffs=diffs, noise=0, metric=metric)
    #np.fill_diagonal(K, 1.01)
    print(K, flush=True)


    model = (K, scaled_y[:, 0], metric)
    scaled_pred, scaled_sigma = bo.predict(model, scaled_X_in, scaled_X_out)

    pred, sigma = bo.postprocess_output(scaled_pred, y[:, 0], scaled_sigma)

    X_out = X_out.reshape(1, -1)[0]
    a = 5

    fig, ax1 = plt.subplots()
    ax1.plot(X_test, y_test, label="true function", c="black")
    ax1.scatter(X_in, y, label="first evaluations", c="blue")
    ax1.plot(X_out, pred, label="surrogate", c="orange")
    ax1.fill(np.hstack([X_out, X_out[::-1]]), np.hstack([pred - a * sigma, (pred + a * sigma)[::-1]]), alpha = 0.5, fc = "b")

    ei = bo.expected_improvement(pred, sigma, max(y), a=a, epsilon=1e-13)



    ax2 = ax1.twinx()
    ax2.plot(X_out, ei, label="expected improvement", c="cyan")

    next_pt = bo.next_points([model], scaled_X_in, data_info, constraints=[], n=10, seed=32, a=a, threads=6)
    X2_in = list(next_pt.values())[0][0]
    y2 = func(X2_in)
    ax1.scatter(X2_in, y2, label="next point", c="red", s=60)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()
    plt.show()