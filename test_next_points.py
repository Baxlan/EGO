
import math
import EfficientGlobalOptimization as ego
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x**2 * math.sin(5 * math.pi * x)**6

if __name__ == '__main__':
    X_in = np.arange(0, 1, 0.08)
    y = [func(x) for x in X_in]
    X_in = X_in.reshape(-1, 1)

    X_out = np.arange(0, 1, 0.001).reshape(-1, 1)

    model = ego.EGO(noise=0, m=10)

    #pred, sigma = model.predict(X_in, y, X_out, [1/0.06**2])

    print(model.next_points(X_in, y, [("lin", 0, 1)], [1/0.06**2]))

    print(ego.first_points([("lin", 0, 1)]))