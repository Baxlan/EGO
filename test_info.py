
import BayesianOptim as bo
import numpy as np


# name, real or discrete, lin or log, bounds
data_info = [
    ["var1", "real", "lin", [0, 1]],
    ["var1", "categorical", "", ["red", "blue", "green"]],
    ["var1", "categorical", "", ["sphere", "cube"]],
    ["var1", "discrete", "lin", [0, 1]],
    ["var1", "real", "lin", [0, 1]]]

data = [[0.41, 0.2, 0.5, 0.44, 0.712, 0.845, 0.214, 0.11]]

print(bo.categorify_and_discretize_data(data, data_info))