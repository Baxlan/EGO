
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt
import math

def foo(angles):
    coords = bo.spherical_to_cartesian(angles)
    dist = []

    vertex = (len(angles)+1)
    for a in range(vertex):
        category = np.zeros(vertex)
        category[a] = 1
        dist.append(math.dist(coords, category))

    if True:
        category = np.zeros(vertex)
        category[dist.index(min(dist))] = 1
        return np.array(category)
    else:
        return dist

n = 10_000
angles = 30
categories = np.ndarray(shape=(n, angles+1))
#np.random.seed(14)

for i in range(n):
    ang = []
    for j in range(angles):
        ang.append(np.random.uniform(0, 1))
    categories[i] = foo(ang)

sampling = 100
means = [[np.mean(categories[j*sampling:(j+1)*sampling,i]) for j in range(int(n/sampling))] for i in range(categories.shape[1])]

mean = [round(np.mean(means[i]), 3) for i in range(categories.shape[1])]
std = [round(np.std(means[i])/math.sqrt(n/sampling), 3) for i in range(categories.shape[1])]
print(mean)
print(std)