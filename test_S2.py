
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt
import math

def foo(theta, phi):
    coords = bo.spherical_to_cartesian([theta, phi])
    dist = []

    cat = ["red", "blue", "green"]

    for a in range(len(coords)):
        category = np.zeros(len(coords))
        category[a] = 1
        dist.append(math.dist(coords, category))

    return cat[dist.index(min(dist))]


angles = np.arange(0, 1, 0.01)

red=[]
blue =[]
green =[]

for i in range(len(angles)):
    for j in range(len(angles)):
        color = foo(angles[i], angles[j])
        if color == "red":
            red.append([angles[i], angles[j]])
        elif color == "blue":
            blue.append([angles[i], angles[j]])
        elif color == "green":
            green.append([angles[i], angles[j]])

len_tot = len(angles)**2
print(len(red)/len_tot, flush=True)
print(len(blue)/len_tot, flush=True)
print(len(green)/len_tot, flush=True)

red = np.array(red)
blue = np.array(blue)
green = np.array(green)


plt.scatter(red[:,0],red[:,1],c="red")
plt.scatter(blue[:,0],blue[:,1],c="blue")
plt.scatter(green[:,0],green[:,1],c="green")

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
