
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt
import math

def foo(theta, phi, phi2):
    coords = bo.spherical_to_cartesian([theta, phi, phi2])
    dist = []

    cat = ["red", "blue", "green", "black"]

    for a in range(len(coords)):
        category = np.zeros(len(coords))
        category[a] = 1
        dist.append(math.dist(coords, category))

    return cat[dist.index(min(dist))]


angles = np.arange(0, 1, 0.05)

red=[]
blue =[]
green =[]
black =[]

for i in range(len(angles)):
    for j in range(len(angles)):
        for k in range(len(angles)):
            color = foo(angles[i], angles[j], angles[k])
            if color == "red":
                red.append([angles[i], angles[j], angles[k]])
            elif color == "blue":
                blue.append([angles[i], angles[j], angles[k]])
            elif color == "green":
                green.append([angles[i], angles[j], angles[k]])
            elif color == "black":
                black.append([angles[i], angles[j], angles[k]])

len_tot = len(angles)**3
print(len(red)/len_tot, flush=True)
print(len(blue)/len_tot, flush=True)
print(len(green)/len_tot, flush=True)
print(len(black)/len_tot, flush=True)

red = np.array(red)
blue = np.array(blue)
green = np.array(green)
black = np.array(black)

ax = plt.axes(projection="3d")

ax.scatter(red[:,0],red[:,1], red[:,2],c="red")
ax.scatter(blue[:,0],blue[:,1], blue[:,2],c="blue")
ax.scatter(green[:,0],green[:,1], green[:,2],c="green")
ax.scatter(black[:,0],black[:,1], black[:,2],c="black")

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.show()
