
import BayesianOptim as bo
import numpy as np
import matplotlib.pyplot as plt
import math

def foo(theta, phi, phi2):
    coords = bo.spherical_to_cartesian([math.acos(theta), phi*math.pi/2, phi2*math.pi/2])
    dist = []

    cat = ["red", "blue", "green", "black"]

    for a in range(len(coords)):
        category = np.zeros(len(coords))
        category[a] = 1
        dist.append(math.dist(coords, category))

    return cat[dist.index(min(dist))]


angles = np.arange(0, 1, 0.01)

red_x =[]
red_y =[]
blue_x =[]
blue_y =[]
green_x =[]
green_y =[]
black_x =[]
black_y = []

for i in range(len(angles)):
    for j in range(len(angles)):
        for k in range(len(angles)):
            color = foo(angles[i], angles[j], angles[k])
            if color == "red":
                red_x.append(angles[i])
                #red_y.append(angles[j])
            elif color == "blue":
                blue_x.append(angles[i])
                #blue_y.append(angles[j])
            elif color == "green":
                green_x.append(angles[i])
                #green_y.append(angles[j])
            elif color == "black":
                black_x.append(angles[i])
                #black_y.append(angles[j])

len_tot = len(angles)**3
print(len(red_x)/len_tot, flush=True)
print(len(blue_x)/len_tot, flush=True)
print(len(green_x)/len_tot, flush=True)
print(len(black_x)/len_tot, flush=True)

"""
plt.scatter(red_x,red_y,c="red")
plt.scatter(blue_x,blue_y,c="blue")
plt.scatter(green_x,green_y,c="green")
plt.show()
"""