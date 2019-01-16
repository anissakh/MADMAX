import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def read_testing(filename="result.txt"):
    X = list()
    Y = list()
    Z = list()
    with open(filename,"r") as f:
        for line in f.readlines():
            line.split(" ")
            X.append(line[0])
            Y.append(line[1])
            Z.append(line[2])

    return X,Y,Z


X,Y,Z = read_testing()
# data = [ go.Surface(z=X),  go.Surface(z=Y),  go.Surface(z=Z) ]
# py.iplot(data,filename='multiple-surfaces')
# Axes3D.plot_surface(X, Y, Z)




# Plot the surface.
surf = ax.plot_surface(np.array(X), np.array(Y), np.array(Z), linewidth=0, antialiased=False)