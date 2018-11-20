import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch.autograd import Variable



def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def plot():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X,Y])

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    fig.show()

def Adam():
    x = Variable(torch.Tensor([0.0, 0.0]), requires_grad=True)
    optimizer = torch.optim.Adam([x,])
    f = himmelblau(x)
    for step in range(3001):
        if step:
            optimizer.zero_grad()
            f.backward()
            optimizer.step()
        f = himmelblau(x)
        if step % 100 == 0:
            print('step {0} : x = {1}, f(x) = {2}'.format(step, x.tolist(), f))

if __name__ == "__main__":
    Adam()
