# -*- coding:utf-8 -*-
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam

net = Sequential(Linear(3,8),
                 ReLU(),
                 Linear(8,8),
                 ReLU(),
                 Linear(8,1))

def g(x,y):

    x0,x1,x2 = x[:,0] ** 0, x[:,1] ** 1, x[:,2] ** 2
    y0 = y[:,0]
    return (x0+x1+x2) * y0 - y0*y0 - x1*x2*x1

optimizer = Adam(net.parameters())
for step in range(2000):
    optimizer.zero_grad()
    x = torch.randn(1000,3)
    y = net(x)
    outputs = g(x,y)
    loss = - outputs.sum()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print('{0} th iter loss = {1}'.format(step,loss))

x_test = torch.randn(1,3)
print('test inout : {0}'.format(x_test))

y_test = net(x_test)
print('nn output : {0}'.format(y_test))
print('g value : {}'.format(g(x_test,y_test)))

def argmax_g(x):
    x0, x1, x2 = x[:, 0] ** 0, x[:, 1] ** 1, x[:, 2] ** 2
    return 0.5 * (x0+x1+x2)[:,None]

yref_test = argmax_g(x_test)
print(yref_test)
