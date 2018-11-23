import torch
import torch.nn
import torch.optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import math

def one_line():
    x = torch.Tensor([[1., 1., 1.], [2., 3., 1.],
                      [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
    y = torch.Tensor([-10., 12., 14., 16., 18.])

    wr, _ = torch.gels(y,x)
    print(wr)
    w = wr[:3]
    print(w)

def two():
    x = torch.Tensor([[1., 1., 1.], [2., 3., 1.],
                      [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
    y = torch.Tensor([-10., 12., 14., 16., 18.])

    w = Variable(torch.Tensor(3).fill_(0.0), requires_grad = True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([w,],)

    pred = torch.mv(x, w)
    loss = criterion(pred, y)
    for step in range(3001):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = torch.mv(x,w)
        loss = criterion(pred, y)
        if step % 100 == 0:
            print('step = {0}, loss = {1}, w = {2}'.format(step, loss, w.tolist()))

def linear():
    x = torch.Tensor([[1., 1.], [2., 3.],[3., 5.], [4., 2.], [5., 4.]])
    y = torch.Tensor([-10., 12., 14., 16., 18.]).reshape(-1,1)

    fc = torch.nn.Linear(2,1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fc.parameters())
    weights, bias = fc.parameters()

    pred = fc(x)
    loss = criterion(pred, y)
    for step in range(3001):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = fc(x)
        loss = criterion(pred, y)
        if step % 100 == 0:
            print('step = {0}, loss = {1}, w = {2}'.format(step, loss, weights[0,:].tolist()))

def linear_no_normalization():
    x = torch.Tensor([[1000000, 0.0001],[2000000, 0.0003],[3000000, 0.0005],
                      [4000000, 0.0002],[5000000, 0.0004]])
    y = torch.Tensor([-1000., 1200., 1400., 1600., 1800.]).reshape(-1,1)
    loss_list = []

    fc = torch.nn.Linear(2, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fc.parameters())
    # weights, bias = fc.parameters()
    pred = fc(x)
    loss = criterion(pred, y)
    loss_list.append(loss)
    for step in range(1001):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = fc(x)
        loss = criterion(pred, y)
        loss_list.append(loss)
        if step % 1000 == 0:
            print('step = {0}, loss = {1}'.format(step, loss))
    return loss_list

def linear_normalization():
    x = torch.Tensor([[1000000, 0.0001],[2000000, 0.0003],[3000000, 0.0005],
                      [4000000, 0.0002],[5000000, 0.0004]])
    y = torch.Tensor([-1000., 1200., 1400., 1600., 1800.]).reshape(-1,1)
    loss_list = []
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_norm = (x-x_mean) / x_std
    y_mean = y.mean(dim=0)
    y_std = y.std(dim = 0)
    y_norm = (y - y_mean) / y_std

    fc = torch.nn.Linear(2, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fc.parameters())
    # weights, bias = fc.parameters()
    pred_norm = fc(x_norm)
    loss_norm = criterion(pred_norm, y_norm)
    pred = pred_norm * y_std + y_mean
    loss = criterion(pred, y)
    loss_list.append(loss)
    for step in range(1001):
        if step:
            optimizer.zero_grad()
            loss_norm.backward()
            optimizer.step()
        pred_norm = fc(x_norm)
        loss_norm = criterion(pred_norm, y_norm)
        pred = pred_norm * y_std + y_mean
        loss = criterion(pred, y)
        loss_list.append(loss)
        if step % 1000 == 0:
            print('step = {0}, loss = {1}'.format(step, loss))
    return loss_list

def plot():
    loss_list1 = linear_no_normalization()
    loss_list2 = linear_normalization()
    for i in range(len(loss_list1)):
        loss_list1[i] = loss_list1[i] / math.pow(10, 7)
    for i in range(len(loss_list2)):
        loss_list2[i] = loss_list2[i] / math.pow(10, 0.5)
    index_list1 = list(range(len(loss_list1)))
    index_list2 = list(range(len(loss_list2)))
    plt.plot(index_list1, loss_list1, color='g')
    plt.plot(index_list1, loss_list2, color='r')
    plt.show()

if __name__ == "__main__":
    # one_line()
    # two()
    # linear()
    # linear_no_normalization()
    # linear_normalization()
    plot()