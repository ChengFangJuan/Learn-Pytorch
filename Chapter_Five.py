import torch
import torch.nn
import torch.optim
from torch.autograd import Variable

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






if __name__ == "__main__":
    # one_line()
    # two()
    linear()