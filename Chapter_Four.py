import torch
import math
from torch.autograd import Variable
import torch.optim

x=Variable(torch.Tensor([math.pi / 3, math.pi / 6]),requires_grad=True)

f = -((x.cos() ** 2).sum()) ** 2

print('value = {0}'.format(f))

# f.backward()

print('grad = {0}'.format(x.grad))

optimizer = torch.optim.SGD([x,], lr= 0.1, momentum= 0)

for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    f = -((x.cos() ** 2).sum()) ** 2
    print('step {0}: x = {1}, f(x) = {2}'.format(step, x.tolist(), f))