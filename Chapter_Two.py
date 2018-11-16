import torch

t2 = torch.Tensor([[0,1,2],[3,4,5]])
t1 = torch.Tensor()
print(t2)
print(t2.reshape(3,2))
print(t2 + 1)

print('----------')
print(t2.size())
print(t2.size()[0])
print(t2.dim())
print(t2.numel())
print(t1.dim())
print(t1.size()[0])

print('-----------')
t1 = torch.empty(2)
t2 = torch.zeros(2,2)
t3 = torch.ones(2,2,2)
t4 = torch.full((2,2,2,2), 3.0)
print(t1)
print(t2)
print(t3)
print(t4)

print('----------')
t5 = torch.ones_like(t2)
print(t5)

print('---------')
t6 = torch.arange(0, 4, step = 1) # 不包含end元素，step表示间距
# 包含end元素，step表示间距
# t7 = torch.range(0, 3, step = 1)
t8 = torch.linspace(0, 3, steps = 4) # 包含end元素，steps表示总共的元素数量
t9 = torch.logspace(0, 3, steps = 4) # 前两个参数表示10的指数，steps表示元素数量
print(t6)
# print(t7)
print(t8)
print(t9)

print('------------')

probs = torch.full((3,4), 0.6)
t10 = torch.bernoulli(probs)
print(t10)

weitght = torch.Tensor([[1,100], [100,1], [1,1]])
t11 = torch.multinomial(weitght,1)
print('-------t11------')
print(t11)

t12 = torch.randperm(5) # 生成0-4的随机排列
print('-------t12-----')
print(t12)

t13 = torch.randint(low = 0, high = 4, size = (3,4)) # 生成服从[low,high]之间的均匀分布数列
print('-------t13------')
print(t13)

t14 = torch.rand(2,3) # 生成标准均匀分布的随机变量[0,1)
print('-------t14----')
print(t14)

t15 = torch.randn(2,3) # 生成标准正态分布的随机变量
print('-------t15-------')
print(t15)

mean = torch.Tensor([0., 1.])
std = torch.Tensor([3., 2.])
t16 = torch.normal(mean, std) # 生成正态分布的随机变量
print('---------t16-------')
print(t16)

print('***********************************************************')

tc = torch.arange(12)
print(tc)
t322 = tc.reshape(3, 2, 2)
print(t322)
t43 = t322.reshape(4,3)
print(t43)

t = torch.Tensor(1,2,3,4)
print(t.size())
a = t.permute(dims = [2,0,1,3])
print(a.size())

t12 = torch.Tensor([[5,-9],])
t21 = t12.transpose(0,1)
print(t21)
t21 = t12.t()
print(t21)

t = torch.arange(24).reshape(2,3,4)
print('-----t------')
print(t)
index = torch.Tensor([1,2]).long()
tt = t.index_select(1, index)
print(tt)

ttt = torch.arange(12)
print('-----ttt-----')
print(ttt[3])
print(ttt[3:6])
s = ttt.reshape(3, 4)
print(s[0,:])

print('-------s------')
print(s)
mask = torch.Tensor([[1,0,0,1],[0,1,1,0],[0,0,1,0]]).byte()
print(s.masked_select(mask))

index = torch.Tensor([2,5,6]).long()
print(s.take(index))

t12 = torch.Tensor([[5,9],]).float()
print('--------------')
print(t12)
t34 = t12.repeat(3,2)
print(t34)

tp = s
tn = -tp
tc0 = torch.cat([tp, tn], 0)
print(tc0)
tc1 = torch.cat([tp,tp],1)
print(tc1)

ts0 = torch.stack([tp, tn], 0) # 张量大小完全一致
print(ts0)
print(ts0.size())
ts1 = torch.stack([tp,tn], 1)
print(ts1)
print(ts1.size())

print('---------- computer ---------')
tl = torch.Tensor([[1., 2., 3.], [4., 5., 6.]])
tr = torch.Tensor([[7., 8., 9], [10., 11., 12]])
print(tl + tr)
print(tl - tr)
print(tl * tr)
print(tl / tr)
print(tl ** tr)
print(tl ** (1 / tr))

print('------------ exp pow sin clamp')
print(tl.exp())
print(tl.pow(2))
print(tl.sin())
print(tl.clamp(0,4))

print('---------- dot --------------') # 向量与向量相乘
x = torch.arange(4)
y = torch.arange(1,5)
print(torch.dot(x,y))

print('-------- mv -------') # 向量与矩阵相乘
x = x.view(2,2)
print(x)
y = torch.arange(2)
print(y)
print(torch.mv(x,y))

print('------ mm --------') # 矩阵与矩阵相乘
x = torch.arange(6).view(2,3)
y = torch.arange(6).view(3,2)
print(torch.mm(x,y))


num_sample = 10000000
sample = torch.rand(num_sample,2)
dist = sample.norm(p = 2, dim = 1)
ratio = (dist < 1).float().mean()
pi = ratio * 4
print(pi)
