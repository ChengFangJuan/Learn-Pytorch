# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim

#产生样本数据
torch.manual_seed(seed=0)
sample_num = 1000
features = torch.rand(sample_num, 2) * 12 - 6
# print(features)
noises = torch.randn(sample_num)
# print(noises)

def himmelblau(x):
    return (x[:,0] ** 2 + x[:,1] - 11) ** 2 + (x[:,0] + x[:,1] ** 2 - 7) ** 2

hims = himmelblau(features) * 0.01
# print(hims)
labels = hims + noises
#

# 划分训练集，验证集，测试集
train_num, validate_num, test_num = 600, 200, 200
train_mse = (noises[:train_num] ** 2).mean()
validate_mse = (noises[train_num:-test_num] ** 2).mean()
test_mse = (noises[-test_num:] ** 2).mean()
print("train_mse: {0}, validate_mse: {1}, test_mse: {2}".format(train_mse, validate_mse, test_mse))

# 建立网络结构
hidden_features = [6, 2] # 指定隐含层数
layers = [nn.Linear(2, hidden_features[0]), ]
for idx, hidden in enumerate(hidden_features):
    layers.append(nn.Sigmoid())
    next_hidden_feature = hidden_features[idx + 1] \
    if idx + 1 < len(hidden_features) else 1
    layers.append(nn.Linear(hidden,next_hidden_feature))

net = nn.Sequential(*layers)

print(net)

optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()
