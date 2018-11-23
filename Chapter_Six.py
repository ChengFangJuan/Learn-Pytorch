# -*_ coding:utf-8 -*-

import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import torch.nn
import torch.optim

# 下载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root= './data/minist', train=True, transform=torchvision.transforms.ToTensor,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root= './data/minist', train=False, transform=torchvision.transforms.ToTensor,
                                           download=True)

# 读取数据
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
print(len(train_loader))
print(len(test_loader))

fc = torch.nn.Linear(28*28, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc.parameters())

num_epoch = 5
for epoch in range(num_epoch):
    idx = 0
    for images, labels in train_loader:
        x = images.reshape(-1, 28*28)
        optimizer.zero_grad()
        preds = fc(x)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('epoch : {0}, batch : {1}, loss : {2}'.format(epoch, idx, loss))
        idx += 1