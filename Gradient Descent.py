#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: 24_nemo
@file: 04_BackPropagation_handType.py
@time: 2022/04/08
@desc:
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, forward(4))

epoch_list = []
loss_list = []

for epoch in np.arange(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data -= 0.01 * w.grad.data
        epoch_list.append(epoch)
        loss_list.append(loss_val.item())  # 往列表里添加的内容，不可是tensor类型，或者item()，或者.detach().numpy()取出来
        w.grad.data.zero_()

    print('process:', epoch, loss_val.item())

print('Predict(after training)', 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
