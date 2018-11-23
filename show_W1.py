#!/usr/bin/env python
#coding:utf-8

import matplotlib.pyplot as plt
import cPickle as pickle

fig = plt.figure()
fig.tight_layout()  # 调整整体空白

model = pickle.load(open('model.v', 'rb'))
row = 15
col = 15
for i in range(min(model['W1'].shape[0], row*col)):
    ax = fig.add_subplot(row, col, i+1)
    ax.imshow(model['W1'][i].reshape(80,80), cmap='gray')
    ax.set_xticks([])   # 去掉坐标轴 
    ax.set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0) # 调整子图间隔
plt.savefig('gray.png', dpi=300) # 1800*1200
#plt.show()
