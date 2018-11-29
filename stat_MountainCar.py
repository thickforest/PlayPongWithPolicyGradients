#!/usr/bin/python
#coding:utf-8
########################################################################
# File Name: reward_stat.py
# Author: forest
# Mail: thickforest@126.com 
# Created Time: 2018年11月24日 星期六 15时44分15秒
########################################################################
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

scores = []
for line in open('MountainCar.log').readlines()[2:-1]:
    score = line.split()[-1][:-1]
    scores.append(float(score))

total_num = len(scores)
num_per_pile = 1000
piles_num = total_num/num_per_pile
print "Total Num:", total_num
print "Num per pile:", num_per_pile
print "Num of piles:", piles_num
for piles_index in range(piles_num):
    start = num_per_pile * piles_index
    end = start + num_per_pile
    piles = scores[start:end]
    #print "[%3d,%3d,%3d]" % (min(piles), np.median(piles), max(piles)), "%.2f"%np.mean(piles), "%.2f"%np.std(piles), start, end
    print "[%d,%d,%d]" % (min(piles), np.median(piles), max(piles)), "%.2f"%np.mean(piles), "%.2f"%np.std(piles),

    # 符合正态分布
    #   3σ原则:
    #   数值分布在（μ-σ,μ+σ)中的概率为0.6827
    #   数值分布在（μ-2σ,μ+2σ)中的概率为0.9545
    #   数值分布在（μ-3σ,μ+3σ)中的概率为0.9973
    piles = np.array(piles)
    mean_value = np.mean(piles)
    std_value = np.std(piles)
    for i in range(1, 4):
        left = mean_value - std_value * i
        right = mean_value + std_value * i
        bigpiles = piles[np.where((left < piles) & (piles < right))]
        size = bigpiles.shape[0]
        prop = 100*size/num_per_pile
        #print " (%d,%d] %d %d%%" % (int(left), int(right), size, prop),
        print " [%d,%d]" % (int(left)+1, int(right)),
    print

row = 10
col = 3
show_num = row*col
plt.figure(figsize=(6,20))  # 像素：600X2000
for show_index in range(show_num):
    plt.subplot(row, col, show_index+1)
    piles_space = piles_num/(show_num - 1)
    start = num_per_pile * (piles_space * show_index)
    end = start + num_per_pile
    if show_index == show_num - 1:
        start -= num_per_pile
        end -= num_per_pile
    #print start, end
    piles = map(int, scores[start:end])
    plt.hist(piles)
    #plt.hist(piles, bins = range(0, 22))
plt.savefig('MountainCar_hist.png')
#plt.show()
