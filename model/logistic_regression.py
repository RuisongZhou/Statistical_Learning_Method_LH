#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 1/21/2019

# 实现逻辑回归算法

import time
import math
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class logisticRegression(object):
    def __init__(self):
        self.learnint_rate = 0.0001
        self.max_iteration = 5000

    def predictOneSimple(self,x):
        wx = math.exp(np.dot(self.w,x)/100.0)  
        #为防止指数爆炸，将内积除以100.0，同下
        
        ans1 = wx / (1 + wx)
        ans0 = 1 / (1 + wx)

        return 1 if(ans1 > ans0) else 0

    def train(self, features, labels):
        self.w = np.zeros(len(features[0]) + 1)  #防止wx相乘为0 

        correct_count = 0
        iter_num = 0

        while iter_num < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = np.concatenate([features[index],[1.0]],axis = 0)
            y = labels[index]

            if y == self.predictOneSimple(x):
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            iter_num+=1
            correct_count = 0
            
            exp_wx = math.exp(np.dot(self.w,x)/100.0)

            self.w -= self.learnint_rate* \
                (-y*x+ (x * exp_wx) / (1 + exp_wx))

    
    def predict(self,features):
        labels = []

        for feature in features:
            x = np.concatenate([feature,[1.0]],axis = 0)
            labels.append(self.predictOneSimple(x))

        return labels

if __name__ == '__main__':
    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]


    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    lr = logisticRegression()
    lr.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n'
)
    print ('Start predicting')
    test_predict = lr.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n'
)
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
   