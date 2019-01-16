#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 1/15/2018

# 实现二分类感知机模型

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class perceptron(object):
    def __init__(self):
        self.learning_step = 0.001
        self.max_iteration = 5000

    def train(self, features, labels):
        self.w = np.zeros(len(features[0]) + 1)  #防止wx相乘为0 
        correct_number = 0
        i = 0
        
        while i < self.max_iteration:
            index = random.randint(0, len(labels) -1)

            x = np.concatenate([features[index],[1.0]],axis = 0)
            y = 2*labels[index] -1  # y = -1 or 1
            wx = np.dot(self.w.T,x)
            
            if wx* y > 0:
                correct_number +=1
                if correct_number > self.max_iteration:
                    break
                continue
            
            self.w += self.learning_step*y*x
        
        #print(self.w.shape)


    def predict(self,features):
        labels = []
        for feature in features:
            x = np.concatenate([feature,[1.0]],axis = 0)
            result = np.dot(self.w.T,x)
            labels.append(int(result>0))
        return labels

if __name__ == '__main__':
    time_1 = time.time()
    print("begin read data")

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    #print(data.shape)
    imgs = data[:, 1:]
    labels = data[:, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33)
    #print (type(train_features))

    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    model = perceptron()
    model.train(train_features, train_labels)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = model.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)
