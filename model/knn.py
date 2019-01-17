#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 1/17/2018

# 实现k-Means算法

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class knn(object):
    def __init__(self):
        self.trainset = None
        self.lables = None
    def get_hog_features(self, trainset):
        features = [];
        hog = cv2.HOGDescriptor("./hog.xml")

        for img in trainset:
            img = np.reshape(img,(28,28))
            cv_img = img.astype(np.uint8)
            hog_feature = hog.compute(cv_img)

            features.append(hog_feature)
        
        features = np.array(features)
        features = np.reshape(features, (-1,324))

        return features

    def train(self, trainset, train_lables):
        self.trainset = trainset
        self.lables = train_lables


    def predect(self, testset, k=10):
        # 采用L2范数进行临近拟合
        predict = []
        for test_vec in testset:
            tmp = np.sqrt(np.square(self.trainset - test_vec))
            tmp = np.sum(tmp, axis=1)
            tmp = tmp.argsort()

            class_count = [0 for i in range(k)]
            for i in range(k):
                label = self.lables[tmp[i]]
                class_count[label] += 1

            
            predict.append(np.argmax(class_count))
            
        return np.array(predict)




if __name__ == '__main__':

    print( 'Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]
    model = knn()
    features = model.get_hog_features(imgs)

    #imgs = imgs[0:1000,:]
    #labels = labels[0:1000,:]
    #print (imgs.shape)
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape
    
    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    model.train(train_features,train_labels)
    time_3 = time.time()    
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = model.predect(test_features)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
            

