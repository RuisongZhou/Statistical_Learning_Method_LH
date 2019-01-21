#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 1/19/2018

# 实现决策树算法
import cv2
import time
import logging
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,0,cv_img)
    return cv_img

def image_preprocess(trainset):
    features = []

    for img in trainset:
        cv_img = img.astype(np.uint8)
        img_b = binaryzation(cv_img)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features,(-1,784))
    return features

class Tree(object):
    def __init__(self,node_type,Class = None, feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        if self.node_type == 'leaf':
            return self.Class

        tree = self.dict[features[self.feature]]
        return tree.predict(features)


def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap

def train(train_set, train_label, features, epsilon):
    LEAF = 'leaf'
    INTERNAL = 'internal'
    global total_class
    #所有class都相同
    labels = set(train_label)
    if(len(labels) == 1):
        return Tree(LEAF, Class=labels.pop())
    
    #features为空,返回trainset中最多的class
    tu=sorted([(np.sum(train_label==i),i) for i in set(train_label.flat)])
    max_class = tu[-1][1]
    # (max_class,max_len) = max([(i,len(filter(lambda x:x==i,train_label))) for i in range(total_class)],key = lambda x:x[1])

    if len(list(features)) == 0:
        return Tree(LEAF,Class = max_class)

    #计算信息增益
    max_feature = 0
    max_KLIC = 0

    hd = calc_ent(train_label)
    for feature in features:
        A = np.array(train_set[:,feature].flat)
        KLIC = hd - calc_condition_ent(A,train_label) #information divergence

        if KLIC > max_KLIC:
            max_KLIC,max_feature = KLIC,feature
    
    #增益不够大，构建叶节点
    if max_KLIC < epsilon:
        return Tree(LEAF, Class= max_class)

    #增益够大，构建非空子集
    sub_features = filter(lambda x:x!=max_feature,features)
    tree = Tree(INTERNAL,feature=max_feature)

    feature_col = np.array(train_set[:,max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    
    for feature_value in feature_value_list:
        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = train(sub_train_set,sub_train_label,sub_features,epsilon)
        tree.add_tree(feature_value,sub_tree)

    return tree
    

def predict(test_set,tree):

    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)


if __name__ == '__main__':
 
    time_1 = time.time()
    print( 'Start read data')
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 图片二值化
    features = image_preprocess(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33)
    print ('Start training')
    tree = train(train_features,train_labels,[i for i in range(784)],0.1)
    print ('Start predicting')
    test_predict = predict(test_features,tree)
    score = accuracy_score(test_labels,test_predict)

    time_4 = time.time()
    print( "The accruacy socre is ", score)
    print ('cost ',time_4 - time_1,' second','\n')