#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 2/1/2019

# 实现隐马尔可夫链模型

import numpy as np
from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster

class BassHMM():
    """
    基类HMM
    n_state : 隐藏状态的数目
    n_iter : 迭代次数
    x_size : 观测值维度
    start_prob : 初始概率
    transmat_prob : 状态转换概率    
    """
    __metaclass__ = ABCMeta

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = np.ones(n_state)*(1.0/n_state)
        self.transmat_prob = np.ones((n_state,n_state))*(1.0/n_state)
        self.trained = False
        self.n_iter = iter

        # initialize emit parameters
        @abstractmethod
        def __init(self, X): 
            pass
        
        # return emit probility
        # x在状态k下的发射概率 P(X|Z)
        def emit_prob(self, x):
            return np.array([0])
        
        #根据隐状态生成观测值x p(x|z)
        @abstractmethod
        def generate_x(self,z):
            return np.array([0])


        #update emit function
        def emit_prob_updated(self, X, post_state):
            pass

        #通过HMM生成序列
        def generate_seq(self, seq_length):
            X = np.zeros((seq_length, self.x_size))
            Z = np.zeros(seq_length)
            Z_pre = np.random.choice(self,n_state,1,p=self.start_prob)  #初始状态的采样，随机取隐藏状态的一个值
            X[0] = self.generate_x(Z_pre)  #采样得到序列的第一个值
            Z[0] = Z_pre

            for i in range(seq_length):
                if i==0: continue
                
                #P(Z_n+1) = P(Z_n+1|Z_n)P(Z_n)
                Z_next = np.random.choice(self.n_state,1,p=self.transmat_prob[Z_pre,:][0])
                Z_pre = Z_next

                # P(Xn+1|Zn+1)
                X[i] = self.generate_x(Z_pre)
                Z[i] = Z_pre
            
            return X,Z

            #估计序列X出现的概率
            def X_prob(self,x,Z_seq = np.array([])):
                # 状态序列预处理
                #判断是否已知隐藏状态
                X_length = len(X)
                if Z.seq.any():
                    Z = np.zeros((X_length, self.n_state))
                    for i in range(X_length):
                        Z[i][int(Z_seq[i])] = 1
                else:
                    Z = np.ones((X_length, self.n_state))
                    
                # 向前向后传递因子
                _, c = self.forward(X, Z)  # P(x,z)
                # 序列的出现概率估计
                prob_X = np.sum(np.log(c))  # P(X)
                return prob_X