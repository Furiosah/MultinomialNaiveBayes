#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:13:57 2019

@author: zoey
"""
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


data = pd.read_csv('reddit_train.csv',',')
test = pd.read_csv('reddit_test.csv',',')

data = data[~data.comments.str.contains("http")]
data = data[~data.comments.str.contains("spam")]
data = data[~data.comments.str.contains("FREE")]
data = data[~data.comments.str.contains("sales")]
data = data[~data.comments.str.contains(".jpg")]
data = data[~data.comments.str.contains(".gif")]
data = shuffle(data).reset_index(drop=True)

target = ['hockey', 'nba', 'leagueoflegends',
                  'soccer', 'funny', 'movies','anime', 'Overwatch', 'trees', 
                  'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 
                  'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
def readFile(x):
    data = pd.read_csv(x,',')
    X_train = data['comments']
    Y_train = data['subreddits']
    return data, X_train, Y_train
    
def vectorize(X_train):
#this is the method that produce a vectorizer object and a tf-idf sparse matrix
    cv = CountVectorizer(stop_words='english', lowercase=True)
    word_count_vector=cv.fit_transform(X_train)
    tfidf_transformer=TfidfTransformer(norm=None,smooth_idf=True,use_idf=True)
    tfidf_v = tfidf_transformer.fit_transform(word_count_vector)
    
    return cv,word_count_vector,tfidf_v

def cross_val(X_t,k):
    
    kf = KFold(n_splits=k,shuffle=True)
    cacc=0
    for train_index,test_index in kf.split(X_t):
        a=X_t.loc[train_index, :]
        b=X_t.loc[test_index, :]
        data = nb(a.comments,a.subreddits)
        data.fit()
        data.predict(b.comments)
        acc = 0
        count = 0
        for l, t in zip(data.preDF.values,b.subreddits.values):
            if (l==t):
                count = count+1
        acc = acc + (count/(data.preDF.shape[0]))
        print(acc)
        cacc=acc+cacc
    return cacc/k
    
class nb:
    def __init__(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.target = ['hockey', 'nba', 'leagueoflegends',
                  'soccer', 'funny', 'movies','anime', 'Overwatch', 'trees', 
                  'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 
                  'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
        self.cv = vectorize(self.X_train)[0]
        self.wc = vectorize(self.X_train)[1]
        self.tfidf_v =vectorize(self.X_train)[2]
        self.t = np.empty(20,dtype = object)
    def fitOnce(self,y):
        tmptheta = np.zeros(self.wc.shape[1])
        tmpthetaC = 0
        wc = self.tfidf_v
        mat = wc.todense()
        loc = np.argwhere((self.Y_train.values)==y)
        #this is the array that contains index of comments from a specific class y
        denom = len(loc)
        self.cons= math.log(len(loc)/wc.shape[0])
        for i in range(0,mat.shape[1]):
            num = 0
            count=0
            for j in loc:
                if mat[j,i].sum()>0:
                      count =count+1
            num = count
            a = (num+1.0)/(denom+2.0)
            b = 1-((num+1.0)/(denom+2.0))
            tmptheta[i] = math.log(a/b)
            tmpthetaC = tmpthetaC + math.log(b)
        return tmptheta,tmpthetaC
    
    def fit(self):
        self.consT = np.zeros(20)
        l = []
        x = []
        for i in range(20):
            x.append(l)
        self.theta = x
        for i in range(20):
            tupleR = self.fitOnce(target[i])
            self.consT[i] = tupleR[1]
            self.theta[i] = tupleR[0]
        
    def predict(self, X_test):
        test_v = self.cv.transform(X_test)
        self.p = ['' for c in range(0,X_test.shape[0])]
        prob = np.zeros(len(self.target))
        index = 0
        for comment in test_v.todense():
            for i in range(0,20):
                prob[i] =np.dot(self.theta[i],comment.A1) + self.consT[i] +self.cons
            self.p[index]=self.target[np.argmax(prob)]
            index = index+1
        df = pd.DataFrame(self.p)
        self.preDF = df
        self.csv_data = df.to_csv(path_or_buf = './predict.csv')
        return self.csv_data
        

def main():
    acc = cross_val(data,5)
    print(acc)

    
