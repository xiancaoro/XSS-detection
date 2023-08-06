# @xiancaoro
# @rongye

# pip install tldextract
# pip install whois

import os
import sys
import re
import matplotlib
import pandas as pd
import numpy as np
from os.path import splitext
import ipaddress as ip
import tldextract
import whois
import datetime
from urllib.parse import urlparse

normal = pd.read_csv('/home/aistudio/data/data52101/dmzo_nomal.csv')
evil = pd.read_csv('/home/aistudio/data/data52101/xssed.csv')

normal['lable'] = np.where(1,0,0)
evil['lable'] = np.where(1,1,1)

col_names = ["param","lable"]
normal.columns = col_names
evil.columns = col_names

evil = evil[0:10000]

df = pd.concat([normal,evil],join='inner')

# 随机取样
df = df.sample(frac=1).reset_index(drop = True)

df.head()


import nltk
import re
import requests
from urllib.parse import unquote

# 计数script
def countscript(param):
    return param.count("script")

# 计数java
def countjava(param):
    return param.count("java")

# 计数iframe
def countifram(param):
    return param.count("iframe")

# 计数 '<'
def quot_1(param):
    return param.count("<")

# 计数 '>'
def quot_2(param):
    return param.count(">")

# 计数 '''
def quot_3(param):
    return param.count('\'')

# 计数 '"'
def quot_4(param):
    return param.count('\"')

# 计数 '%'
def quot_5(param):
    return param.count("%")

# 计数 '('
def quot_6(param):
    return param.count("(")

# 计数 ')'
def quot_7(param):
    return param.count(")")


featureSet = pd.DataFrame(columns=('script','java','iframe','<','>',' \'',' \"','%','(',')','lable'))
# featureSet.head()

from urllib.parse import urlparse
import tldextract

def getFeatures(param, label):
    result = []
    param = str(param)
    
    # 检查'Script'
    result.append(countscript(param))

    # 检查'java'
    result.append(countjava(param))
    
    # 'ifram'
    result.append(countifram(param))

    # '<'
    result.append(quot_1(param))

    # '>'
    result.append(quot_2(param))

    # '\""
    result.append(quot_3(param))

    # '\''
    result.append(quot_4(param))

    # '%'
    result.append(quot_5(param))

    # '('
    result.append(quot_6(param))

    # ')'
    result.append(quot_7(param))

    result.append(str(label))

    return result


for i in range(len(df)):
    features = getFeatures(df["param"].loc[i], df["lable"].loc[i])    
    featureSet.loc[i] = features  


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl
import sklearn.ensemble as ek
from sklearn import model_selection
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

# featureSet.groupby(featureSet['lable']).size()
x = featureSet.drop('lable',axis=1).values
y = featureSet['lable'].values

# 随机森林
model = ek.RandomForestClassifier(n_estimators=50)

x_train, x_test, y_train, y_test = model_selection.train_test_split( 
                                                                x,  
                                                                y, 
                                                                random_state = 1,
                                                                test_size = 0.3)

model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print ("%s : %s " %("RandomForest",score))                                              