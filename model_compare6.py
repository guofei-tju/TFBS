#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/5/29 9:34

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn import svm
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import lightgbm



#读取特征文件,使用pandas读取特征文件
def read_feature(filename):
    feature_list=pd.read_excel(io=filename,header=None)
    return feature_list

#对数据集进行切分操作,进行K折的切分
def dataSet_split(X,Y,k):
    train_list=[]
    test_list=[]
    tagtest_list=[]
    tagtrain_list=[]
    skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=1)
    for train,test in skf.split(X,Y):#K折，就会出现K个训练集和测试集的组合
        train_list.append(np.array(X)[train])
        test_list.append(np.array(X)[test])
        tagtrain_list.append(np.array(Y)[train])
        tagtest_list.append(np.array(Y)[test])
    return np.array(train_list),np.array(test_list),np.array(tagtrain_list),np.array(tagtest_list)


if __name__=='__main__':
    #featurename='features/feature/Feature.xlsx'
    #featurename='features/feature/feature_sel/feature_after_XGB.xlsx'#需要read123维特征
    #featurename = 'features/feature/feature_sel/feature_after_ANOVA.xlsx'#需要read140维特征
    featurename = 'features/feature/feature_sel/feature_after_Chi2.xlsx'#需要read93维特征,看起来93维比较好
    #featurename = 'features/feature/feature_sel/feature_after_Lasso.xlsx'
    tagname='features/feature/Lable.xlsx'
    X=read_feature(featurename)
    X=np.array(X)
    X_temp=np.zeros((X.shape[0],93))
    X_temp[:,:]=X[:,0:93]
    y=read_feature(tagname)
    y=pd.Series(y[0].values)# (7168, 1) to (7168, )
    #X_train,X_test,y_train,y_test=dataSet_split(X_temp,y,10)
    X_train, X_test, y_train, y_test = dataSet_split(X_temp, y, 10)
    #y_train,y_test=dataSet_split(y,y,10)
    #input_f=open('knn2.txt','a')
    #两个最优特征组合的最优参数
    sum=0
    for x1,y1,x2,y2 in zip(X_train,y_train,X_test,y_test):
        clf = XGBClassifier(learning_rate=0.15,max_depth=4)#0.15,4
        #clf = lightgbm.LGBMClassifier(learning_rate=0.1,num_leaves=100,n_estimators=80,max_depth=5)
        clf.fit(x1,y1)
        predict=clf.predict(x2)
        data={'y_true':predict,'y_pre':y2}
        df=pd.DataFrame(data)
        #df.to_csv('features/result/predict_best_lightGBM2.csv',mode='a',header=None,index=None)
        df.to_csv('features/result/predict_best_XGB2.csv',mode='a',header=None,index=None)
        acc = clf.score(x2,y2)
        sum+=acc
        print(acc)
    print('------------------------------------------------')
    print(sum/10)
            #print(acc)
        #cr_matrix = classification_report(y2, model.predict(x2),digits=3,)
            #print(cr_matrix)
            #input_f.write(cr_matrix)
            #input_f.write('\n')
            #print(cr_matrix)
        #input_f.close()














