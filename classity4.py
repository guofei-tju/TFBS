#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

#读取特征文件,使用pandas读取特征文件
def read_feature(filename):
    feature_list=pd.read_excel(io=filename,header=None)
    return feature_list

#矩阵拼接
def matrix_concate(X1,X2):
    result=np.vstack((X1,X2))
    return result
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
    featurename='features/feature/feature24.xlsx'
    tagname='features/feature/Lable.xlsx'
    X=read_feature(featurename)
    y=read_feature(tagname)
    y=pd.Series(y[0].values)# (7168, 1) to (7168, )
    X_train,X_test,y_train,y_test=dataSet_split(X,y,10)
    #y_train,y_test=dataSet_split(y,y,10)
    #input_f=open('knn2.txt','a')
    #model1 = RandomForestClassifier(random_state=14, criterion='gini', max_depth=15, n_estimators=90)
    #两个最优特征组合的最优参数
    model1 = RandomForestClassifier(random_state=14, criterion='gini', max_depth=15, n_estimators=120)
    model2 = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=5, random_state=10)
    model3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=157, splitter='best')
    model4 = KNeighborsClassifier(algorithm='brute', n_neighbors=2, weights='distance')
    models = [model1,model2, model3, model4]
    model_name = ['RF', 'GBDT','DTree', 'KNN']
    #model_name = ['RF']
    for index in range(len(model_name)):
        sum=0
        for x1,y1,x2,y2 in zip(X_train,y_train,X_test,y_test):
            model=models[index]
            #model = RandomForestClassifier(random_state=14, criterion='gini', max_depth=20, n_estimators=50)
            #model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=157, splitter='best')
            #model = KNeighborsClassifier(algorithm='brute', n_neighbors=2, weights='distance')
            model.fit(x1,y1)
            acc = model.score(x2,y2)
            sum+=acc
            #print(acc)
            cr_matrix = classification_report(y2, model.predict(x2),digits=3,)
            #print(cr_matrix)
            #with open(result_name[index], 'a') as f_o:
                #f_o.write(model_name[index] + '\n')
                #f_o.write('accuracy, ' + str(acc) + '\n')
                #f_o.write('cr_matrix:\n' + cr_matrix + '\n')
        print('model {} acc:'.format(model_name[index]),sum/10)
            #input_f.write(cr_matrix)
            #input_f.write('\n')
            #print(cr_matrix)
        #input_f.close()














