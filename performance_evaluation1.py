#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/6/27 16:19
from sklearn import metrics
import pandas as pd
import numpy as np
import os

def read_tag(filename):
    fn=pd.read_csv(filename,header=None)
    return fn

def filename(string):
    for root,dirs,files in os.walk(string):
        return files

if __name__=='__main__':
    #df=read_tag('features/result/predict_best_lightGBM2.csv')
    #df = read_tag('features/result/predict_best_XGB3.csv')
    #df = read_tag('features/result/predict_best_lightGBM3.csv')
    namelist=filename('features/result1')
    for name in namelist:
        print(name)
        df=read_tag('features/result1/'+name)
        y_pre = df.iloc[:, 0]
        y_true=df.iloc[:,1]#type pandas.Series
        class_TF=[]#记录各个类别正确和错误的个数,[0类正确,0类错误,1类正确,1类错误......]

        for i in range(16):
            sumT=0.0
            sumF=0.0
            for index,val in enumerate(y_true):
                if i==val:#判断是否是同一类别
                    if y_pre[index]==val:
                        sumT+=1
                    else:
                        sumF+=1
            class_TF.append(sumT)
            class_TF.append(sumF)
        '''TP=0.0
        FP=0.0
        for t in class_TF[::2]:
            TP+=t
        for f in class_TF[1::2]:
            FP+=f
        all_sample=TP+FP'''
        class_TFN=[]
        for j in range(16):
            sumTN=0.0
            sumFN=0.0
            for ind,value in enumerate(y_true):
                if j!=value:#找到负类
                    if y_pre[ind]!=j:#TN
                        sumTN+=1
                    elif y_pre[ind]==j:#FP
                        sumFN+=1
            class_TFN.append(sumTN)
            class_TFN.append(sumFN)
        '''TN=0.0
        FN=0.0
        for t in class_TFN[::2]:
            TN+=t
        for f in class_TFN[1::2]:
            FN+=f'''
        #两个class列表中 存放[TP,FP]  [TN,FN]
        TP=[]
        FP=[]
        TN=[]
        FN=[]
        for val1 in class_TF[::2]:
            TP.append(val1)
        for val2 in class_TF[1::2]:
            FN.append(val2)
        for val3 in class_TFN[::2]:
            TN.append(val3)
        for val4 in class_TFN[1::2]:
            FP.append(val4)
        Sn=0.0
        for num1,num2 in zip(TP,FN):
            Sn+=num1/(num1+num2)
        Sn=Sn/16
        print('Sn:',Sn)
        Sp=0.0
        for num3,num4 in zip(TN,FP):
            Sp+=num3/(num3+num4)
        Sp=Sp/16
        print('Sp:',Sp)
        Acc=0.0
        sum=0.0
        for a in TP:
            sum+=a
        Acc=sum/7168
        print('Acc:',Acc)
        Mcc=0.0
        for e,f,g,h in zip(TP,TN,FP,FN):
            sum=(e+h)*(e+g)*(f+h)*(f+g)
            sum=sum**0.5
            Mcc+=((e*f)-(g*h))/sum
        Mcc=Mcc/16
        print("Mcc:",Mcc)
        f1=0.0
        for num1,num2,num3 in zip(TP,FP,FN):
            sum2=2*num1/(2*num1+num2+num3)
            f1=f1+sum2
        f1=f1/16
        print("f1:",f1)
        print('----------------------------------')

        #print(metrics.classification_report(y_true,y_pre,digits=4))
        #print(metrics.confusion_matrix(y_true,y_pre))







        #precise=metrics.precision_score(y_true,y_pre)







