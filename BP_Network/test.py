import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from bp_network import *
from BPNN_improve import *

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
raw_data = urlopen(url) #download data from url
attr = ['sepal_length','sepal_width','petal_length','petal_width','species']
dataset = pd.read_csv(raw_data,delimiter=',',header=None,names=attr)

#获取输入向量
X = dataset.iloc[:,:4].get_values()
#获取所有label
#label输出值为['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dataset.iloc[:,-1] = dataset.iloc[:,-1].astype('category')
label = dataset.iloc[:,4].values.categories
#将类别标签转化为数字
dataset.iloc[:,4].cat.categories = [0,1,2]
y = dataset.iloc[:,4].get_values()

#对标签进行独热处理
Y = pd.get_dummies(dataset.iloc[:,4]).get_values()
#使用sklearn将数据集划分为测试集与训练集
train_X, test_X, train_y, test_y, train_Y, test_Y = \
    train_test_split(X,y,Y,test_size = 0.5,random_state = 42)
#
# bp_standard = bp_network()
# bp_standard.creatNN(4,6,3)
# ell = []
# for i in range(1000):
#     err,err_k = bp_standard.trainStandard(train_X,train_Y)
#     ell.append(err)
#
# f1 = plt.figure(4)
# plt.xlabel("epochs")
# plt.ylabel("error")
# plt.title("training error convergence curve with fixed learning rate")
# plt.plot(ell)
# plt.show()

#训练好后统计预测数据集上的错误率
# pred = bp_standard.PredLabel1(test_X);
# count = 0
# for i in range(len(test_y)):
#     if pred[i] == test_y[i]: count+=1
# test_err = 1 - count/len(test_y)
# print('test error rate:%.3f'%test_err)

bp_improve = BP_network()
bp_improve.creatNN(4,6,3,0.05)

e = []
for i in range(1000):
    err,err_k = bp_improve.trainStandard_dyn(train_X,train_Y)
    e.append(err)
f1 = plt.figure()
plt.xlabel("epochs")
plt.ylabel("error")
plt.title("training error convergence curve with dyn learning rate")
plt.plot(e)
plt.show()

