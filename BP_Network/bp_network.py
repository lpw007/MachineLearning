import numpy as np
import random
'''
定义一个BP神经网络类
保存输入层、输出层、隐层神经元的数据
'''
class bp_network:
    def __init__(self):
        #初始化各层神经元数目
        self.in_num = 0
        self.out_num = 0
        self.h_num = 0

        #初始化各层输出值
        self.in_value = []
        self.out_value = []
        self.h_value = []

        #初始化各层神经元之间的阈值以及权重
        self.in_h_w = [] #输入层到隐层之间的权重
        self.h_out_w = [] #隐层到输出层之间的权重
        self.h_t = [] #初始化隐层阈值
        self.out_t = [] #初始化输出层阈值

    def creatNN(self,ni,nh,no):
        '''
        创建初始神经网络 将阈值以及权重初始化为随机值
        :param ni: 输入层神经元的数目
        :param nh: 隐层神经元的数目
        :param no: 输出层神经元的数目
        :return:
        '''
        self.in_num = ni
        self.h_num = nh
        self.out_num = no
        #给各层输出值赋值
        self.h_value = np.zeros(self.h_num)
        self.in_value = np.zeros(self.in_num)
        self.out_value = np.zeros(self.out_num)

        #初始化权重
        self.in_h_w = np.zeros((self.in_num,self.h_num))
        self.h_out_w = np.zeros((self.h_num,self.out_num))
        for i in range(self.in_num):
            for h in range(self.h_num):
                self.in_h_w[i][h] = random.random()
        for h in range(self.h_num):
            for j in range(self.out_num):
                self.h_out_w[h][j] = random.random()

        self.h_t = np.zeros(self.h_num)
        self.out_t = np.zeros(self.out_num)
        for h in range(self.h_num):
            self.h_t[h] = random.random()
        for j in range(self.out_num):
            self.out_t[j] = random.random()

    def predict(self,inX):
        '''
        返回神经网络对输入向量的输出结果
        :param inX: 输入的待预测的向量
        :return: 预测结果
        '''
        #激活输入层
        for i in range(self.in_num):
            self.in_value[i] = inX[i]
        #激活隐层(获得隐层各神经元的输出值)
        for h in range(self.h_num):
            h_in_value = 0.0
            for i in range(self.in_num):
                h_in_value += self.in_h_w[i][h]*self.in_value[i]
            self.h_value[h] = sigmoid(h_in_value - self.h_t[h])
        #激活输出层神经元(获得输出层各神经元的值)
        for j in range(self.out_num):
            o_in_value = 0.0
            for h in range(self.h_num):
                o_in_value += self.h_out_w[h][j]*self.h_value[h]
            self.out_value[j] = sigmoid(o_in_value - self.out_t[j])

    def backPropagate(self,x,y,eta):
        '''
        针对单个样例更新神经网络中的各参数(标准bp算法)
        :param x: 数据集中的一个输入样本
        :param y: 输入样本对应的输出向量
        :param eta: 梯度下降算法的学习率
        :return:
        '''
        #计算当前样本的输出值
        self.predict(x)

        #计算输出层神经元的梯度项og
        o_grid = np.zeros(self.out_num)
        for j in range(self.out_num):
            o_grid[j] = sigmoidDerivate(self.out_value[j])*(y[j]-self.out_value[j])

        #计算隐层神经元的梯度项hg
        h_grid = np.zeros(self.h_num)
        for h in range(self.h_num):
            for j in range(self.out_num):
                h_grid[h]+=self.h_out_w[h][j]*o_grid[j]
            h_grid[h] = h_grid[h]*sigmoidDerivate(self.h_value[h])

        #更新输出层与隐层权值和阈值(输出层的)
        for h in range(self.h_num):
            for j in range(self.out_num):
                self.h_out_w[h][j]+=eta*o_grid[j]*self.h_value[h]

        for j in range(self.out_num):
            self.out_t[j]-=eta*o_grid[j]

        #更新输入层与隐层权值与隐层的阈值
        for i in range(self.in_num):
            for h in range(self.h_num):
                self.in_h_w[i][h]+=eta*h_grid[h]*self.in_value[i]

        for h in range(self.h_num):
            self.h_t[h] -= eta*h_grid[h]


    def trainStandard(self,dataIn,dataOut,eta = 0.05):
        '''
        计算误差率
        :param dataIn: 样本数据集
        :param dataOut: 数据标签
        :param eta: 学习速率
        :return:
          e-k: 保存每一步的错误率的列表
          e: 累计的误差(即Ek)
        '''
        e_k = []
        m = len(dataIn)
        for k in range(m):
            x = dataIn[k]
            y = dataOut[k]
            self.backPropagate(x,y,eta)

            #计算每一步的误差
            y_delta = 0.0
            for j in range(self.out_num):
                delta = self.out_value[j]-y[j]
                y_delta += delta**2
            e_k.append(y_delta/2)
        e = sum(e_k)/len(e_k)
        return e,e_k

    def predLabel(self,inX):
        y = []
        for m in range(len(inX)):
            self.predict(m)
            if self.out_value[0] > 0.5:
                y.append(1)
            else:
                y.append(0)

        return np.array(y)

    def PredLabel1(self, x):
        y = []

        for m in range(len(x)):
            self.predict(x[m])
            max_y = self.out_value[0]
            label = 0
            for j in range(1, self.out_num):
                if max_y < self.out_value[j]:
                    label = j
                    max_y = self.out_value[j]
            y.append(label)
        return np.array(y)

def sigmoid(inX):
    '''
    对数几率函数
    :param inX: 自变量
    :return: 函数值
    '''
    return 1.0/float(np.exp(-inX)+1)

def sigmoidDerivate(y):
    '''
    返回sigmoid函数的导数：y' = y*(1-y)
    :param y: 函数值
    :return: 函数的导数
    '''
    return y*(1-y)


