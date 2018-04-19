import numpy as np
import random
'''
基于附加动量项和自适应算法改进的BP算法
'''
class BP_network:
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
        # 初始化输出层与隐层的学习率
        self.lr_out = []
        self.lr_h = []

    def creatNN(self,ni,nh,no,lr):
        '''
        创建初始神经网络 将阈值以及权重初始化为随机值
        :param ni: 输入层神经元的数目
        :param nh: 隐层神经元的数目
        :param no: 输出层神经元的数目
        :param lr: 学习率
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
        #初始化阈值
        self.h_t = np.zeros(self.h_num)
        self.out_t = np.zeros(self.out_num)
        for h in range(self.h_num):
            self.h_t[h] = random.random()
        for j in range(self.out_num):
            self.out_t[j] = random.random()

        self.lr_out = np.ones(self.out_num)*lr
        self.lr_h = np.ones(self.h_num)*lr

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

    def backPropagate_dynamic(self,x,y,d_ho_w,d_ih_v,d_out_t,d_h_t,o_grid_l,h_grid_l,alpha):
        '''
        针对单个样例更新神经网络中的各参数(标准bp算法)
        :param x: 数据集中的一个输入样本
        :param y: 输入样本对应的输出向量
        :param d_ho_w: 上一步的输出层和隐层的连接权重的更新项
        :param d_ih_v: 上一步的输入层和隐层的连接权重的更新项
        :param d_out_t: 上一步的输出层阈值的梯度
        :param d_h_t:  上一步的隐层阈值的梯度
        :param o_grid: 上一步梯度项g(t-1)
        :param h_grid:  上一步梯度项e(t-1)
        :return:
            当前调整后的值
        '''
        self.predict(x)
        # 计算输出层神经元的梯度项og
        o_grid = np.zeros(self.out_num)
        for j in range(self.out_num):
            o_grid[j] = sigmoidDerivate(self.out_value[j]) * (y[j] - self.out_value[j])

        # 计算隐层神经元的梯度项hg
        h_grid = np.zeros(self.h_num)
        for h in range(self.h_num):
            for j in range(self.out_num):
                h_grid[h] += self.h_out_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * sigmoidDerivate(self.h_value[h])

        #更新输出层相关参数
        lamda = np.sign(o_grid*o_grid_l)
        o_grid_l = o_grid #更新上一步的梯度项
        for h in range(self.h_num):
            for j in range(self.out_num):
                #更新学习率
                o_grid_l[j] = o_grid[j]
                lr = self.lr_out[j]*(3**lamda[j])
                self.lr_out[j] = 0.5 if lr > 0.5 else (0.005 if lr < 0.005 else lr)
                d_ho_w[h][j] = self.lr_out[j]*o_grid[j]*self.h_value[h]+\
                    alpha*d_ho_w[h][j]
                self.h_out_w[h][j] += d_ho_w[h][j]

        for j in range(self.out_num):
            d_out_t[j] = -(self.lr_out[j]*o_grid[j])+alpha*d_out_t[j]
            self.out_t[j] += d_out_t[j]
        #更新隐层相关参数
        lamda = np.sign(h_grid*h_grid_l)
        h_grid_l = h_grid
        for i in range(self.in_num):
            for h in range(self.h_num):
                #更新学习率
                lr = self.lr_h[h]*(3**lamda[h])
                self.lr_h[h] = 0.5 if lr > 0.5 else (0.005 if lr < 0.005 else lr)
                d_ih_v[i][h] = self.lr_h[h]*h_grid[h]*self.in_value[i]+\
                    alpha*d_ih_v[i][h]
                self.in_h_w[i][h] += d_ih_v[i][h]

        for h in range(self.h_num):
            d_h_t[h] = -(self.lr_h[h]*h_grid[h]) + alpha*d_h_t[h]
            self.h_t[h] += d_h_t[h]

        return d_ho_w,d_ih_v,d_out_t,d_h_t,o_grid_l,h_grid_l

    def trainStandard_dyn(self,data_in,data_out):
        '''
        bp training
        :param data_in: 输入样本集
        :param data_out: 样本类别
        :return:
        '''
        d_ho_w = np.zeros((self.h_num,self.out_num))
        d_ih_v = np.zeros((self.in_num,self.h_num))
        d_out_t = np.zeros(self.out_num)
        d_h_t = np.zeros(self.h_num)
        o_grid_l = np.zeros(self.out_num)
        h_grid_l = np.zeros(self.h_num)

        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            d_ho_w, d_ih_v, d_out_t, d_h_t, o_grid_l, h_grid_l = \
            self.backPropagate_dynamic(x,y,d_ho_w,d_ih_v,d_out_t,d_h_t,o_grid_l,h_grid_l,0.2)

            #error rate for each step
            y_delta = 0.0
            for j in range(self.out_num):
                delta = self.out_value[j] - y[j]
                y_delta += delta ** 2
            e_k.append(y_delta / 2)
        e = sum(e_k) / len(e_k)
        return e, e_k

    def PredLabel(self,x):
        y = []
        for m in range(len(x)):
            self.predict(x[m])
            max_y = self.out_value[0]
            label = 0
            for j in range(1,self.out_num):
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

