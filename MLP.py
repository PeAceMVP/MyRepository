#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[ ]:


class MLP_NN:
    def _init_(self, feat_num, hide_layer_size=list([]) , act='relu', output_unit='softmax', loss='MSE', norm_initialize=True, label_class=5):
        self.sizes = hide_layer_size
        self.output_unit = output_unit
        self.weights = list([])
        self.biases = list([])
        for i in range(len(self.sizes)):
            if norm_initialize:
                np.random.seed(i)
                if i == 0:
                    self.weights.append(np.random.normal(0, 1, (feat_num, self.sizes[i])))
                else:
                    self.weights.append(np.random.normal(0, 1, (self.sizes[i-1], self.sizes[i])))
                self.biases.append(np.random.normal(0, 1, (1, self.sizes[i])))

            else:
                if i == 0:
                    self.weights.append(np.random.randn(feat_num, self.sizes[i]))
                else:
                    self.weights.append(np.random.randn(self.sizes[i-1], self.sizes[i]))
                self.biases.append(np.random.randn(1,self.sizes[i]))

        # 输出单元层
        if output_unit == 'liner':
            if norm_initialize:
                np.random.seed(len(self.sizes)),
                self.weights.append(np.random.normal(0, 1, (self.sizes[i], 1)))
            else:
                self.weights.append(np.random.randn(self.sizes[i], 1))

            if norm_initialize:
                np.random.seed(len(self.sizes)),
                self.biases.append(np.random.normal(0, 1, (1, 1)))
            else:
                self.biases.append(np.random.randn(1, 1))

        elif output_unit == 'sigmod':
            if norm_initialize:
                np.random.seed(len(self.sizes))
                self.weights.append(np.random.normal(0, 1, (self.sizes[i], 1)))
            else:
                self.weights.append(np.random.randn(self.sizes[i], 1))
            if norm_initialize:
                np.random.seed(len(self.sizes))
                self.biases.append(np.random.normal(0, 1, (1, 1)))
            else:
                self.biases.append(np.random.randn(1, 1))

        elif output_unit == 'softmax':
            if norm_initialize:
                np.random.seed(len(self.sizes))
                self.weights.append(np.random.normal(0, 1, (self.sizes[len(self.sizes)-1], label_class))),
            else:
                self.weights.append(np.random.randn(self.sizes[len(self.sizes)-1], label_class)),
            if norm_initialize:
                np.random.seed(len(self.sizes))
                self.biases.append(np.random.normal(0, 1, (1, label_class)))
            else:
                self.biases.append(np.random.randn(1, self.label_class))

        self.act = act
        self.loss = loss
        self.out = list([])
        # Adam算法
        self.V_dw = [np.zeros(w.shape) for w in self.weights]
        self.V_db = [np.zeros(b.shape) for b in self.biases]
        self.V_beta = 0.9
        self.S_dw = [np.zeros(w.shape) for w in self.weights]
        self.S_db = [np.zeros(b.shape) for b in self.biases]
        self.S_beta = 0.99

    def __activation__(self, z, act='relu'):
        if act == 'relu':
            return np.maximum(z, 0)
        elif self.act == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif act == 'tanh':
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif act == 'linear':
            return z
        else:
            raise ValueError("支持激活函数——relu, sigmoid, tanh, linear")

    def __activation_der__(self, g, act='relu'):
        if act == 'sigmoid':
            return g * (1 - g)
        elif act == 'relu':
            g = np.maximum(g, 0)
            g[g > 0] = 1
            return g
        elif act == 'tanh':
            return 1 - self.__activation__(g, act='tanh') ** 2
        elif act == 'linear':
            return g
        else:
            raise ValueError("支持激活函数求导——relu, sigmoid, tanh, linear")

    def __loss__(self, pred, label, loss='MSE'):
        if loss == 'MSE':
            loss_func = 0.5 * np.sum((pred - label) ** 2)
            return loss_func
        elif loss == 'CrossEntropy_loss_BS':
            loss_func = -np.sum((label * np.log(pred) + (1 - label) * np.log(1 - pred)))
            return loss_func
        elif loss == 'CrossEntropy_loss':
            loss_func = -np.sum( np.log(pred[:, label]))
            return loss_func
        else:
            raise ValueError("支持损失函数——MSE, CrossEntropy_loss_BS（二分类交叉熵）, CrossEntropy_loss")

    def __loss_der__(self, pred, label, loss='MSE'):
        if loss == 'MSE':
            return pred - label
        elif loss == 'CrossEntropy_loss_BS':
            return pred * (1 - label) - (label * (1 - pred))
        elif loss == 'CrossEntropy_loss':
            return pred-label
        else:
            raise ValueError("支持损失函数求导——MSE, CrossEntropy_loss_BS（二分类交叉熵）, CrossEntropy_loss")

    def __output_unit__(self, x, output_unit='liner'):
        if output_unit == 'liner':
            w = self.weights[len(self.weights)-1]
            b = self.biases[len(self.biases)-1]
            z = np.dot(x, w) + b
            y = z
            return y
        elif output_unit == 'sigmod':
            w = self.weights[len(self.weights)-1]
            b = self.biases[len(self.biases)-1]
            z = np.dot(x, w) + b
            y = 1.0 / (1.0 + np.exp(-z))
            return y
        elif output_unit == 'softmax':
            w = self.weights[len(self.weights)-1]
            b = self.biases[len(self.biases)-1]
            z = np.dot(x, w) + b
            exp_x = np.exp(z)
            sum_exp_x = np.sum(exp_x)
            y = exp_x / sum_exp_x
            return y
        else:
            raise ValueError("支持输出单元——liner, sigmod, softmax")

    def __forward__(self, x, train=True):
        self.out.clear()
        for i in range(len(self.sizes)):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(x, w)+b
            if i == len(self.sizes) - 1:
                x = z
            else:
                x = self.__activation__(z, act=self.act)
            if train:
                self.out.append(x)
        return self.__output_unit__(x, output_unit=self.output_unit)

    def __backward__(self, x, y, pred, learning_rate=0.01):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        dx = [np.zeros(x.shape) for x in self.out]

        for i in range(len(dw)):
            if i == 0:
                dz = self.__loss_der__(pred, y, loss=self.loss)
                dw[len(dw) - 1] = np.dot(self.out[len(self.out) - 1].T, dz)
                db[len(db) - 1] = np.sum(dz, axis=0)
                dx[len(dx) - 1] = np.dot(dz, self.weights[len(self.weights) - 1].T)
            elif i == len(dw) - 1:
                dz = self.__activation_der__(dx[0], act=self.act)
                dw[0] = np.dot(x.T, dz)
                db[0] = np.sum(dz, axis=0)
            else:
                dz = self.__activation_der__(dx[len(dx) - i], act=self.act)
                dw[len(dw) - 1 - i] = np.dot(self.out[len(self.out) - 1 - i].T, dz)
                db[len(db) - 1 - i] = np.sum(dz, axis=0)
                dx[len(dx) - 1 - i] = np.dot(dz, self.weights[len(self.weights) - 1].T)

        for i in range(len(self.weights)):
            self.V_dw[i] = self.V_beta * self.V_dw[i] + (1 - self.V_beta) * dw[i]
            self.V_db[i] = self.V_beta * self.V_db[i] + (1 - self.V_beta) * db[i]
            self.S_dw[i] = self.V_beta * self.V_dw[i] + (1 - self.V_beta) * (dw[i] ** 2)
            self.S_db[i] = self.V_beta * self.V_db[i] + (1 - self.V_beta) * (db[i] ** 2)

            self.weights[i] -= learning_rate * (self.V_dw[i] / np.sqrt(self.S_dw[i]))
            self.biases[i] -= learning_rate * (self.V_db[i] / np.sqrt(self.S_db[i]))
    
    def __train__(self, x, y):
        tag=True
        oldloss = 0.0
        newloss = 0.0

        while tag:
            oldpred = self.__forward__(x)
            oldloss = self.__loss__(oldpred, y, loss=self.loss)
            self.__backward__(x, y, oldpred)
            newpred = self.__forward__(x)
            newloss = self.__loss__(newpred, y, loss=self.loss)
            if oldloss<=newloss:
                tag = False
            
    def __pred__(self, x):
        return self.__forward__(x, train=False)

    def __count_loss__(self, pred, label):
        return self.__loss__(pred, label, loss=self.loss)

