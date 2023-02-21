#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import loss_function as loss

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return np.maximum(0, 1)

def softmax(z):
    exp_z = np.exp(z)
    # print(exp_z.shape)
    sum_exp_z = np.expand_dims(np.sum(exp_z, axis=1), axis=1) # axis = 0(row), axis = 1(columns)
    # (20,) -> (20,1)
    result = exp_z / sum_exp_z

    return result

class Network():
    def __init__(self, input_size, output_size, hidden_layers):
        
        # Define and initialize Weights and Bias in Dictionary format
        init_coef = 0.1
        self.num_layers = len(hidden_layers) + 1
        
        # 첫 번째 Layer wieght bias 초기화
        self.input_size = input_size
        self.size_of_input = self.input_size
        self.output_size = output_size

        self.W = dict()
        self.b = dict()    

        # self.num_layers = 4
        for i in range(self.num_layers):
            if i == 0:
                self.W.update({f'layer{i + 1}' : init_coef * np.random.randn(input_size, hidden_layers[i])})
                self.b.update({f'layer{i + 1}' : init_coef * np.random.randn(1, hidden_layers[i])})
                # w_li = []
                # w_li.append(self.W)
                # a_w_li = np.array(w_li)
                # print(a_w_li.shape)
                # print('a_w_li.shape')
                
            elif i == len(hidden_layers): # 3
                self.W.update({f'layer{i + 1}' : init_coef * np.random.randn(hidden_layers[i - 1], output_size)})
                self.b.update({f'layer{i + 1}' : init_coef * np.random.randn(1, output_size)})
                # w_li1 = []
                # w_li1.append(self.W)
                # a_w_li1 = np.array(w_li1)
                # print(a_w_li1.shape)
                # print('a_w_li1.shape')
                
            else:
                self.W.update({f'layer{i + 1}' : init_coef * np.random.randn(hidden_layers[i - 1], hidden_layers[i])})
                self.b.update({f'layer{i + 1}' : init_coef * np.random.randn(1, hidden_layers[i])})
                # w_li2 = []
                # w_li2.append(self.W)
                # a_w_li2 = np.array(w_li2)
                # print(a_w_li2.shape)
                # print('a_w_li2.shape')
            
    def forward(self, x): # x shape = batch X input_size : 20 * 400, 차원 생각해보기
        self.Z = dict()
        self.A = dict() # activation function dict initialization , last layer 예외처리 a가 마지막에 필요 없음
        
        #### softmax까지 하고 나서 다 더 했을 때 1에 가까운 숫자가 나오는데 한 숫자가 이상하게 큼 9개 더한 숫자랑 1개랑 비슷함 크기가
        #### 그거 왜 그런지 찾아야 될 거 같은데 --> one_hot에 존재하는 1을 뺐을 때 1이 빼진 값이 크기가 엄청 작아짐
        
        for i in range(1, self.num_layers + 1):  # 0 ~ 3 //  4 = self.num_layers z = 5개, a = 4개
            if i < self.num_layers: # 1,2,3
                x = np.matmul(x, self.W[f'layer{i}']) + self.b[f'layer{i}']
                self.Z.update({f'layer{i}' : x})
                # print(x.shape)
                # print('ifx------')
                a = ReLU(x)
                self.A.update({f'layer{i}' : a})
                # print(a.shape)
                # print('ifa------')
            
            else:
                x = np.matmul(x, self.W[f'layer{i}']) + self.b[f'layer{i}']
                self.Z.update({f'layer{i}' : x})
                # print(x.shape)
                # print('ex------')
                
                a = softmax(x)
                self.A.update({f'layer{i}' : a})
                # print(a)
                # print(a.shape)
                # print('ea------')
                # print('---------------------------------------')

        return a
    
    def backward(self, X, y):
        self.delta = dict() # 국부적 기울기
        self.dw = dict(); # 미분된 w값
        self.db = dict(); # 미분된 b값
    
        for i in reversed(range(1, self.num_layers + 1)):
            batch_size = X.shape[0]

            if i == self.num_layers:
                # print(self.Z[f'layer{i}'])
                # print(softmax(self.Z[f'layer{i}']))
                # print(loss.one_hot(y, self.output_size))
                # print(softmax(self.A[f'layer{i}']) - loss.one_hot(y, self.output_size))
                delta = softmax(self.Z[f'layer{i}']) - loss.one_hot(y, self.output_size) # delta 사용 O
                self.delta.update({f'layer{i}' : delta})
                # print('delta.shape')
                # print(self.delta[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print(self.delta[f'layer{i}']) ##### 다른 숫자들에 비해서 작은 숫자가 하나 존재함 --> 그게 답임
                # print('if_delta')
                # print(delta.shape)
                # print('-------------------')
                dw = np.matmul(self.A[f'layer{i - 1}'].T, delta) / batch_size # dw(변화량)
                self.dw.update({f'layer{i}' : dw})
                # print('dw.shape')
                # print(self.dw[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print(self.dw[f'layer{i}'])
                # print('if_dw')
                # print(dw.shape)
                # print('-------------------')
                # print(self.delta[f'layer{i}'].shape)
                db = np.matmul(np.ones((1, batch_size)), delta) / batch_size
                self.db.update({f'layer{i}' : db})
                # print(self.db[f'layer{i}'].shape)
                # print('db.shape')
                # print(self.db[f'layer{i}'].shape)
                # print('----------------------------------------')
            
            elif i > 1:
                delta = np.matmul(self.delta[f'layer{i + 1}'], self.W[f'layer{i + 1}'].T) * dReLU(self.Z[f'layer{i}'])
                self.delta.update({f'layer{i}' : delta})
                # print('delta.shape')
                # print(self.delta[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print('elif_delta')
                # print(delta.shape)
                # print('-------------------')
                dw = np.matmul(self.A[f'layer{i - 1}'].T, delta) / batch_size
                self.dw.update({f'layer{i}' : dw})
                # print('dw.shape')
                # print(self.dw[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print('dw.shape')
                # print(self.dw[f'layer{i}'].shape)
                # print('elif_dw')
                # print(dw.shape)
                # print('-------------------')
                db = np.matmul(np.ones((1, batch_size)), delta) / batch_size
                self.db.update({f'layer{i}' : db})
                # print('db.shape')
                # print(self.db[f'layer{i}'].shape)
                # print('db.shape')
                # print(self.db[f'layer{i}'].shape)
                # print('----------------------------------------')
                
          
            else:
                delta = np.matmul(self.delta[f'layer{i + 1}'], self.W[f'layer{i + 1}'].T) * dReLU(self.Z[f'layer{i}'])
                self.delta.update({f'layer{i}' : delta})
                # print('delta.shape')
                # print(self.delta[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print('else_delta')
                # print(delta.shape)
                # print('-------------------')
                dw = np.matmul(X.T, delta) / batch_size
                self.dw.update({f'layer{i}' : dw})
                # print('dw.shape')
                # print(self.dw[f'layer{i}'].shape)
                # print('----------------------------------------')
                # print('else_dw')
                # print(dw.shape)
                # print('-------------------')
                db = np.matmul(np.ones((1, batch_size)), delta) / batch_size
                self.db.update({f'layer{i}' : db})
                # print('db.shape')
                # print(self.db[f'layer{i}'].shape)
                # print('----------------------------------------')

        # Calculate Gradients
        
    def update(self, alpha):
        for i in range(1, self.num_layers + 1):
            self.W[f'layer{i}'] = self.W[f'layer{i}'] - alpha * (self.dw[f'layer{i}'])
            self.b[f'layer{i}'] = self.b[f'layer{i}'] - alpha * (self.db[f'layer{i}'])