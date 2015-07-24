#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#非線形関数としてシグモイド関数を使用
def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

#誤差逆伝播法の際に用いるf'(Xk) = (1 - f(x))f(x)を関数化したもの
def sigmoid_deriv(u):
    return (u * (1 - u))

#仮想ニューロンの入力処理, 最前列に1を代入
def add_bias(x, axis=None):
    return np.insert(x, 0, 1, axis=axis)

#Multi-Layer Perceptron(多層パーセプトロン)
class MLP(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        #入力→中間層の重みの初期値代入
        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        #中間→出力層の重みの初期値代入
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))
        #3Dplot用のnd配列
        self.x = np.array([])
        self.y = np.array([])

    #leaning_rateは学習係数, 最急降下法で用いる．epochsは学習の回数
    def fit(self, inputs, targets, learning_rate=0.2, epochs=20000):
        inputs = add_bias(inputs, axis=1)
        targets = np.array(targets)

        for loop_cnt in range(epochs):
            p = np.random.randint(inputs.shape[0]) #0~4までの値のどれかを返す
            ip = inputs[p] #inputs_pattern. パターンpの入力信号
            tp = targets[p] #teach_pattern. パターンpの教師信号

            #入力した値を出力するまでの処理
            oj = sigmoid(np.dot(self.v, ip)) #j列のニューロンを求める処理
            oj = add_bias(oj) #j列のニューロンに仮想ニューロンを追加
            ok = sigmoid(np.dot(self.w, oj)) #k列のニューロンを求める処理

            #出力と教師信号の差から重みの修正を行う(back propagation)
            #デルタkを求める処理
            delta_k = sigmoid_deriv(ok)*(ok - tp)
            #デルタjを求める処理
            delta_j = sigmoid_deriv(oj) * np.dot(self.w.T, delta_k)

            oj = np.atleast_2d(oj)
            delta_k = np.atleast_2d(delta_k)
            self.w = self.w - learning_rate * np.dot(delta_k.T, oj) #最急降下法

            ip = np.atleast_2d(ip)
            delta_j = np.atleast_2d(delta_j)
            self.v = self.v - learning_rate * np.dot(delta_j.T, ip)[1:, :] #最急降下法

    def predict(self, inputs):
        #入力した値を出力するまでの処理
        inputs = add_bias(inputs, axis=1)
        for i in inputs:
            oj = sigmoid(np.dot(self.v, i))
            oj = add_bias(oj)
            ok = sigmoid(np.dot(self.w, oj))
            print(ok)


if __name__ == '__main__':
    first_w = np.load("first_w.dmp")
    print(len(first_w))

    mlp = MLP(n_input_units=2, n_hidden_units=3, n_output_units=1)
    #学習
    #mlp.fit(inputs, targets)
    #重みが求まったかを確認する出力部
    print('--- output ---')
    mlp.predict(inputs)
