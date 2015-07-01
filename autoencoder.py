#config:utf-8

import numpy as np

def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

def sigmoid_deriv(u):
    return (u * (1 - u))

def encode(x):
    return sigmond(numpy.dot())

def decode(x):

class AutoEncoder(object):
    def __init__(self, n_visible_units, n_hidden_units, noise):
        #高性能擬似乱数生成器である"メルセンヌツイスター"を用いた乱数生成.
        self.rng = np.random.RandomState(1)
        #重みの設定.0に近い範囲で重みを設定すると解への収束が早まる
        r = np.sqrt(6. / (n_hidden_units + n_visible_units + 1))
        #encode weight, uniformは-r~rの範囲での一様乱数生成に用いる.
        self.enc_w = np.array(self.rng.uniform(-r, r, (n_visible_units, n_hidden_units)))
        #decode weight. enc_wの転置行列
        self.dec_w = self.enc_w.T
        #バイアス初期化
        self.enc_b = np.zeros(n_hidden_units)
        self.dec_b = np.zeros(n_visible_units)
        #その他初期化
        self.n_visible = n_visible_units
        self.n_hidden = n_hidden_units
        self.noise = noise

    #learning_rateは学習係数, epochsは学習の回数
    def sgd_train(self, learning_rate=0.1, epochs=20):
        print("hoge")

if __name__ == '__main__':
    ae = AutoEncoder(n_visible_units=784, n_hidden_units=100, noise=0.2)
