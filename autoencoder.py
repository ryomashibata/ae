#config:utf-8

import numpy as np
import pickle
import gzip
from PIL import Image

def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

def sigmoid_deriv(u):
    return (u * (1 - u))

class AutoEncoder(object):
    def __init__(self, n_visible_units, n_hidden_units, noise):
        #create merusenne twister
        self.rng = np.random.RandomState(1)
        #initial weight scope
        r = np.sqrt(6. / (n_hidden_units + n_visible_units + 1))
        #encode weight setting
        self.enc_w = np.array(self.rng.uniform(-r, r, (n_visible_units, n_hidden_units)))
        #decode weight setting
        self.dec_w = self.enc_w.T
        #bias setting
        self.enc_b = np.zeros(n_hidden_units)
        self.dec_b = np.zeros(n_visible_units)
        #initial value setting
        self.n_visible = n_visible_units
        self.n_hidden = n_hidden_units
        self.noise = noise

    #learning_rate, epochs:repeat learning count
    def sgd_train(self, learning_rate=0.1, epochs=20):
        #minibatch algorizhm
        #partition length learning_data
        batch_num = len(X) / batch_size

        #online
        for epoch in range(epochs):
            total_cost = 0.0 #sum error(gosa)
            #batch
            for i in range(batch_num):
                batch = X[i*batch_size : (i+1)*batch_size] #slice

                cost, gradEnc_w, gradDec_w, gradEnc_b, gradDec_b = \
                    self.partial_differentiation(batch, len(X)) #partial differentiation

                #update weight and bias
                total_cost += cost
                self.enc_w -= learning_rate * gradEnc_w
                self.dec_w -= learning_rate * gradDec_w
                self.enc_b -= learning_rate * gradEnc_w
                self.dec_b -= learning_rate * gradDec_w

                grad_sum = gradEnc_w.sum() + gradDec_w.sum() + gradEnc_w.sum() + gradDec_w.sum()

            print(epoch)
            print((1. / batch_num) * total_cost)

    def partial_differentiation(self, x_batch, dnum):

        #cost:diff enc,dec. initial grad
        cost = 0.
        grad_enc_w = np.zeros(self.enc_w.shape)
        grad_dec_w = np.zeros(self.dec_w.shape)
        grad_enc_b = np.zeros(self.enc_b.shape)
        grad_dec_b = np.zeros(self.dec_b.shape)

        for x in x_batch:
            #add noise in data.
            tilde_x = self.corrupt(x, self.noise)
            #encode
            y = self.encode(tilde_x)
            #decode
            z = self.decode(y)

            #get_cost:L = -Lh = xlogz + (1-x)log(1-z)
            cost += self.get_cost(x,z)

            delta1 = - (x - z)

            #np.outer: outer product => nerrow sense tensor product
            grad_enc_w += np.outer(delta1, y).T
            grad_dec_b += delta1

            delta2 = np.dot(self.dec_w.T, delta1) * self.sigmoid_prime(y)
            grad_enc_w += np.outer(delta2, tilde_x)
            grad_enc_b += delta2

        #Normalization
        cost /= len(x_batch)
        grad_enc_w /= len(x_batch)
        grad_dec_w /= len(x_batch)
        grad_enc_b /= len(x_batch)
        grad_dec_b /= len(x_batch)

        return cost, grad_enc_w, grad_dec_w, grad_enc_b, grad_dec_b

    def encode(self, x):
        #evaluate y
        return sigmoid(np.dot(self.enc_w, x) + self.enc_b)

    def decode(self, y):
        #evaluate z
        return sigmoid(np.dot(self.dec_w, y) + self.dec_b)

    def corrupt(self, x, noise):
        #make noise
        return self.rng.binomial(size = x.shape, n = 1, p = 1.0 - noise) * x

    def get_cost(self, x, z):
        eps = 1e-10
        return - np.sum((x * np.log(z + eps) + (1.-x) * np.log(1.-z + eps)))

    def display(self):
        tile_size = (int(np.sqrt(self.enc_w[0].size)), int(np.sqrt(self.enc_w[0].size)))

        panel_shape = (10, 10)

        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

        #panel_shape = (int(np.sqrt(self.enc_w.shape[0])), int(np.sqrt(self.enc_w.shape[0])))
        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

if __name__ == '__main__':
    ae = AutoEncoder(n_visible_units=784, n_hidden_units=100, noise=0.2)
    #load_data
    #with gzip.open('mnist.pkl.gz', 'rb') as f:
    #    train_data, test_data, valid_data = pickle.load(f)

    #ae.sgd_train()
