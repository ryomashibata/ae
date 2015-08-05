#config:utf-8

import numpy as np
import pickle
import gzip
from PIL import Image
import pylab

def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

def sigmoid_deriv(u):
    return (u * (1 - u))

def scale(x):
    eps = 1e-8
    x = x.copy()
    x -= x.min()
    x *= 1.0 / (x.max() + eps)
    return 255.0*x

class AutoEncoder(object):
    def __init__(self, n_visible_units, n_hidden_units, noise):
        #create merusenne twister
        self.rng = np.random.RandomState(1)
        #initial weight scope
        r = np.sqrt(6. / (n_hidden_units + n_visible_units + 1))
        #encode weight setting
        self.enc_w = np.array(self.rng.uniform(-r, r, (n_visible_units, n_hidden_units)))
        #bias setting
        self.enc_b = np.zeros(n_hidden_units)
        self.dec_b = np.zeros(n_visible_units)
        #initial value setting
        self.n_visible = n_visible_units
        self.n_hidden = n_hidden_units
        self.noise = noise
        print('hoge')

    def encode(self, x):
        #evaluate y
        return sigmoid(np.dot(self.enc_w.T, x) + self.enc_b)

    def decode(self, y):
        #evaluate z
        return sigmoid(np.dot(self.enc_w, y) + self.dec_b)

    def corrupt(self, x, noise):
        #make noise
        return self.rng.binomial(size = x.shape, n = 1, p = 1.0 - noise) * x

    def get_cost(self, x, z):
        eps = 1e-10
        return - np.sum((x * np.log(z + eps) + (1.-x) * np.log(1.-z + eps)))

    def partial_differentiation(self, x_batch):

        #cost:diff enc,dec. initial grad
        cost = 0.
        grad_enc_w = np.zeros(self.enc_w.shape)
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

            alpha_h2 = x - z
            alpha_h1 = np.dot(self.enc_w.T, alpha_h2) * sigmoid_deriv(y)

            grad_enc_b = alpha_h1
            grad_dec_b = alpha_h2

            alpha_h1 = np.atleast_2d(alpha_h1)
            tilde_x = np.atleast_2d(tilde_x)
            alpha_h2 = np.atleast_2d(alpha_h2)
            y = np.atleast_2d(y)

            grad_enc_w = (np.dot(alpha_h1.T, tilde_x) + (np.dot(alpha_h2.T, y)).T).T

        #Normalization
        cost /= len(x_batch)
        grad_enc_w /= len(x_batch)
        grad_enc_b /= len(x_batch)
        grad_dec_b /= len(x_batch)

        return cost, grad_enc_w, grad_enc_b, grad_dec_b

    #learning_rate, epochs:repeat learning count
    def sgd_train(self, X, learning_rate=0.5, epochs=5, batch_size = 20):
        #minibatch algorizhm
        #partition length learning_data
        batch_num = len(X) / batch_size
        batch_num = int(batch_num)

        #online
        for epoch in range(epochs):
            total_cost = 0.0 #sum error(gosa)
            #batch
            for i in range(batch_num):
                batch = X[i*batch_size : (i+1)*batch_size] #slice

                cost, gradEnc_w, gradEnc_b, gradDec_b = \
                    self.partial_differentiation(batch) #partial differentiation

                #update weight and bias
                total_cost += cost
                self.enc_w += learning_rate * gradEnc_w
                self.enc_b += learning_rate * gradEnc_b
                self.dec_b += learning_rate * gradDec_b

            print(epoch)
            print((1. / batch_num) * total_cost)

    def conpare_image(self, p, inputs, targets):
        rt = int(np.sqrt(inputs[0].size))
        for i in range(p.size):
            tilde_x = self.corrupt(inputs[p[i]], self.noise)
            y = self.encode(tilde_x)
            z = self.decode(y)
            pylab.subplot(2, p.size, i + 1)
            pylab.axis('off')
            pylab.imshow(tilde_x.reshape(rt, rt), cmap=pylab.cm.gray_r, interpolation='nearest')
            pylab.title('nimg %i' % targets[p[i]])
            pylab.subplot(2, p.size, i + 6)
            pylab.axis('off')
            pylab.imshow(z.reshape(rt, rt), cmap=pylab.cm.gray_r, interpolation='nearest')
            pylab.title('dimg %i' % targets[p[i]])
        pylab.show()

    def fix_parameters(self, X):
        data = []
        for i in range(int(len(X))):
            tilde_x = self.corrupt(X[i], self.noise)
            data.append(self.encode(tilde_x))
        return data

    def display(self):
        tile_size = (int(np.sqrt(self.enc_w[0].size)), int(np.sqrt(self.enc_w[0].size)))
        panel_shape = (10, 10)
        margin_y = np.zeros(tile_size[1])
        margin_x = np.zeros((tile_size[0] + 1) * panel_shape[0])
        image = margin_x.copy()

        for y in range(panel_shape[1]):
            tmp = np.hstack( [ np.c_[ scale( x.reshape(tile_size) ), margin_y ]
                for x in self.enc_w[y*panel_shape[0]:(y+1)*panel_shape[0]]])
            tmp = np.vstack([tmp, margin_x])

            image = np.vstack([image, tmp])

        img = Image.fromarray(image)
        img = img.convert('RGB')
        img.show()

        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

        #panel_shape = (int(np.sqrt(self.enc_w.shape[0])), int(np.sqrt(self.enc_w.shape[0])))
        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

if __name__ == '__main__':
    #load_data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_data, test_data, valid_data = pickle.load(f, encoding='latin1')

    #train_data[0] => len(train_data)=2
    first = AutoEncoder(n_visible_units=784, n_hidden_units=256, noise=0.1)
    first.sgd_train(train_data[0])
    #p = np.random.random_integers(0, len(valid_data[0]), 5)
    #first.conpare_image(p, np.array(valid_data[0]), np.array(valid_data[1]))
    data = first.fix_parameters(train_data[0])
    vl_data = first.fix_parameters(valid_data[0])

    second = AutoEncoder(n_visible_units=256, n_hidden_units=100, noise=0.2)
    second.sgd_train(data)
    #p = np.random.random_integers(0, len(valid_data[0]), 5)
    #second.conpare_image(p, np.array(vl_data), np.array(valid_data[1]))
    data = second.fix_parameters(data)
    vl_data = second.fix_parameters(vl_data)

    third = AutoEncoder(n_visible_units=100, n_hidden_units=49, noise=0.3)
    third.sgd_train(data)
    #p = np.random.random_integers(0, len(valid_data[0]), 5)
    #third.conpare_image(p, np.array(vl_data), np.array(valid_data[1]))
    data = third.fix_parameters(data)
    vl_data = third.fix_parameters(vl_data)

    last = AutoEncoder(n_visible_units=49, n_hidden_units=25, noise=0.3)
    last.sgd_train(data)
    p = np.random.random_integers(0, len(valid_data[0]), 5)
    last.conpare_image(p, np.array(vl_data), np.array(valid_data[1]))
    Pre_data = last.fix_parameters(data)
    print(Pre_data)
    #third.display()
