#coding:utf-8

import numpy as np
import gzip
import pickle
import pylab

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_data, test_data, valid_data = pickle.load(f, encoding='latin1')

p = np.random.random_integers(0, len(test_data[0]), 25)
test_data = np.array(test_data)

for index in range(25):
    pylab.subplot(5, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(data.reshape(28 ,28), cmap=pylab.cm.gray_r, interpolation='nearset')
    pylab.title('%i' % label)
pylab.show()
