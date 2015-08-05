import numpy as np
import pylab
import gzip
import pickle

def output(p, inputs, targets):
    for i in range(p.size):
        pylab.subplot(5, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(inputs[p[i]].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%i' % targets[p[i]])
    pylab.show()

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_data, test_data, valid_data = pickle.load(f, encoding='latin1')

p = np.random.random_integers(0, len(train_data[0]), 25)
output(p, np.array(train_data[0]), np.array(train_data[1]))
