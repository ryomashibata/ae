#coding:utf-8

import numpy as np
import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_data, test_data, valid_data = pickle.load(f, encoding='latin1')

print(type(train_data[0]))

train_data[0].dump("tr_data.dmp")
tr = np.load("tr_data.dmp")

print(len(tr))
