#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import h5py


DATA_ROOT = 'data3'
join = os.path.join
train_file = join(DATA_ROOT, 'train.h5')
test_file = join(DATA_ROOT, 'test.h5')

# logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# generate data
logger.info('Generate data')
sampleSize = 42000
dataSize = (3,3)
data = np.random.randint(0,256, size=(sampleSize, dataSize[0]*dataSize[1]+1))
scalV = np.random.randint(0,256, size=(dataSize[0]*dataSize[1]))

for i in range(len(data)):
    data[i][-1] = sum(data[i][0:-1]*scalV)

logger.info('Get %d Rows in dataset', len(data))

# random shuffle
np.random.shuffle(data)


# all dataset
#print data
labels = data[:, -1].astype(float)
rawData = data[:, 0:-1].astype(float)

# process data
#meanv = np.zeros(len(rawData[0]), dtype=np.float32)
#stdv  = np.zeros(len(rawData[0]), dtype=np.float32)
rawData %= 256
rawData -= 128
rawData = rawData.reshape((len(rawData), 1, 1, len(rawData[0])))

#for k in range(len(rawData[0])):
#    meanv[k] = rawData[:,k,:,:].mean()
#    stdv[k] = rawData[:,k,:,:].std()
#for k in range(len(rawData)):
#    for i in range(len(rawData[0])):
#        rawData[k][i] = (rawData[k][i] - meanv[i])/stdv[i]
#labmeanv = labels.mean()
#labstdv = labels.std()
#labels = (labels - labmeanv)/labstdv
#print rawData[0:1]

labels = np.log(labels)
# train dataset number
trainset = len(labels) * 4 / 5

# train dataset
labels_train = labels[:trainset]
rawData_train = rawData[:trainset]
# test dataset
labels_test = labels[trainset:]
rawData_test = rawData[trainset:]

# write to hdf5
if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(test_file):
    os.remove(test_file)
#print labels_train.astype(np.float32)
#####I am not very sure about the sequence###############
logger.info('Write train dataset to %s', train_file)
with h5py.File(train_file, 'w') as h:
    h.create_dataset('data', data=rawData_train)
    h.create_dataset('score', data=labels_train)
with open('{}_h5.txt'.format(train_file), 'w') as f:
    f.write(train_file)
print rawData_train[0]

logger.info('Write test dataset to %s', test_file)
with h5py.File(test_file, 'w') as h:
    h.create_dataset('data', data=rawData_test)
    h.create_dataset('score', data=labels_test)
with open('{}_h5.txt'.format(test_file), 'w') as f:
    f.write(test_file)

logger.info('Done')
