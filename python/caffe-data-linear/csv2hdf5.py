#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import h5py


DATA_ROOT = 'data'
join = os.path.join
TRAIN = join(DATA_ROOT, 'train.csv')
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

# load data from train.csv
logger.info('Load data from %s', TRAIN)
df = pd.read_csv(TRAIN)
data = df.values
trans_data = []

for k in data:
    if k[-1] != '\N':
        trans_data.append([int(v) for v in k])
data = np.array(trans_data)

logger.info('Get %d Rows in dataset', len(data))

# random shuffle
np.random.shuffle(data)


# all dataset
#print data
labels = data[:, -1].astype(float)
rawData = data[:, 0:-1].astype(float)

# process data
meanv = np.zeros(len(rawData[0]), dtype=np.float32)
stdv  = np.zeros(len(rawData[0]), dtype=np.float32)
rawData = rawData.reshape((len(rawData), 1, 1, len(rawData[0])))
#rawData = rawData.reshape((len(rawData), len(rawData[0]), 1, 1))
for k in range(len(rawData[0])):
    meanv[k] = rawData[:,k,:,:].mean()
    stdv[k] = rawData[:,k,:,:].std()
for k in range(len(rawData)):
    for i in range(len(rawData[0])):
        rawData[k][i] = (rawData[k][i] - meanv[i])/stdv[i]
labmeanv = labels.mean()
labstdv = labels.std()
labels = (labels - labmeanv)/labstdv
#print rawData[0:1]
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
with h5py.File(train_file, 'w') as f:
    f['score'] = labels_train.astype(np.float32)
    f['data'] = rawData_train.astype(np.float32)

logger.info('Write test dataset to %s', test_file)
with h5py.File(test_file, 'w') as f:
    f['score'] = labels_test.astype(np.float32)
    f['data'] = rawData_test.astype(np.float32)

logger.info('Done')
