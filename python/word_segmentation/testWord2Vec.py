# -*- coding: utf-8 -*-
import os
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

input_path = 'data/output/'
modelfile = 'w2v.bin'

model = KeyedVectors.load(os.path.join(input_path, modelfile))
'''Test'''
try:
    res = model.wv.most_similar(u'自然环境')
    for res_tmp in res:
        print("%s,%f" % (res_tmp[0], res_tmp[1]))
    print(model.wv.similarity(u'潘金莲', u'普契尼'))
    print(model.wv[u'意大利'])
except KeyError, e:
    print "There is a word that does not exist in the vocabulary: ", e

