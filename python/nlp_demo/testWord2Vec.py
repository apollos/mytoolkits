# -*- coding: utf-8 -*-
import os
from gensim.models import Word2Vec

input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/model'
modelfile = 'wiki-zh-model'

model = Word2Vec.load(os.path.join(input_path, modelfile))
'''Test'''
try:
    res = model.wv.most_similar(unicode('杜甫', 'utf-8'))
    for res_tmp in res:
        print("%s,%f" % (res_tmp[0], res_tmp[1]))
    print(model.wv.similarity('woman', 'red'))
    print(model.wv[unicode('美国', 'utf-8')])
except KeyError, e:
    print "There is a word that does not exist in the vocabulary: ", e

