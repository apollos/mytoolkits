# -*- coding: utf-8 -*-
import os
import multiprocessing
from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences



#input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/simpled/zhwiki-latest-pages-articles1.xml.bz2.txt'
input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/simpled'
output_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/model'

outp1 = 'wiki-zh-model'
outp2 = 'wiki-zh-vector'

sentences = PathLineSentences(input_path)
#sentences = LineSentence(input_path)
model = Word2Vec(sentences, size=400, window=5, min_count=3, workers=multiprocessing.cpu_count())
model.save(os.path.join(output_path, outp1))
model_wv = model.wv
del model

''' save as binary'''
#model_wv.save_word2vec_format(os.path.join(output_path, outp2+".bin"), binary=True)
''' save as txt, each line is a vector'''
model_wv.save_word2vec_format(os.path.join(output_path, outp2+".txt"), binary=False)

