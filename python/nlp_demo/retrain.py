# -*- coding: utf-8 -*-
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

input_model_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/model'
input_txt_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/new'
modelfile = 'wiki-zh-model'
modelvecfile = 'wiki-zh-vector'

model = Word2Vec.load(os.path.join(input_model_path, modelfile))
sentences = PathLineSentences(input_txt_path)
'''train'''
try:
    model.train(sentences, epochs=100, word_count=1, total_examples=5200,
                total_words=1104602)

    print(model.wv[unicode('宋煜', 'utf-8')])
except KeyError, e:
    print "There is a word that does not exist in the vocabulary: ", e
if os.path.exists(os.path.join(input_model_path, modelfile+"new")):
    os.remove(os.path.join(input_model_path, modelfile+"new"))
if os.path.exists(os.path.join(input_model_path, modelvecfile+"_new.bin")):
    os.remove(os.path.join(input_model_path, modelvecfile+"_new.bin"))
if os.path.exists(os.path.join(input_model_path, modelvecfile+"_new.txt")):
    os.remove(os.path.join(input_model_path, modelvecfile+"_new.txt"))

model.save(os.path.join(input_model_path, modelfile+"new"))
model_wv = model.wv
del model
''' save as binary'''
model_wv.save_word2vec_format(os.path.join(input_model_path, modelvecfile+"_new.bin"), binary=True)
''' save as txt, each line is a vector'''
model_wv.save_word2vec_format(os.path.join(input_model_path, modelvecfile+"_new.txt"), binary=False)

