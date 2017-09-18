# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec

model = Doc2Vec.load('model/phrase_seg.d2v')
print model.docvecs.most_similar('cx_INC01670215.txt')