# -*- coding: utf-8 -*-
import os
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
#from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import multiprocessing
from random import shuffle


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        self.sentences = []

    def __iter__(self):
        for source in self.sources:
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [os.path.basename(source)])

    def to_array(self):
        self.sentences = []
        for source in self.sources:
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [os.path.basename(source)]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def generate_file_lst(in_path, postfix):
    file_list = []
    if os.path.isdir(in_path):
        files = os.listdir(in_path)
        for file_name in files:
            if os.path.isfile(os.path.join(in_path, file_name)):
                filename, fileext = os.path.splitext(file_name)
                if fileext == postfix:
                    file_list.append(os.path.join(in_path, file_name))
            elif os.path.isdir(os.path.join(in_path, file_name)):
                file_list += generate_file_lst(os.path.join(in_path, file_name))
    else:
        if os.path.isfile(in_path):
            filename, fileext = os.path.splitext(in_path)
            if fileext == postfix:
                file_list.append(in_path)
    return file_list

source_path = "/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/phrase_seg"
file_prefix = ["cx_", "sx_"]
file_list = generate_file_lst(source_path, ".txt")
sentences = LabeledLineSentence(file_list)

model = Doc2Vec(min_count=1, window=10, size=256, sample=1e-4, negative=3, workers=multiprocessing.cpu_count(),
                alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences.to_array())
print "Start training: %d, %d" % (model.iter, model.corpus_count)
try:
    model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=10)
except Exception as e:
    print("Error %s" % e)
    exit(1)
print "End training"
model.save('model/phrase_seg.d2v')
print model.docvecs.most_similar('cx_INC01975504.txt')

