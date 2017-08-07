# -*- coding: utf-8 -*-
import os
import numpy as np
from gensim.models import Word2Vec

input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/simpled/zhwiki-latest-pages-articles1.xml.bz2.txt'
output_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/vector'

input_text = open(input_path, 'ro')
sentence_num = 0
sentence_stat = []
sentences = []
for text in input_text:
    text_array = text.split(" ")
    text_list = list(set(text_array))
    if len(text_list) == 1:
        continue
    sentence_num += 1
    sentence_stat.append(len(text_array))
    sentences.append(text_array)
    if len(text.split(" ")) < 5:
        print text
sentence_stat = np.array(sentence_stat)
max_sent_len = int(np.median(sentence_stat) * 1.7)

input_model_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/model'
modelfile = 'wiki-zh-model'

model = Word2Vec.load(os.path.join(input_model_path, modelfile))
'''Gen Doc Vec'''

word_num = 0
wor2vec_size = 400
print "Start write file"
print "max_sent_len %d" % max_sent_len

unknow_word_list = []
text_num = 0
for text in sentences:
    text_vec_array = np.zeros((max_sent_len, wor2vec_size))
    word_num = 0
    text_num += 1
    for word in text:
        try:
            res = model.wv[unicode(word, 'utf-8')]
            text_vec_array[word_num % max_sent_len] = np.add(text_vec_array[word_num % max_sent_len], res)
            word_num += 1
        except KeyError, e:
            #print "There is a word [%s] that does not exist in the vocabulary: %s" % (word, e)
            unknow_word_list.append(word)
        if word_num == max_sent_len:
            break
    basename = os.path.basename(input_path)
    '''
    vec_file = open(os.path.join(output_path, basename+str(text_num)+".txt"), 'wb')
    first_write = True
    for word_vec in text_vec_array:
        for vec_val in word_vec:
            if not first_write:
                vec_file.write(",")
            else:
                first_write = False
            vec_file.write("%s" % vec_val)
    vec_file.close()
    '''
    np.savetxt(os.path.join(output_path, basename+str(text_num)+".txt"),
               np.reshape(text_vec_array, (max_sent_len*wor2vec_size)), fmt='%.6f', delimiter=',')
print "Wrote Done"
if len(unknow_word_list) > 0:
    unknow_word_list = list(set(unknow_word_list))
    unknow_word_file = open(os.path.join(output_path, "unknown_file.txt"), "wb")
    for word in unknow_word_list:
        unknow_word_file.write("%s\n" % word)
    unknow_word_file.close()
    print("Unknown Word Num is %d" % len(unknow_word_list))


