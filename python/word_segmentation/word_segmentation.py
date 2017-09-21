# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import codecs
import logging
import mylogs
import collections
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.word2vec import LineSentence
import gensim

logLevel = logging.DEBUG
recordLogs = mylogs.myLogs(logLevel)
WORD_DIM = 128
WORD2VEC_WIND = 5
WORD2VEC_MIN_CNT = 1

def generate_file_lst(in_path, file_ext):
    file_list = []
    files = os.listdir(in_path)
    for file_full_path in files:
        if os.path.isfile(os.path.join(in_path, file_full_path)):
            filename, fileext = os.path.splitext(file_full_path)
            if fileext == file_ext:
                file_list.append(os.path.join(in_path, file_full_path))
        elif os.path.isdir(os.path.join(in_path, file_full_path)):
            file_list += generate_file_lst(os.path.join(in_path, file_full_path))
    return file_list


def clean(sentence): # 整理一下数据，有些不规范的地方
    '''
    if u'“' not in sentence:
        tmp_re = re.compile(u'”\s*')
        sentence = tmp_re.sub('', sentence)
    if u'”' not in sentence:
        tmp_re = re.compile(u'“\s*')
        sentence = tmp_re.sub('', sentence)
    if u'‘' not in sentence:
        tmp_re = re.compile(u'’\s*')
        sentence = tmp_re.sub('', sentence)
    if u'’' not in sentence:
        tmp_re = re.compile(u'‘\s*')
        sentence = tmp_re.sub('', sentence)
    '''
    tmp_re = re.compile(u'”\s*|“\s*|’\s*|‘\s*|\'\s*|\"\s*|【\s*|】\s*|（\s*|）\s*|{\s*|}\s*|-\s*|——\s*|_\s*')
    sentence = tmp_re.sub(' ', sentence)
    return sentence


def gen_label_file(input_file_list):
    content = ''
    for input_file in input_file_list:
        recordLogs.logger.info("Read source word file %s" % input_file)
        content += open(input_file).read().decode("utf-8")
        content += "\r\n"
    content_list = content.split('\r\n')
    content_list = u''.join(map(clean, content_list))
    content_list = re.split(u'[，。！？、；]', content_list)
    data = []
    label = []
    for sentence in content_list:
        word_list = sentence.strip().split()
        tmp_data = []
        tmp_label = []
        for word in word_list:
            if word == u'':
                continue
            else:
                tmp_data.append(word)
                if len(word) == 1:
                    tmp_label.append('S')
                else:
                    tmp_label.append('B')
                    for i in range(len(word) - 2):
                        tmp_label.append('M')
                    tmp_label.append('E')
        data.append(tmp_data)
        label.append(tmp_label)
    return data, label


def write_list_to_file(file_path, data):
    output_data = codecs.open(file_path, 'w', 'utf-8')
    for sentence in data:
        text_sentence = ' '.join(sentence)+'\n'
        output_data.write(text_sentence)


def read_file_to_list(file_path):
    input_data = codecs.open(file_path, 'r', 'utf-8')
    data = []
    for sentence in input_data.readlines():
        data.append(sentence.split())
    return data


def gen_w2v_file(output_file, data):
    bigram_transformer = gensim.models.Phrases(data)
    model = Word2Vec(bigram_transformer[data], size=WORD_DIM, window=WORD2VEC_WIND, min_count=WORD2VEC_MIN_CNT,
                     workers=multiprocessing.cpu_count())
    model.wv.save(output_file)
    return model


def main():
    data = []
    label = []
    if FLAGS.gen_label_file:
        input_file_list = generate_file_lst(FLAGS.source_dir, ".utf8")
        recordLogs.logger.info("Start to generate the label file")
        data, label = gen_label_file(input_file_list)
        if FLAGS.output_dir:
            data_file = os.path.join(FLAGS.output_dir, "data.txt")
            label_file = os.path.join(FLAGS.output_dir, "label.txt")
            write_list_to_file(data_file, data)
            write_list_to_file(label_file, label)

    if FLAGS.gen_w2v_file:
        recordLogs.logger.info("Start to generate the word2vector file")
        if len(data) == 0:
            data = read_file_to_list(os.path.join(FLAGS.source_dir, "data.txt"))
        model = gen_w2v_file(os.path.join(FLAGS.output_dir, "w2v.bin"), data)
        """
        rst_lst = model.wv.most_similar(u'教科书')
        for rst in rst_lst:
            print ("%s:%f\n" % (rst[0], rst[1]))
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        default='',
        help='Path to folders of text files for training.',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='Path to folders of text files.'
    )
    parser.add_argument(
        '--gen_label_file',
        action="store_true",
        default=0,
        help="""\
      Translate the train text (word segmentation) to labeled text file\
      """
    )
    parser.add_argument(
        '--gen_w2v_file',
        action="store_true",
        default=0,
        help="""\
      Translate the train text (word segmentation) to labeled text file\
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
