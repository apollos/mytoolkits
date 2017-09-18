# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import random
import re

import cv2
import mylogs
import collections
import numpy as np

def generate_file_lst(in_path):
    file_list = []
    files = os.listdir(in_path)
    for file_full_path in files:
        if os.path.isfile(os.path.join(in_path, file_full_path)):
            filename, fileext = os.path.splitext(file_full_path)
            if fileext == ".txt":
                file_list.append(os.path.join(in_path, file_full_path))
        elif os.path.isdir(os.path.join(in_path, file_full_path)):
            file_list += generate_file_lst(os.path.join(in_path, file_full_path))
    return file_list


def clean(sentence): # 整理一下数据，有些不规范的地方
    if u'“/s' not in sentence:
        return sentence.replace(u'”/s', '')
    elif u'”/s' not in sentence:
        return sentence.replace(u'“/s ', '')
    elif u'‘/s' not in sentence:
        return sentence.replace(u'’/s', '')
    elif u'’/s' not in sentence:
        return sentence.replace(u'‘/s ', '')
    else:
        return sentence

def gen_label_file(input_file_list, output_path):
    content = ''
    for input_file in input_file_list:
        content += open(input_file).read().decode("utf-8")
        content += "\r\n"
    content_list = content.split('\r\n')
    content_list = u''.join(map(clean, content_list))
    content_list = re.split(u'[，。！？、]/[bems]', content_list)

def main():
    input_file_list = gen_label_file(FLAGS.source_dir)
    if FLAGS.gen_label_file :
        gen_label_file(input_file_list, FLAGS.output_dir)


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
    FLAGS, unparsed = parser.parse_known_args()
    main()
