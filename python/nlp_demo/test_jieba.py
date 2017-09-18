# -*- coding: utf-8 -*-

import codecs
import jieba
infile = '/root/workspace/data/icwb2-data/testing/cityu_test.utf8'
outfile = '/root/workspace/data/pythonjieba/rst_cityu_test.utf8'
descsFile = codecs.open(infile, 'rb', encoding='utf-8')
i = 0
with open(outfile, 'w') as f:
    for line in descsFile:
        i += 1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        words = jieba.cut(line, HMM=True)
        for word in words:
            f.write(word.encode('utf-8') + ' ')
        f.write('\n')
