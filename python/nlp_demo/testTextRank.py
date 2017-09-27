#-*- encoding:utf-8 -*-

from textrange import TextRank4Keyword
from textrange import TextRank4Sentence
import argparse
import os
import codecs
import mylogs
import logging
import collections

logLevel = logging.DEBUG
recordLogs = mylogs.myLogs(logLevel)
KEYWORD_WINDOW = 2
KEYWORD_NUM = 10
WORD_MIN_LEN = 1
PHRASE_MIN_LEN = 2
SUMMARY_NUM = 3


def generate_file_lst(in_path, file_ext):
    file_list = []
    files = os.listdir(in_path)
    for file_full_path in files:
        if os.path.isfile(os.path.join(in_path, file_full_path)):
            filename, fileext = os.path.splitext(file_full_path)
            if fileext in file_ext:
                file_list.append(os.path.join(in_path, file_full_path))
        elif os.path.isdir(os.path.join(in_path, file_full_path)):
            file_list += generate_file_lst(os.path.join(in_path, file_full_path))
    return file_list


def generate_keywords(file_list):
    tr4w = TextRank4Keyword()
    keywords_dic = collections.defaultdict(dict)
    for file_name in file_list:
        text = codecs.open(file_name, 'r', encoding='utf-8').read()
        tr4w.analyze(text=text, lower=True, window=KEYWORD_WINDOW)   # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
        file_name = os.path.basename(file_name)
        keywords_dic[file_name]['keywords'] = tr4w.get_keywords(KEYWORD_NUM, word_min_len=WORD_MIN_LEN)
        keywords_dic[file_name]['keyphrase'] = tr4w.get_keyphrases(keywords_num=KEYWORD_NUM, min_occur_num=PHRASE_MIN_LEN)
    return keywords_dic


def generate_summary(file_list):
    tr4s = TextRank4Sentence()
    keywords_dic = collections.defaultdict(dict)
    for file_name in file_list:
        text = codecs.open(file_name, 'r', encoding='utf-8').read()
        tr4s.analyze(text=text, lower=True, source='all_filters')
        file_name = os.path.basename(file_name)
        keywords_dic[file_name]['summary'] = tr4s.get_key_sentences(num=SUMMARY_NUM)
    return keywords_dic


def write_dict2file(output_path, keywords_dic, summary_dic):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    keys = keywords_dic.keys()
    for key in keys:
        fd = codecs.open(os.path.join(output_path, key), "w", encoding='utf-8')
        if keywords_dic is not None:
            fd.write("keywords:\n")
            values = keywords_dic[key]['keywords']
            for value in values:
                fd.write("%f: %s\n" % (value.weight, value.word))
            fd.write("keyphrase:\n")
            values = keywords_dic[key]['keyphrase']
            for value in values:
                fd.write("%s\n" % value)
        if summary_dic is not None:
            fd.write("summary:\n")
            values = summary_dic[key]['summary']
            for value in values:
                fd.write("%f: %s\n" % (value.weight, value.sentence))
        fd.close()


def show_dict(keywords_dic, summary_dic):
    if keywords_dic is None and summary_dic is None:
        return
    if keywords_dic is not None:
        keys = keywords_dic.keys()
        for key in keys:
            recordLogs.logger.info("keywords:")
            values = keywords_dic[key]['keywords']
            for value in values:
                recordLogs.logger.info("%f: %s\n" % (value.weight, value.word))
            recordLogs.logger.info("keyphrase:\n")
            values = keywords_dic[key]['keyphrase']
            for value in values:
                recordLogs.logger.info("%s\n" % value)
    if summary_dic is not None:
        keys = summary_dic.keys()
        for key in keys:
            recordLogs.logger.info("summary:\n")
            values = summary_dic[key]['summary']
            for value in values:
                recordLogs.logger.info("%f: %s\n" % (value.weight, value.sentence))


def show_segmentation(file_list):
    tr4w = TextRank4Keyword()

    for file_name in file_list:
        text = codecs.open(file_name, 'r', encoding='utf-8').read()
        tr4w.analyze(text=text, lower=True, window=KEYWORD_WINDOW)   # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
        print("File: %s" % os.path.basename(file_name))
        print('sentences:')
        for s in tr4w.sentences:
            print(s)                 # py2中是unicode类型。py3中是str类型。
        print('words_no_filter')
        for words in tr4w.words_no_filter:
            print('/'.join(words))   # py2中是unicode类型。py3中是str类型。
        print('words_no_stop_words')
        for words in tr4w.words_no_stop_words:
            print('/'.join(words))   # py2中是unicode类型。py3中是str类型。
        print('words_all_filters')
        for words in tr4w.words_all_filters:
            print('/'.join(words))   # py2中是unicode类型。py3中是str类型。


def main():
    keywords_dic = None
    summary_dic = None
    if FLAGS.source_dir:
        input_file_list = generate_file_lst(FLAGS.source_dir, [".txt", ".utf8"])

    if FLAGS.gen_keyword:
        recordLogs.logger.info("Start to generate the keyword and keyphrase")
        keywords_dic = generate_keywords(input_file_list)

    if FLAGS.gen_summary:
        recordLogs.logger.info("Start to generate the summary")
        summary_dic = generate_summary(input_file_list)

    if FLAGS.output_dir:
        write_dict2file(FLAGS.output_dir, keywords_dic, summary_dic)
    else:
        show_dict(keywords_dic, summary_dic)

    if FLAGS.show_seg:
        show_segmentation(input_file_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        default='',
        help='Path to folders of text files for text range.',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='Path to folders of text range result'
    )
    parser.add_argument(
        '--gen_keyword',
        action="store_true",
        default=0,
        help="""\
      Generate keywords and keyphrases of text\
      """
    )
    parser.add_argument(
        '--gen_summary',
        action="store_true",
        default=0,
        help="""\
      Generate summary of text\
      """
    )
    parser.add_argument(
        '--show_seg',
        action="store_true",
        default=0,
        help="""\
      Generate summary of text\
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


