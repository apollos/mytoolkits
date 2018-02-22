# -*- coding:utf-8 -*-
from aip import AipOcr
import json


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

'''APPID AK SK'''
APP_ID = '10472621'
API_KEY = 'Ak9PEPez4xDW7xzwQG3kPats'
SECRET_KEY = '7SMjIboC1XE27EE4NNol5SdwUeC6XqjS'

test_file = '/root/workspace/data/icbc-handwriting/chengyan1.jpg'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
''' define option'''
options = {
    'detect_direction': 'true',
    'language_type': 'CHN_ENG',
    'probability': 'true'
}

'''invoke baidu api'''
result = client.basicGeneral(get_file_content(test_file), options)
rst_num = result['words_result_num']
rst_content = result['words_result']

for line in rst_content:
    print line['words']

'''j_rst = json.loads(result, encoding="utf-8")

print j_rst.keys()'''

'''invoke baidu api'''
print '=============================================='
result = client.receipt(get_file_content(test_file))
rst_num = result['words_result_num']
rst_content = result['words_result']
print "Total %d" % rst_num

for line in rst_content:
    print line['location']
    print line['words']
print '=============================================='
result = client.tableRecognition(get_file_content(test_file), {'result_type': 'excel'})
print result
url = result['result']['result_data']
print url
