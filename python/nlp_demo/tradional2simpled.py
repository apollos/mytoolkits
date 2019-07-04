from hanziconv import HanziConv
import os


def read_unicode(text, charset='utf-8'):
    if isinstance(text, basestring):
        if not isinstance(text, unicode):
            text = unicode(text, charset)
    return text


def write_unicode(text, charset='utf-8'):
    return text.encode(charset)


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

input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/source'
output_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/simpled'

space = b" "
i = 0
file_list = generate_file_lst(input_path, ".txt")
if len(file_list) == 0:
    print("Do not find txt file in %s" % input_path)
    exit(-1)
for filepath in file_list:
    output = open(os.path.join(output_path, os.path.basename(filepath)), 'wb')
    txtfile = open(filepath, 'ro')
    print("Start work on " + filepath)
    for text_content in txtfile.readlines():
        line = read_unicode(text_content)
        line = HanziConv.toSimplified(line)
        output.write(write_unicode(line) + b"\n")
    output.close()
print("Transfer all text files")
