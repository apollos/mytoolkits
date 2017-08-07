from gensim.corpora import WikiCorpus
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

input_path = '/home/yu/workspace/Data/wiki'

space = b" "
i = 0
file_list = generate_file_lst(input_path, ".bz2")
if len(file_list) == 0:
    print("Do not find bz2 file in %s" % input_path)
    exit(-1)
for filepath in file_list:
    output = open(os.path.basename(filepath)+".txt", 'wb')
    wiki = WikiCorpus(filepath, lemmatize=False, dictionary={})
    print("Start work on " + filepath)
    for text in wiki.get_texts():
        line = read_unicode(space.join(text))
        output.write(write_unicode(line) + b"\n")
        i = i + 1
        if i % 10000 == 0:
            print("Saved " + str(i) + " articles")
    output.close()
    print("Finished Saved " + str(i) + " articles")
print("Transfer all bz2 files")