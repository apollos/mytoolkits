#coding:utf8
import chardet
import argparse
import os

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化                  
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring


def utf_8_encode(in_path, out_path):
    ''' 把filename指定的文件编码改为utf8 '''
    #打开原始文件
    inf = open(in_path, 'r')
    #打开一个对应的新文件
    of = open(out_path, 'w')
    #检测原始文件编码
    encoding = chardet.detect(inf.read())
    #print ('%s the encoding is %s' % (in_path, encoding['encoding']))
    inf.seek(0)
    #转码
    for line in inf.readlines():
        '''
        utf8_line = line.decode('gbk', 'ignore').encode('utf8')
        #utf8_line = strQ2B(line.decode('cp936'))
        of.write(utf8_line)
        '''
        if encoding['encoding'] is None:
            of.write(line.decode('gbk', 'ignore').encode('utf8'))
        elif encoding['encoding'].find("ISO") >= 0: #I do not know why my system can not decode ISO
            of.write(line.decode('gbk', 'ignore').encode('utf8'))
        elif encoding['encoding'] != 'utf-8':
            of.write(line.decode(encoding['encoding'], 'ignore').encode('utf8'))
        else:
            of.write(line)

    of.close()
    inf.close()


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


def main(in_path, out_path):
    if os.path.isfile(in_path):
        if os.path.isfile(out_path):
            utf_8_encode(in_path, out_path)
        elif os.path.isdir(out_path):
            filename, fileext = os.path.splitext(os.path.basename(out_path))
            utf_8_encode(in_path, os.path.join(out_path, filename+"-utf8"+fileext))
        else:
            print("Unsupport target full path %s" % out_path)
            return -1
    elif os.path.isdir(in_path):
        if os.path.isdir(out_path):
            file_list = generate_file_lst(in_path)
            if len(file_list) == 0:
                print("Do not find txt file in %s" % in_path)
                return -1
            for filepath in file_list:
                filename, fileext = os.path.splitext(os.path.basename(filepath))
                utf_8_encode(filepath, os.path.join(out_path, filename+"-utf8"+fileext))
        else:
            print("Unsupport target full path %s" % out_path)
            return -1
    else:
        print("Unsupport input full path %s" % in_path)
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer txt to utf8")
    parser.add_argument('-i', "--input", action="store", help="Specify the input full path", required=True,
                        type=str, dest="in_full_path")
    parser.add_argument('-o', "--output", action="store", help="Specify the output full path", required=True,
                        type=str, dest="out_full_path")
    results = parser.parse_args()
    main(results.in_full_path, results.out_full_path)
