# import the necessary packages
from os.path import isdir
from os.path import isfile
from os import listdir
import argparse


def main(source, target):
    # loop over the example detections
    if not isdir(source) and not isfile(source):
        print ("%s is not file path or a file." % source)
        return
    if not isdir(target) and not isfile(target):
        print ("%s is not file path or a file." % target)
        return

    if isdir(source):
        src_files = listdir(source)
    else:
        src_f_h = open(source, 'r')
        src_files = map(lambda x:x.strip(), src_f_h.readlines())
    if isdir(target):
        dst_files = listdir(target)
    else:
        dst_f_h = open(target, 'r')
        dst_files = map(lambda x:x.strip(), dst_f_h.readlines())
    print ("Intersection:")
    print (list(set(src_files) & set(dst_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate the files from input folder')

    parser.add_argument('-s', "--source_path", action="store", help="Specify the source file path", required=True, dest="source")
    parser.add_argument('-t', "--target_path", action="store", help="Specify the target file path", required=True, dest="target")
    results = parser.parse_args()

    main(results.source, results.target)
