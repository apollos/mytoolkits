# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
from os import listdir, getcwd
from os.path import join
from os.path import exists
from os.path import isdir
from os.path import isfile
from os.path import basename
from os import listdir
from os import mkdir
from os import rmdir
import argparse
import xml.etree.ElementTree as ET
import re
from random import shuffle

def main(input_file, file_ext, ratio):
    # loop over the example detections
    if not isdir(input_file):
        print "%s is not file path." % input_file
    req_files = []
    files = listdir(input_file)
    for filen in files:
        if filen.endswith(file_ext):
            req_files.append(filen)
            
    
    if (len(req_files) == 0):
        print "No file end with %s in %s." % (file_ext, input_file)
        return
    shuffle(req_files)
    
    for i in range(0, int(len(req_files)*ratio)):
        print req_files[i]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate the files from input folder')

    parser.add_argument('-i', "--input_file_path", action="store", help="Specify the input file path", required=True, dest="input_file")
    parser.add_argument('-e', "--extention_file_name", action="store", help="Specify the ext name of file", required=True, dest="file_ext")
    parser.add_argument('-r', "--ratio", action="store", help="Specify the ratio", required=True, default=0.2, dest="ratio")
    results = parser.parse_args()

    main(results.input_file, results.file_ext, float(results.ratio))
