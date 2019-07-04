# USAGE
# python intersection_over_union.py

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

def get_file_list(in_pic, in_label, gt_label):
    #only search current folder
    filelist = []
    replace_patt = re.compile("\.\w+")

    if not (exists(in_pic) and exists(in_label)):
        print("%s or %s does not exists" % (in_pic, in_label))
        return []
    if ((not gt_label is None) and (not exists(gt_label))):
        print("%s does not exists" % (gt_label))
        return []
    if isfile(in_pic):
        if(in_pic.lower().endswith(".jpg") or in_pic.lower().endswith(".png")):
            file_group = [in_pic]
            if isfile(in_label) and in_label.lower().endswith(".xml"):
                file_group.append(in_label)
            else:
                print("AA: %s is not correct" % (in_label))
                return []

            if (not gt_label is None):
                if (isfile(gt_label)) and gt_label.lower().endswith(".xml"):
                    file_group.append(gt_label)
                else:
                    print("FF: %s is not correct" % (gt_label))
                    return []
            filelist.append(file_group)
        else:
            print("DD: %s is not correct" % (in_pic))
            return []
    else: #folder
        for filename in listdir(in_pic): 
            if(filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                file_group = ["%s/%s" % (in_pic, filename)]
                xmlfile = replace_patt.sub(".xml", filename)
                if listdir(in_label) and exists("%s/%s" % (in_label, xmlfile)):
                    file_group.append("%s/%s" % (in_label, xmlfile))
                else:
                    print("BB: %s/%s is not correct" % (in_label, xmlfile))
                    #return []
                    continue

                if (not gt_label is None):
                    if (listdir(gt_label)) and exists("%s/%s" % (gt_label, xmlfile)):
                        file_group.append("%s/%s" % (gt_label, xmlfile))
                    else:
                        print("CC: %s is not correct" % (gt_label))
                        return []
                filelist.append(file_group)
    return filelist

def genXmlInfo(xmlfile):
        
    items = []
    in_file = open(xmlfile)
    tree=ET.parse(in_file)
    root = tree.getroot()
        
    for obj in root.iter('object'):  
        try:   
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            if ((not cls is None) and (not xmlbox is None)):
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                item = [cls, b]
                items.append(item)
        except Exception:
            print ("Miss important key in file %s" % (xmlfile))
            continue

    return items

def main(in_pic, in_label, gt_label=None, save=False):
    # loop over the example detections
    filelist = get_file_list(in_pic, in_label, gt_label)

    if (len(filelist) == 0):
        return
    #print filelist
    label_image_path = "label_image"
    if(save=="True"):
        if not exists(label_image_path):
            mkdir(label_image_path)
    for file_group in filelist:
    # load the image
        image = cv2.imread(file_group[0])
        xml = genXmlInfo(file_group[1])
        for obj in xml:
            # draw the ground-truth bounding box along with the predicted
            # bounding box
            #print obj
            #print obj[1][:2]
            if (obj[1][1] > 10):
                cv2.putText(image, obj[0], (obj[1][0], obj[1][1]-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0, 255, 0) )
            else:
                cv2.putText(image, obj[0], (obj[1][0], obj[1][1]+15), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0, 255, 0) )
            cv2.rectangle(image, obj[1][:2], obj[1][2:], (0, 255, 0), 2) #green
        
        if(len(file_group) == 3):
            xml = genXmlInfo(file_group[2])
            for obj in xml:
                if (obj[1][1] > 10):
                    cv2.putText(image, obj[0], (obj[1][0], obj[1][1]-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0, 0, 255) )
                else:
                    cv2.putText(image, obj[0], (obj[1][0], obj[1][1]+15), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0, 0, 255) )
                cv2.rectangle(image, obj[1][:2], obj[1][2:], (0, 0, 255), 2)	  #red  
        
        # show the output imag
        if(save=="True"):
            filename = basename(file_group[0])
            cv2.imwrite("%s/%s" % (label_image_path, filename), image)
        else:
            cv2.imshow("Image", image)
            cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding box according to the input image file and .xml files')

    parser.add_argument('-p', "--input_pic", action="store", help="Specify the input pic file path or file", required=True, dest="input_pic")
    parser.add_argument('-l', "--input_label", action="store", help="Specify the file or path of the .xml files", required=True, dest="input_label")
    parser.add_argument('-g', "--gt_label", action="store", help="Specify the ground truth .xml files", dest="gt_label")
    parser.add_argument('-s', "--save", action="store", help="Save to jpg file", default=False, dest="save")
    results = parser.parse_args()

    main(results.input_pic, results.input_label, results.gt_label, results.save)

