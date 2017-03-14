import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from os.path import exists
from os.path import isdir
from os import listdir
from collections import namedtuple


def genXmlFileList(file_path):
    #only search current folder
    xmlfilelist = []
    if not exists(file_path):
        print("%s does not exists" % file_path)
        return None
    if not isdir(file_path):
        print("%s is not a dir" % file_path)
        return None
    for file in listdir(file_path):
        if isdir(file):
            rstXmlList = genXmlFileList(file)
            if not rstXmlList is None
                xmlfilelist += rstXmlList
        elif file.endswith(".xml"):
                xmlfilelist.append(file)
        
    return xmlfilelist

def genXmlDict(fileLst):
    
    items = []
    xmlDict = {}
    for xmlfile in fileLst:
        in_file = open(xmlfile)
        tree=ET.parse(in_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if ((w is None) || (h is None)):
            continue

        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            if ((not cls is None) && (not xmlbox is None)):
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                item = [cls, b]
                items.append(item)
         
         if(len( items ) > 0):      
            xmlDict['xmlfile']=[w, h, items]
      return xmlDict

def InforfromXml(file_path):

    if not exists(file_path):
        print("%s does not exists" % file_path)
        return None
    if not isdir(file_path):
        print("%s is not a dir" % file_path)
        return None
    xmlfilelist = genXmlFileList(file_path)
    
    return genXmlDict(xmlfilelist)     
    
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
	

def CaculateIOU( base_info, infiles_info )
    IOU_thresh = 0.5
    average_IOU = 0.0
    total = 0
    correct = 0
    for filename in base_info:
        total += len(base_info[filename][2])
        if filename in infiles_info.keys():
            # (["width", "high", ["class", "xmin", "ymin", "xmax", "ymax"]])
            n = len(infiles_info[filename][2]) if len(infiles_info[filename][2] <= len(base_info[filename][2]) else len(base_info[filename][2])
            best_iou = 0
            count = 0            
            for prediction in infiles_info[filename][2]:                
                for groundtruth in base_info[filename][2]:
                    iou = bb_intersection_over_union(base_info[filename][bboxIdx], infiles_info[filename][b)
                    if (iou > best_iou)
                        best_iou = iou
                count += 1 #object count
                average_IOU += best_iou # all IOU sum 
                if (iou >= IOU_thresh):
                    correct +=1 #good IOU count
                if (count == n):                    
                    break
    average_IOU = average_IOU/total
    return (correct, average_IOU)


def main(infiles, basefiles, outfile):
    """main function"""

    base_info = InforfromXml(basefiles)
    infiles_info = InforfromXml(infiles)
    if(((not base_info is None) && len(base_info) > 0) 
        && ((not infiles_info is None) && len(infiles_info)):        
        mIOU = CaculateIOU(base_info, infiles_info)
        mAP = CaculateAveragePrecise(base_info, infiles_info)    
    
    out_file = open(outfile, 'w')
    out_file.write("mIOU: %f, mAP: %f\n" % (mIOU, mAP))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate is the mAP and IOU for the input result .xml files')

    parser.add_argument('-i', "--input_file_path", action="store", help="Specify the input file path which contain the .xml files", rquired=True, dest="input_file_path")
    parser.add_argument('-b', "--base_file_path", action="store", help="Specify the base result file which contain the .xml files", required=True, dest="base_file_path")
    parser.add_argument('-o', "--output_result", action="store", help="Specify the output file name", default="result.txt",  dest="output_result")
    results = parser.parse_args()

    main(results.input_file_path, results.base_file_path, results.output_result)

