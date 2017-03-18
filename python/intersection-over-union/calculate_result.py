import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from os.path import exists
from os.path import isdir
from os import listdir
import argparse
import re


def genXmlFileList(file_path):
    #only search current folder
    xmlfilelist = []

    if not exists(file_path):
        print("%s does not exists" % file_path)
        return None
    if not isdir(file_path):
        print("%s is not a dir" % file_path)
        return None
    for filename in listdir(file_path):        
        if isdir("%s/%s" % (file_path, filename)):
            rstXmlList = genXmlFileList("%s/%s" % (file_path, filename))
            if not rstXmlList is None:
                xmlfilelist += rstXmlList
        elif filename.endswith(".xml"):
                xmlfilelist.append("%s/%s" % (file_path, filename))
    return xmlfilelist

def genXmlDict(fileLst, file_path, gt_flag=False):
        
    xmlDict = {}
    gtDict = {}
    for xmlfile in fileLst:
        in_file = open(xmlfile)
        tree=ET.parse(in_file)
        root = tree.getroot()
        
        size = root.find('size')
        try:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        except:
            print "miss key value in %s" % (xmlfile)
            return (xmlDict, gtDict)
        if ((w is None) or (h is None)):
            continue
        items = []
        for obj in root.iter('object'):  
            try:   
                cls = obj.find('name').text
                if(not gt_flag):
                    prb = obj.find('possibility').text
                else:
                    prb = 1
                if cls in gtDict.keys():
                    gtDict[cls] += 1
                else:
                    gtDict[cls] = 1
                xmlbox = obj.find('bndbox')
                if ((not cls is None) and (not xmlbox is None)):
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                    item = [cls, prb, b]
                    items.append(item)
            except Exception:
                print "Miss important key in file %s" % (xmlfile)
                continue

        if(len( items ) > 0):  
            p = re.compile(file_path+"/*")    
            xmlfile = p.sub("", xmlfile)
            xmlDict[xmlfile]=[w, h, items]
    return (xmlDict, gtDict)

def InforfromXml(file_path, gt_flag=False):

    if not exists(file_path):
        print("%s does not exists" % file_path)
        return None
    if not isdir(file_path):
        print("%s is not a dir" % file_path)
        return None
    xmlfilelist = genXmlFileList(file_path)
    
    return genXmlDict(xmlfilelist, file_path, gt_flag)     
    
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #print boxA, boxB
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    if xB < xA or yB < yA : 
        return 0
    else:
        interArea = (xB - xA + 1) * (yB - yA + 1)
    
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        #print boxA, boxB, iou
        # return the intersection over union value
        return iou
	

def CaculateIOU( base_info, infiles_info ):
    IOU_thresh = 0.5
    average_IOU = 0.0
    total = 0
    correct = 0
    predict_list = {}
    max_IOU = 0
    min_IOU = 1
    best_file=""
    worst_file = ""

    for filename in base_info:
        total += len(base_info[filename][2])
        if filename in infiles_info.keys():
            # (["width", "high", ["class", "possibility", ("xmin", "ymin", "xmax", "ymax")]])            
            n = len(infiles_info[filename][2]) if len(infiles_info[filename][2]) <= len(base_info[filename][2]) else len(base_info[filename][2]) 
            count = 0
            for prediction in infiles_info[filename][2]:
                best_iou = 0
                count += 1 
                for groundtruth in base_info[filename][2]:  
                    iou = bb_intersection_over_union(groundtruth[2], prediction[2])
                    if (iou > best_iou):
                        best_iou = iou
                        if groundtruth[0] == prediction[0]:
                            tmp_prd = [prediction[1], 1, prediction[0]]
                        else:
                            tmp_prd = [prediction[1], 0, prediction[0]]
                average_IOU += best_iou # all IOU sum 
                if (best_iou > max_IOU):
                        max_IOU = best_iou
                        best_file = filename
                if (best_iou < min_IOU):
                        min_IOU = best_iou
                        worst_file = filename
                if tmp_prd[2] in predict_list.keys():
                    predict_list[tmp_prd[2]].append(tmp_prd[:2])
                else:
                    predict_list[tmp_prd[2]] = [tmp_prd[:2]]
                #print filename, ": ", iou, best_iou, average_IOU
                if (best_iou >= IOU_thresh):
                    correct +=1 #good IOU count
                if (count == n):                    
                    break
    average_IOU = average_IOU/total
    recall = float(correct)/float(total)
    #print correct, total
    print "best %s; worst %s" % (best_file, worst_file)
    return (recall, average_IOU, predict_list, total)

def ap(predict_rst, positive_num, total):
    #[possibilty, true/false]
    predict_rst.sort(key=lambda tup: tup[0], reverse=True)
    apv = 0.0
    #print predict_rst
    last_recall = 0.0
    for i in range(1, total):
        TP = 0.0        
        if i > len(predict_rst):
            break
        for j in range(0, i):
            TP += predict_rst[j][1]
        precision = TP/i
        recall = TP/positive_num
        apv += precision*(recall - last_recall)
        last_recall = recall
        #print precision, recall, apv
    return apv


def CaculateAveragePrecise(predict_list, gt_dict, total):
    all_ap = 0.0
    for cls in predict_list.keys():
        if cls in gt_dict.keys():            
            if gt_dict[cls] == 0:
                print "%s positive number is 0" % (cls)
                continue
            all_ap += ap(predict_list[cls], gt_dict[cls], total)
        
    return all_ap/len(gt_dict.keys())

def main(infiles, basefiles, outfile):
    """main function"""
    recall = 0
    mIOU = 0
    mAP=0 
    base_info, gt_dict = InforfromXml(basefiles, gt_flag=True)
    infiles_info, pred_dict = InforfromXml(infiles)

    if(((not base_info is None) and len(base_info) > 0)
        and ((not infiles_info is None) and len(infiles_info) > 0)):        
        recall, mIOU, predict_list, total = CaculateIOU(base_info, infiles_info)
        mAP = CaculateAveragePrecise(predict_list, gt_dict, total)         
    out_file = open(outfile, 'w')
    out_file.write("recall: %f, mIOU: %f, mAP: %f\n" % (recall, mIOU, mAP))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate is the mAP and IOU for the input result .xml files')

    parser.add_argument('-i', "--input_file_path", action="store", help="Specify the input file path which contain the .xml files", required=True, dest="input_file_path")
    parser.add_argument('-b', "--base_file_path", action="store", help="Specify the base result file which contain the .xml files", required=True, dest="base_file_path")
    parser.add_argument('-o', "--output_result", action="store", help="Specify the output file name", default="result.txt",  dest="output_result")
    results = parser.parse_args()

    main(results.input_file_path, results.base_file_path, results.output_result)

