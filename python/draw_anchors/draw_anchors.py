import cv2
import numpy as np
import sys
import argparse




def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-anchor_file', required=True, 
                        help='path to anchors\n', )
    
    
    args = parser.parse_args()
    
    
    print "anchors list you provided{}".format(args.anchor_file)
    print 'differ'


    f = open(args.anchor_file)
    line = f.readline().rstrip('\n')


    anchors = line.split(', ')


    [H,W] = (416,416)
    stride = 32
    blank_image = np.zeros((H,W,3),np.uint8)
    
    cv2.namedWindow('Image')
    cv2.moveWindow('Image',100,100)


    colors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(0,255,255)]
    for i in range(len(anchors)):
        (w,h) = map(float,anchors[i].split(','))




        w=int (w*stride)
        h=int(h*stride)
        print w,h
        cv2.rectangle(blank_image,(0,0),(w,h),colors[i])


        cv2.imshow('Image',blank_image)
        cv2.imwrite('anchors.png',blank_image)
        #cv2.waitKey(10000)


if __name__=="__main__":
    main(sys.argv)
