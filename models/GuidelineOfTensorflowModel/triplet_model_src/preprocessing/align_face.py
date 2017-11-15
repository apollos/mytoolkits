###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2017 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################
import cv2
import os, glob
import argparse
import detect_face
import tensorflow as tf
import numpy as np
import time

parser = argparse.ArgumentParser(description='Detect&Align faces')
parser.add_argument('--model_path',  default=None, type=str, help='the model path of face detector')
parser.add_argument('--draw_face', default=False, type=bool, help='draw the face detection&alignment results')
parser.add_argument('--imgs_src_path', type=str, help='the source directory of images')
parser.add_argument('--imgs_dst_path', default=None, type=str, help='the destination directory where we store the aligned face')
parser.add_argument('--threshold_0', default=0.6, type=float, help='threhold of stage 1 of face detector')
parser.add_argument('--threshold_1', default=0.7, type=float, help='threhold of stage 2 of face detector')
parser.add_argument('--threshold_2', default=0.7, type=float, help='threhold of stage 3 of face detector')
parser.add_argument('--factor', default=0.709, type=float, help='rescale factor of face detector')
parser.add_argument('--minsize', default=0.1, type=float, help='minsize of face detector')

# # Test phase default setting
# parser.add_argument('--crop_size', default=128, type=int, help='crop_size of face')
# parser.add_argument('--ec_mc_y', default=48, type=int, help='ec_mc_y of face')
# parser.add_argument('--ec_y', default=40, type=int, help='ec_y of face')

# Train phase default settion
parser.add_argument('--crop_size', default=144, type=int, help='crop_size of face')
parser.add_argument('--ec_mc_y', default=48, type=int, help='ec_mc_y of face')
parser.add_argument('--ec_y', default=48, type=int, help='ec_y of face')

parser.add_argument('--max_nfaces_per_image', default=1, type=int, help='maximum number of faces in one images')


#Align face
def guard(x,N):
    x[x<0] = 0
    x[x>N-1] = N-1
    return [int(i) for i in x]

def transform(x, y, trans_rot):
    # x,y position
    # trans_rot rotation matrix
    xx = trans_rot[0,0]*x + trans_rot[0,1]*y + trans_rot[0,2]
    yy = trans_rot[1,0]*x + trans_rot[1,1]*y + trans_rot[1,2]
    return xx, yy

def align(img, f5pt, crop_size, ec_mc_y, ec_y):
    f5pt = f5pt.reshape(2,5).T
    ang_tan = (f5pt[0,1]-f5pt[1,1])/(f5pt[0,0]-f5pt[1,0])
    ang = np.arctan(ang_tan) / np.pi * 180

    center = (0.5*img.shape[0], 0.5*img.shape[1])
    rot = cv2.getRotationMatrix2D(center, ang, 1.0)
    img_rot = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]))

    #eye center
    x = (f5pt[0,0]+f5pt[1,0])/2
    y = (f5pt[0,1]+f5pt[1,1])/2

    [xx, yy] = transform(x, y, rot)
    eyec = np.round([xx, yy])

    #mouth center
    x = (f5pt[3,0]+f5pt[4,0])/2
    y = (f5pt[3,1]+f5pt[4,1])/2
    [xx, yy] = transform(x, y, rot)
    mouthc = np.round([xx, yy])

    resize_scale = ec_mc_y/(mouthc[1]-eyec[1])

    img_resize = cv2.resize(img_rot, None, fx=resize_scale, fy=resize_scale)

    eyec2 = (eyec - np.array([img_rot.shape[1]/2., img_rot.shape[0]/2.])) * resize_scale +\
            np.array([img_resize.shape[1]/2., img_resize.shape[0]/2.])
    eyec2 = np.round(eyec2)

    img_crop = np.zeros((crop_size, crop_size, 3), dtype=img_resize.dtype)

    crop_x = eyec2[0] - np.floor(crop_size / 2.)
    crop_x_end = crop_x + crop_size - 1

    crop_y = eyec2[1] - ec_y
    crop_y_end = crop_y + crop_size - 1

    box = np.concatenate((guard(np.array([crop_x, crop_x_end]), img_resize.shape[1]), \
                          guard(np.array([crop_y, crop_y_end]), img_resize.shape[0])))

    crop_y = int(crop_y)
    crop_x = int(crop_x)
    img_crop[box[2]-crop_y:box[3]-crop_y, box[0]-crop_x:box[1]-crop_x,:] = img_resize[box[2]:box[3],box[0]:box[1],:]
    return img_crop

#Draw face detection results
def draw_detection(img, bounding_boxes, points):
    global args
    args = parser.parse_args()
    crop_size = args.crop_size
    ec_mc_y = args.ec_mc_y
    ec_y = args.ec_y
    nfaces = bounding_boxes.shape[0]
    for iface in range(nfaces):
        face_rect = [int(x) for x in bounding_boxes[iface]]
        landmarks = [int(x) for x in points[:, iface]]

        cv2.rectangle(img, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), [255,0,0], 2)
        for i in range(5):
            cv2.circle(img, (landmarks[i], landmarks[i+5]), 2, [0,255,0], 1)

        face = align(img, points[:, iface], crop_size, ec_mc_y, ec_y)
        cv2.imshow("face #%d"%iface, face)
        cv2.imshow("image #%d"%iface, img)
    cv2.waitKey(0)

def main():
    global args
    args = parser.parse_args()
    print(args)

    all_imgs = {}
    total_num_imgs = 0
    imgs_src_path = args.imgs_src_path
    classes = os.listdir(imgs_src_path)
    for c in classes:
        img_dir = imgs_src_path+"/"+c+"/"
        if not os.path.isdir(img_dir):
            continue
        imgs = os.listdir(img_dir)
        imgs = [img_dir+im_f for im_f in imgs
             if im_f.endswith(".jpg") or im_f.endswith(".jpeg")\
            or im_f.endswith(".png") or im_f.endswith(".bmp")]
        if len(imgs) > 0:
            all_imgs[c] = imgs
            total_num_imgs += len(imgs)
    print("There are ", len(all_imgs.keys()), "classes, total number of images ", total_num_imgs)

    crop_size = args.crop_size
    ec_mc_y = args.ec_mc_y
    ec_y = args.ec_y
    imgs_dst_path = args.imgs_dst_path
    max_nfaces_per_image = args.max_nfaces_per_image

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    start_time = time.time()
    has_processed = 0
    for c in all_imgs:
        for imgfile in all_imgs[c]:
            try:
                img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
                img2 = img.copy()
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img2)
            except Exception as e:
                print("Fail to process ", imgfile, " error: ",e)
                continue
            height, width, channels = img2.shape
            minsize = args.minsize*min(height, width)
            threshold = [args.threshold_0, args.threshold_1, args.threshold_2]  # three steps's threshold
            factor = args.factor  # scale factor
            bounding_boxes, points = detect_face.detect_face(img2, minsize, pnet, rnet, onet, threshold, factor)
            if points.shape[1] == 0:
                continue
            if args.draw_face:
                draw_detection(img, bounding_boxes, points)
            if imgs_dst_path != None:
                img_dst_dir = imgs_dst_path+'/'+c+'/'
                if not os.path.exists(img_dst_dir):
                    os.makedirs(img_dst_dir)
                for iface in range(min(points.shape[1], max_nfaces_per_image)):
                    face = align(img, points[:, iface], crop_size, ec_mc_y, ec_y)
                    img_basename = os.path.basename(imgfile)
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(img_dst_dir + img_basename+"_%d.bmp"%(iface), face)
                    # print(img_dst_dir + img_basename+"_%d.bmp"%(iface))
            has_processed += 1
            if has_processed%(max(total_num_imgs/100, 1)) == 0:
                print("Has processed ", has_processed, "images of ", total_num_imgs, "time elapse", time.time()-start_time,'seconds')

if __name__ == '__main__':
    main()


