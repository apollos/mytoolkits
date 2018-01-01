from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import random
import re
import cv2
import mylogs
import collections
import numpy as np
import imutils
from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.removeborder import RemoveBorder
from pyimagesearch.getcontours import ContoursDetect

FLAGS = None

recordLogs = mylogs.myLogs()


class ImageProcessConf:
    def __init__(self):
        self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
        self.MIN_NUM_IMAGES_PER_CLASS = 20
        self.MODEL_INPUT_WIDTH = 299
        self.MODEL_INPUT_HEIGHT = 299
        self.MODEL_INPUT_DEPTH = 3
        self.RANDOM_SEED = 65535
        self.CROP_SCALE_THRESHOLD = 39
        self.MIN_SCALE_THRESHOLD = 67
        self.MAX_SCALE_THRESHOLD = 199
        self.BRIGHTNESS_A_MIN = 10
        self.BRIGHTNESS_A_MAX = 30
        self.BRIGHTNESS_BIAS_MIN = 0
        self.BRIGHTNESS_BIAS_MAX = 100
        self.BRIGHTNESS_A_STEP = 1
        self.BRIGHTNESS_BIAS_STEP = 1
        self.ANGLE_MIN = 1
        self.ANGLE_MAX = 359


imageconf = ImageProcessConf()


def create_image_lists(image_dir):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not os.path.exists(image_dir):
        recordLogs.logger.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.defaultdict(dict)
    if os.path.isfile(image_dir):
        result['single_file'] = {
            'dir': '',
            'training': image_dir
        }
        return result

    sub_dirs = [x[0] for x in os.walk(image_dir)]

    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG', 'tiff']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        is_root_dir = False
        if sub_dir == image_dir:
            is_root_dir = True

        recordLogs.logger.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            '''only support search one level directory'''
            if is_root_dir:
                file_glob = os.path.join(image_dir, '*.' + extension)
            else:
                file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            recordLogs.logger.warning('No files found')
            continue
        if len(file_list) < imageconf.MIN_NUM_IMAGES_PER_CLASS:
            recordLogs.logger.warning('Folder has less than %d images, which may cause issues.' %
                                      imageconf.MIN_NUM_IMAGES_PER_CLASS)
        elif len(file_list) > imageconf.MAX_NUM_IMAGES_PER_CLASS:
            recordLogs.logger.warning('Folder {0} has more than {1} images. Some images will \
            never be selected.'.format(dir_name, imageconf.MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            training_images.append(base_name)
        if is_root_dir:
            dir_name = ''
        result[label_name] = {
            'dir': dir_name,
            'training': training_images
        }

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        recordLogs.logger.info('Label does not exist %s.' % label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        recordLogs.logger.info('Category does not exist %s.' % category)
    category_list = label_lists[category]
    if not category_list:
        recordLogs.logger.info('Label %s has no images in the category %s.' % (label_name, category))
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_shuffl_image_list(image_lists, ratio, seed, root):
    random.seed(seed)
    keys = image_lists.keys()
    if ratio > 1:
        ratio = 1
    image_files = {}

    for key in keys:
        random.shuffle(image_lists[key]['training'])
        tmp_img_list = image_lists[key]['training']
        flip_list = []
        for image_file in tmp_img_list[:int(len(tmp_img_list)*ratio)]:
            flip_list.append(os.path.join(root, image_lists[key]['dir'], image_file))
        image_files[key] = flip_list
    return image_files


def flip_left_right(image_lists):
    image_dicts = get_shuffl_image_list(image_lists, FLAGS.flip_left_right, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_dicts.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir

    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:  # Python >2.5
                recordLogs.logger.error("makedirs failed %d" % exc.errno)
                return
        for image_file in image_dicts[key]:
            try:
                img = cv2.imread(image_file)
                flip_mode = random.randrange(-1, 2, 1)
                new_img = cv2.flip(img, flipCode=flip_mode)
                output_file = os.path.join(output_path, "flip_" + os.path.basename(image_file))
                cv2.imwrite(output_file, new_img)
            except cv2.error:
                recordLogs.logger.error("OpenCV error({0})".format(image_file))
                continue


def random_crop(image_lists):
    image_dicts = get_shuffl_image_list(image_lists, FLAGS.random_crop, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_dicts.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_dicts[key]:
            try:
                img = cv2.imread(image_file)
                height, width, channels = img.shape
                scale_value = random.randint(1, imageconf.CROP_SCALE_THRESHOLD)
                scale_height = int(height * (1 - scale_value / 100))
                scale_width = int(width * (1 - scale_value / 100))
                start_y = random.randint(0, height - scale_height)
                start_x = random.randint(0, width - scale_width)
                cropped = img[start_y:start_y+scale_height, start_x:start_x+scale_width]
                resize_flag = False
                output_file = os.path.join(output_path, "crop_" + os.path.basename(image_file))
                if scale_height < imageconf.MODEL_INPUT_HEIGHT:
                    scale_height = imageconf.MODEL_INPUT_HEIGHT
                    resize_flag = True
                if scale_width < imageconf.MODEL_INPUT_WIDTH:
                    scale_width = imageconf.MODEL_INPUT_WIDTH
                    resize_flag = True
                if resize_flag:
                    resized = cv2.resize(cropped, (scale_width, scale_height), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_file, resized)
                else:
                    cv2.imwrite(output_file, cropped)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def random_scale(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, FLAGS.random_scale, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                height, width, channels = img.shape
                scale_value = random.randint(imageconf.MIN_SCALE_THRESHOLD, imageconf.MAX_SCALE_THRESHOLD)
                scale_height = int(height * (scale_value / 100))
                scale_width = int(width * (scale_value / 100))
                output_file = os.path.join(output_path, "scale_" + os.path.basename(image_file))
                resized = cv2.resize(img, (scale_width, scale_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_file, resized)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def random_brightness(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, FLAGS.random_brightness, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                a = random.randrange(imageconf.BRIGHTNESS_A_MIN, imageconf.BRIGHTNESS_A_MAX,
                                     imageconf.BRIGHTNESS_A_STEP)/10
                b = random.randrange(imageconf.BRIGHTNESS_BIAS_MIN, imageconf.BRIGHTNESS_BIAS_MAX,
                                     imageconf.BRIGHTNESS_BIAS_STEP)
                hsv[:, :, 2] = a * hsv[:, :, 2] + b
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                output_file = os.path.join(output_path, "brightness_" + os.path.basename(image_file))
                cv2.imwrite(output_file, img)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def random_rotation(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, FLAGS.random_rotation, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                height, width, channels = img.shape
                (cX, cY) = (width // 2, height // 2)
                angle = random.randrange(imageconf.ANGLE_MIN, imageconf.ANGLE_MAX, 1)
                M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((height * sin) + (width * cos))
                nH = int((height * cos) + (width * sin))
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
                img = cv2.warpAffine(img, M, (nW, nH))
                output_file = os.path.join(output_path, "rot_" + os.path.basename(image_file))
                cv2.imwrite(output_file, img)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def random_noise(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, FLAGS.random_noise, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                noisy1 = img + float(np.random.randint(1, 30, 1))/11 * img.std() * np.random.random(img.shape)
                output_file = os.path.join(output_path, "noise_" + os.path.basename(image_file))
                cv2.imwrite(output_file, noisy1)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def morphological(image_lists, kernel_list):
    image_list_dict = get_shuffl_image_list(image_lists, 1, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for kernel in kernel_list:
                    # construct a rectangular kernel and apply a blackhat operation which
                    # enables us to find dark regions on a light background
                    if kernel == 'blackhat':
                        # kernel size need be modified in future as input parameter
                        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 5))
                        gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
                    elif kernel == 'tophat':
                        # similarly, a tophat (also called a "whitehat") operation will enable
                        # us to find light regions on a dark background
                        # kernel size need be modified in future as input parameter
                        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 5))
                        gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
                    elif kernel == 'dilation':
                        # kernel size need be modified in future as input parameter
                        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        gray = cv2.dilate(gray, rect_kernel, iterations=1)
                    else:
                        recordLogs.logger.error("Unknown morphological kernel: %s" % kernel)
                        return

                output_file = os.path.join(output_path, "-".join(kernel_list)+"_" + os.path.basename(image_file))
                cv2.imwrite(output_file, gray)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def cal_gradient(image_lists, kernel):
    image_list_dict = get_shuffl_image_list(image_lists, 1, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if kernel == 'sobel':
                    # compute gradients along the X and Y axis, respectively
                    gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
                    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

                    # the `gX` and `gY` images are now of the floating point data type,
                    # so we need to take care to convert them back to an unsigned 8-bit
                    # integer representation so other OpenCV functions can utilize them
                    gX = cv2.convertScaleAbs(gX)
                    gY = cv2.convertScaleAbs(gY)

                    # combine the sobel X and Y representations into a single image
                    gray = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
                elif kernel == 'scharr':
                    # compute gradients along the X and Y axis, respectively
                    gX = cv2.Scharr(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
                    gY = cv2.Scharr(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

                    # the `gX` and `gY` images are now of the floating point data type,
                    # so we need to take care to convert them back to an unsigned 8-bit
                    # integer representation so other OpenCV functions can utilize them
                    gX = cv2.convertScaleAbs(gX)
                    gY = cv2.convertScaleAbs(gY)

                    # combine the sobel X and Y representations into a single image
                    gray = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
                else:
                    recordLogs.logger.error("Unknown morphological kernel: %s" % kernel)
                    return

                output_file = os.path.join(output_path, kernel+"_" + os.path.basename(image_file))
                cv2.imwrite(output_file, gray)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def transfer_gray(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, 1, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                #print ("%d %d %d --- %d %d %d" % (img[0][0][0],img[0][0][1],img[0][0][2],img[70][70][0],img[70][70][1],img[70][70][2]))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                output_file = os.path.join(output_path, "gray"+"_" + os.path.basename(image_file))
                cv2.imwrite(output_file, gray)
                #print ("%d --- %d" % (gray[0][0],gray[70][70]))
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def auto_split(image_lists, shape):
    image_list_dict = get_shuffl_image_list(image_lists, 1, imageconf.RANDOM_SEED, FLAGS.image_dir)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                #output_file = os.path.join(output_path, shape+"_" + os.path.basename(image_file))
                #resized = imutils.resize(img, width=300)#we may not need it, let's try big pic
                resized = img
                ratio = img.shape[0] / float(resized.shape[0])

                cnts = cv2.findContours(resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                sd = ShapeDetector()
                # loop over the contours
                print(len(cnts))
                for c in cnts:
                    # compute the center of the contour, then detect the name of the
                    # shape using only the contour
                    print(cv2.contourArea(c))
                    '''
                    if cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 410000:
                        continue
                    '''
                    '''
                    M = cv2.moments(c)
                    cX = int((M["m10"] / M["m00"]) * ratio)
                    cY = int((M["m01"] / M["m00"]) * ratio)

                    shape_predict = sd.detect(c)
                    if shape_predict is not shape:
                        continue
                    '''

                    # multiply the contour (x, y)-coordinates by the resize ratio,
                    # then draw the contours and the name of the shape on the image
                    '''
                    c = c.astype("float")
                    c *= ratio
                    c = c.astype("int")
                    '''
                    # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(c)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    resized = imutils.resize(img, width=1500)
                    cv2.imshow("Image", resized)
                    cv2.waitKey(0)
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def remove_black_border(image_lists):
    image_list_dict = get_shuffl_image_list(image_lists, 1, imageconf.RANDOM_SEED, FLAGS.image_dir)
    keys = image_list_dict.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, 'removeblackborder', image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_list_dict[key]:
            try:
                img = cv2.imread(image_file)
                if np.shape(img)[1] > 800:
                    resized = imutils.resize(img, width=800)
                else:
                    resized = img
                ratio = img.shape[0] / float(resized.shape[0])
                rm_boarder = RemoveBorder()
                x, y, w, h = rm_boarder.remove_border(resized)
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                # Crop with the largest rectangle
                crop = img[y:y + h, x:x + w]

                cv2.imshow("Image", resized)
                cv2.waitKey(0)
                continue
                output_file = os.path.join(output_path, "black_border"+"_" + os.path.basename(image_file))
                cv2.imwrite(output_file, gray)
                #print ("%d --- %d" % (gray[0][0],gray[70][70]))
            except cv2.error:
                recordLogs.logger.info("OpenCV error({0})".format(image_file))
                continue


def main():

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir)
    class_count = len(image_lists.keys())
    if class_count == 0:
        recordLogs.logger.info('No valid folders of images found at ' + FLAGS.image_dir)
        return -1

    # See if the command-line flags mean we're applying any distortions.
    if FLAGS.flip_left_right > 0:
        flip_left_right(image_lists)
    if FLAGS.random_crop > 0:
        random_crop(image_lists)
    if FLAGS.random_scale > 0:
        random_scale(image_lists)
    if FLAGS.random_brightness > 0:
        random_brightness(image_lists)
    if FLAGS.random_rotation > 0:
        random_rotation(image_lists)
    if FLAGS.random_noise > 0:
        random_noise(image_lists)
    if FLAGS.morph is not None:
        morphological(image_lists, FLAGS.morph)
    if FLAGS.calGradient is not None:
        cal_gradient(image_lists, FLAGS.calGradient)
    if FLAGS.toGray is not None and FLAGS.toGray:
        transfer_gray(image_lists)
    if FLAGS.autoSplit is not None:
        auto_split(image_lists, FLAGS.autoSplit)
    if FLAGS.removeBlackBorder is not None and FLAGS.removeBlackBorder:
        remove_black_border(image_lists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Path to folders of distortion.'
    )
    parser.add_argument(
        '--flip_left_right',
        type=float,
        default=0,
        help="""\
      A percentage of randomly flip half of the training images horizontally.\
      """,
    )
    parser.add_argument(
        '--random_crop',
        type=float,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=float,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=float,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
        '--random_rotation',
        type=float,
        default=0,
        help="""\
      A percentage determining how much to randomly rotate the training image\
      """
    )
    parser.add_argument(
        '--random_noise',
        type=float,
        default=0,
        help="""\
      A percentage determining how much to randomly add noise in the training image\
      """
    )
    parser.add_argument(
        '--morph',
        type=str,
        choices=['blackhat', 'tophat', 'dilation'],
        action='append',
        help="""\
      Action for image morphological. Sequence follow the parameter sequence\
      """
    )
    parser.add_argument(
        '--calGradient',
        type=str,
        choices=['sobel', 'scharr'],
        help="""\
      Calculate gradient magnitude and orientation by different kernels\
      """
    )
    parser.add_argument(
        '--toGray',
        action='store_true',
        help="""\
      Transform image to gray\
      """
    )
    parser.add_argument(
        '--autoSplit',
        type=str,
        choices=['rectangle', 'triangle'],
        help="""\
          Automatically detect and crop the specified shape\
          """
    )
    parser.add_argument(
        '--removeBlackBorder',
        action='store_true',
        help="""\
          Remove the blackborder from image\
          """
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
