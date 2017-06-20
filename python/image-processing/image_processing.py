from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import random
import re

import cv2

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3


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
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            '''only support search one level directory'''
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            training_images.append(base_name)
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
        print('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        print('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        print('Label %s has no images in the category %s.', label_name, category)
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
        tmp_list = random.shuffle(image_lists[key]['training'])
        flip_list = [os.path.join(root, image_lists[key]['dir'], x) for x in tmp_list[:len(tmp_list)*ratio]]
        image_files[key] = flip_list
    return image_files


def flip_left_right(image_lists):
    image_files = get_shuffl_image_list(image_lists, FLAGS.flip_left_right, 11, FLAGS.image_dir)
    keys = image_files.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir

    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_files[key]:
            try:
                img = cv2.imread(image_file)
                flip_mode = random.randrange(-1, 2, 1)
                new_img = cv2.flip(img, flipCode=flip_mode)
                output_file = os.path.join(output_path, "flip_" + os.path.basename(image_file))
                cv2.imwrite(output_file, new_img)
            except cv2.error:
                print("OpenCV error({0})".format(image_file))
                continue


def random_crop(image_lists):
    image_files = get_shuffl_image_list(image_lists, FLAGS.flip_left_right, 13, FLAGS.image_dir)
    keys = image_files.keys()
    if FLAGS.output_dir is None:
        output_path_root = FLAGS.image_dir
    else:
        output_path_root = FLAGS.output_dir
    for key in keys:
        output_path = os.path.join(output_path_root, image_lists[key]['dir'])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for image_file in image_files[key]:
            try:
                img = cv2.imread(image_file)
                height, width, channels = img.shape
                scale_value = random.randint(1, 39)
                scale_height = int(height * (1 - scale_value / 100))
                scale_width = int(width * (1 - scale_value / 100))
                start_y = random.randint(0, height - scale_height)
                start_x = random.randint(0, width - scale_width)
                cropped = img[start_y:start_y+scale_height, start_x:start_x+scale_width]
                resize_flag = False
                output_file = os.path.join(output_path, "crop_" + os.path.basename(image_file))
                if scale_height < MODEL_INPUT_HEIGHT:
                    scale_height = MODEL_INPUT_HEIGHT
                    resize_flag = True
                if scale_width < MODEL_INPUT_WIDTH:
                    scale_width = MODEL_INPUT_WIDTH
                    resize_flag = True
                if resize_flag:
                    resized = cv2.resize(cropped, (scale_width, scale_height), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_file, resized)
                else:
                    cv2.imwrite(output_file, cropped)
            except cv2.error:
                print("OpenCV error({0})".format(image_file))
                continue


def main():

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.image_dir)
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
        random_scale(image_lists)
    if FLAGS.random_noise > 0:
        random_brightness(image_lists)

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
        default='',
        help='Path to folders of distortion.'
    )
    parser.add_argument(
        '--flip_left_right',
        type=float,
        default=0,
        help="""\
      A percentage of randomly flip half of the training images horizontally.\
      """,
        action='store_true'
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
    FLAGS, unparsed = parser.parse_known_args()
    main()
