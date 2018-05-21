# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from numpy.random import randint
import shutil

MIN_CHAR_NUM = 3
MAX_CHAR_NUM = 20
SMALL_FONT_SIZE = [16]  # (8, 12, 16)
MED_FONT_SIZE = (20, 24, 28)
LARGE_FONT_SIZE = (32, 36, 40)
WHITE_BG = (215, 225, 235, 245)
BLACK_BG = (0, 10, 20, 30)
GRAY_BG = (135, 145, 155)
FONT_H_MARGIN = 10
FONT_W_MARGIN = 3
IMG_MARGIN = 32
FONT_RED = [(255, 87, 51), (192, 57, 43), (211, 84, 0), (176, 58, 46), (169, 50, 38)]
FONT_BLUE = [(41, 128, 185), (31, 97, 141), (21, 67, 96), (52, 152, 219), (27, 79, 114)]
FONT_GREEN = [(46, 204, 113), (24, 106, 59), (30, 132, 73), (39, 174, 96), (17, 122, 101)]
FONT_BLACK = [(17, 38, 29), (13, 32, 22), (0, 10, 20), (28, 39, 10), (29, 55, 20)]
FONT_WIGHT = [(244, 246, 247), (151, 154, 154), (166, 172, 175), (229, 231, 233)]  # default it is not in the font list


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_font_list(font_path):
    font_name_lists = [os.path.join(font_path, f) for f in os.listdir(font_path) if os.path.isfile(os.path.join(font_path, f)) and os.path.splitext(f)[1].lower() == '.ttf']
    if len(font_name_lists) == 0:
        font_name_lists = None
    print("Font lists: {}".format(font_name_lists))
    return font_name_lists


def get_character_list(path):
    character_file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() == '.txt']
    if len(character_file_list) == 0:
        return None
    print("Character lists: {}".format(character_file_list))
    character_list = []
    for filename in character_file_list:
        with open(os.path.join(path, filename), encoding='utf8') as f:
            character_list.append(f.read())

    character_list = [f for f in "".join(character_list)]
    print("Total {} character(s)".format(len(character_list)))
    return character_list


def get_bgimg_list(background_img_path):
    img_name_lists = [os.path.join(background_img_path, f) for f in os.listdir(background_img_path) if os.path.isfile(os.path.join(background_img_path, f)) and os.path.splitext(f)[1].lower() in ['.png', '.jpg']]
    if len(img_name_lists) == 0:
        font_name_lists = None
    print("Background Image lists: {}".format(img_name_lists))
    return img_name_lists


def main(FLAGS):
    font_lst = get_font_list(FLAGS.fonts)
    character_lst = get_character_list(FLAGS.characters)
    np.random.seed(FLAGS.seed)
    background_flag = False

    font_color_lst = FONT_RED + FONT_BLUE + FONT_GREEN + FONT_BLACK
    if FLAGS.white_font:
        font_color_lst += FONT_WIGHT
    font_sizes = []
    for font_size in FLAGS.font_size_list:
        font_sizes.extend(eval("{}_FONT_SIZE".format(font_size)))

    background_colors = WHITE_BG + BLACK_BG + GRAY_BG
    if FLAGS.background:
        background_img = get_bgimg_list(FLAGS.background)
        background_flag = True

    if os.path.exists(FLAGS.output):
        shutil.rmtree(FLAGS.output, ignore_errors=True)
    out_img_path = os.path.join(FLAGS.output, "image")
    out_lb_path = os.path.join(FLAGS.output, "label")
    os.makedirs(out_img_path)
    os.makedirs(out_lb_path)

    for cnt in range(FLAGS.target):
        if FLAGS.char_num is not None:
            character_number_per_img = FLAGS.char_num
        else:
            character_number_per_img = randint(MIN_CHAR_NUM, MAX_CHAR_NUM)
        font_name = font_lst[randint(0, len(font_lst))]
        font_size = font_sizes[randint(0, len(font_sizes))]

        resolution_x = randint((font_size+FONT_W_MARGIN)*character_number_per_img,
                               (font_size+FONT_W_MARGIN)*character_number_per_img+randint(1,
                                                                                          FONT_W_MARGIN*character_number_per_img+int(float(font_size)/float(max(font_sizes))*IMG_MARGIN)))
        resolution_y = randint((font_size+FONT_H_MARGIN), randint((font_size+FONT_H_MARGIN+1),
                                                                  font_size+FONT_H_MARGIN+1+int(float(font_size)/float(max(font_sizes))*IMG_MARGIN)))

        if not background_flag:
            background_color = background_colors[randint(0, len(background_colors))]
            char_image = Image.new("RGB", (resolution_x, resolution_y), (background_color, background_color, background_color))
            font_color = 245 - background_color + randint(0, 10)
            font_color = (font_color, font_color, font_color)
        else:
            char_image = Image.open(background_img[randint(0, len(background_img))])
            char_image = char_image.resize((resolution_x, resolution_y))
            font_color = font_color_lst[randint(0, len(font_color_lst))]

        draw = ImageDraw.Draw(char_image)
        font_obj = ImageFont.truetype(font_name, font_size)
        font_y = randint(0, max(resolution_y - (font_size+FONT_H_MARGIN), 2))
        font_x = randint(int(-float(FONT_W_MARGIN)/2.), resolution_x - (font_size+FONT_W_MARGIN)*character_number_per_img)
        content = []
        next_x = 0
        next_y = 0
        for char_idx in range(character_number_per_img):
            rand_idx = randint(0, len(character_lst))
            # Get character width and height
            (font_width, font_height) = font_obj.getsize(character_lst[rand_idx])
            draw.text((font_x + next_x, font_y + next_y), character_lst[rand_idx], font_color, font=font_obj)

            if font_height + font_y + next_y >= resolution_y + 3:
                print("High: resolution_y: {} font_size: {} - {} {} {} {}".format(resolution_y, font_size, font_width, font_height, font_y, next_y))
            if font_width + font_x + next_x >= resolution_x + 3:
                print("Width: resolution_x: {} font_size: {} - {} {} {} {}".format(character_number_per_img, char_idx,
                                                                         resolution_x, font_size, font_width, font_height, font_x, next_x))
            font_x = font_x + next_x + font_width
            next_x = randint(-2,
                             resolution_x - font_x - (character_number_per_img - char_idx - 1) * (font_size+FONT_W_MARGIN))

            # Calculate y position
            next_y = randint(0, int(font_size/4))  # we expect the text at the almost same horizon

            content.append(character_lst[rand_idx])

        # Final file name
        file_name = os.path.join(out_img_path, "img_{}.png".format(cnt))
        # Save image
        char_image.save(file_name)
        # Save label
        label_name = os.path.join(out_lb_path, "img_{}.txt".format(cnt))
        with open(label_name, 'w') as f:
            f.write(" ".join(content))
        #print("Save image file {} and content {}".format(file_name, content))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bg-img-path',
        dest="background",
        type=str,
        help='Path to folders of background images. If not set, will use black and white as background'
    )
    parser.add_argument(
        '--fonts-path',
        type=str,
        default='fonts',
        required=True,
        dest="fonts",
        help="""Path to folder of fonts"""
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        dest="configuration",
        help='Path to the json configuration file for image generation'
    )
    parser.add_argument(
        '--characters-path',
        type=str,
        default='',
        dest="characters",
        required=True,
        help='Path to the candidate character files'
    )
    parser.add_argument(
        '--target-num',
        type=int,
        default=10,
        dest="target",
        help='To be generated image number'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default="output",
        dest="output",
        help='Output image location'
    )
    parser.add_argument(
        '--font-size-list',
        type=str,
        choices=["SMALL", "MED", "LARGE"],
        default=["SMALL", "MED", "LARGE"],
        nargs='+',
        dest="font_size_list",
        help='To be used font size. It can be multiple choices'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=65535,
        dest="seed",
        help='Random Seed'
    )
    parser.add_argument(
        '--char-num',
        type=int,
        dest="char_num",
        help='Generated character number. If not set, will be random between 3 - 20'
    )
    parser.add_argument(
        '--white-font',
        type=str2bool,
        nargs='?',
        const=True,
        dest="white_font",
        help='Add white color font in the generate list'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
