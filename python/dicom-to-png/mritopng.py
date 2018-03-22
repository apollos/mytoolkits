import os
import pydicom as dicom
import argparse
import numpy as np
from libtiff import TIFF


def mri_to_png(mri_file_path, png_file_path):
    """ Function to convert from a DICOM image to png

        @param mri_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """

    with open(mri_file_path, 'rb') as mri_file:
        plan = dicom.read_file(mri_file)
        mri_file.close()
        # Extracting data from the mri file
        shape = plan.pixel_array.shape
        image_2d = plan.pixel_array
        pre_max_val = 825
        pre_min_val = 110
        zero_percent = 0.20
        high_percent = 0.02
        tmp_array = np.reshape(image_2d, shape[0]*shape[1])
        tmp_array.sort()
        if tmp_array[int(zero_percent * shape[0] * shape [1])+1] < pre_min_val and \
            tmp_array[-int(high_percent * shape[0] * shape [1])+1] > pre_max_val:
            image_2d[image_2d <= pre_min_val] = 0
            image_2d[image_2d >= pre_max_val] = pre_max_val
            pre_min_val = 0
        elif tmp_array[-1] < pre_min_val: 
            pre_max_val = tmp_array[-1]
            pre_min_val = 0
            image_2d[image_2d <= pre_min_val] = 0
        elif tmp_array[int(zero_percent * shape[0] * shape [1])+1] >= pre_max_val:
            pre_min_val = tmp_array[int(zero_percent * shape[0] * shape [1])+1]
            pre_max_val = tmp_array[-int(high_percent * shape[0] * shape [1])+1]
            image_2d[image_2d <= pre_min_val] = 0
            image_2d[image_2d >= pre_max_val] = pre_max_val
        else:
            pre_min_val = tmp_array[int(zero_percent * shape[0] * shape [1])+1]
            pre_max_val = min(tmp_array[-1], pre_max_val)
            image_2d[image_2d <= pre_min_val] = 0
            image_2d[image_2d >= pre_max_val] = pre_max_val

        # Rescaling grey scale between 0-255
        image_2d_scaled = np.asarray(image_2d, dtype='float')
        image_2d_scaled = image_2d_scaled / pre_max_val*255
        # Writing the PNG file
        tiff = TIFF.open(png_file_path, mode='w')
        tiff.write_image(image_2d_scaled)
        tiff.close()


def convert_file(mri_file_path, png_file_path):
    """ Function to convert an MRI binary file to a
        PNG image file.

        @param mri_file_path: Full path to the mri file
        @param png_file_path: Fill path to the png file
    """

    # Making sure that the mri file exists
    if not os.path.exists(mri_file_path):
        raise Exception('File "%s" does not exists' % mri_file_path)

    # Making sure the png file does not exist
    if os.path.exists(png_file_path):
        raise Exception('File "%s" already exists' % png_file_path)


    mri_to_png(mri_file_path, png_file_path)


def convert_folder(mri_folder, png_folder):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """

    # Create the folder for the pnd directory structure
    os.makedirs(png_folder)

    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)

            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)

                try:
                    # Convert the actual file
                    convert_file(mri_file_path, png_file_path)
                    #print ('SUCCESS>', mri_file_path, '-->', png_file_path)
                except Exception as e:
                    os.remove(png_file_path)
                    print ('FAIL>', mri_file_path, '-->', png_file_path, ':', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('dicom_path', help='Full path to the mri file')
    parser.add_argument('png_path', help='Full path to the generated png file')

    args = parser.parse_args()
    print (args)
    if args.f:
        convert_folder(args.dicom_path, args.png_path)
    else:
        convert_file(args.dicom_path, args.png_path)
