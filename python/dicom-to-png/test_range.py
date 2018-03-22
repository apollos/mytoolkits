import os
import pydicom as dicom
import argparse
import numpy as np


def mri_to_png(mri_file ):
    """ Function to convert from a DICOM image to png

        @param mri_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """

    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    shape = plan.pixel_array.shape
    image_2d = plan.pixel_array
    max_val = image_2d.max() 
    min_val = image_2d.min()
    tmp_array = np.reshape(image_2d, shape[0]*shape[1])
    tmp_array[::-1].sort()
    idx1 = int(shape[0]*shape[1]*0.025)+1
    idx2 = int(shape[0]*shape[1]*0.0008)+1
    if max_val > 810 and max_val < 830:
        print(mri_file)
        print(min_val, max_val, shape, tmp_array[idx1], tmp_array[idx2])



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

    mri_file = open(mri_file_path, 'rb')
    #png_file = open(png_file_path, 'wb')

    mri_to_png(mri_file )

    #png_file.close()


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
