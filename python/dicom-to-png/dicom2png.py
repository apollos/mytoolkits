import os
import png
import pydicom as dicom
import argparse
import numpy as np
import bisect
import copy


def imadjust(mri_file_path, src, tol=1., vin=[0, 255], vout=[0, 255]):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img
    
    assert len(src.shape) == 2  #'Input image should be 2-dims'
    
    tol = max(0., min(100., tol))
    #print(src.max(), src.min(), tol)

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src, bins=list(range(vout[1]+2)), range=vin)[0]
        
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, vout[1]+1): cum[i] = cum[i - 1] + hist[i]
        
        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)
    
    # Stretching
    if vin[1] - vin[0] != 0:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    else:
        scale = (vout[1] - vout[0])
        print("Warning vin[1] {} == vin[0] {}: {}".format(vin[1], vin[0], mri_file_path))
    vs = src-vin[0]
    vs[src < vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd
    
    return dst


def mri_to_png(mri_file, png_file, mri_file_path):
    """ Function to convert from a DICOM image to png
        @param mri_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """
    
    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    shape = plan.pixel_array.shape
    
    image_2d = plan.pixel_array
    tmp_array = copy.deepcopy(image_2d)
    
    tmp_array = np.asarray(tmp_array, dtype=np.uint16)
    inv = False
    if tmp_array.max() > 32678:
        inv = True
    try:
        offset = getattr(plan, 'RescaleIntercept')
    except AttributeError:
        offset = 0
    #print(offset)
    tmp_array[tmp_array < 0] = 0
    for row in np.arange(shape[0]):
        for col in np.arange(shape[1]):
            if inv:
                if tmp_array[row][col] > 32678:
                    image_2d[row][col] = max(0, int(tmp_array[row][col] - 65536 + offset))
                else:
                    image_2d[row][col] = max(0, int(tmp_array[row][col] + offset))

            else:
                image_2d[row][col] = max(0, int(tmp_array[row][col] + offset))
            """
            #define LEVEL_LUNG 50
            #define WIDTH_LUNG 80
            pix = (pix - (level-width/2))*255/width;
            """
    max_val = image_2d.max()
    min_val = image_2d.min()

    for row in np.arange(shape[0]):
        for col in np.arange(shape[1]):
            image_2d[row][col] = (image_2d[row][col] - min_val)*255/max_val
    if inv:
        tol = max(0., float(max_val - min_val)/10**(len(str(max_val - min_val)) - 1))+0.5
    else:
        tol = max(0., float(max_val - min_val)/10**(len(str(max_val - min_val))))
    if tol >= 4:
        print("Warning: TOL maybe too large:{}, {}, {}: {}".format(
            max_val, min_val, tol, mri_file_path))
    max_val = image_2d.max()
    min_val = image_2d.min()

    image_2d_scaled = imadjust(mri_file_path, image_2d, tol=tol, vin=[min_val, max_val], vout=[0, 255])
    
    image_2d_scaled = np.asarray(image_2d_scaled, dtype=np.uint8)
    w = png.Writer(shape[0], shape[1], greyscale=True)
    w.write(png_file, image_2d_scaled)


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
        os.remove(png_file_path)
    
    mri_file = open(mri_file_path, 'rb')
    png_file = open(png_file_path, 'wb')
    
    mri_to_png(mri_file, png_file, mri_file_path)
    
    png_file.close()


def convert_folder(mri_folder, png_folder):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """
    
    # Create the folder for the pnd directory structure
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    elif not os.path.isdir(png_folder):
        os.remove(png_folder)
        os.makedirs(png_folder)
    
    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)
            
            # Make sure path is an actual file
            if os.path.isfile(mri_file_path) and os.path.splitext(mri_file_path)[1] == '.dcm':
                
                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)
                
                try:
                    # Convert the actual file
                    convert_file(mri_file_path, png_file_path)
                    print('.', end='', flush=True)
                    #print('SUCCESS>', mri_file_path, '-->', png_file_path)
                except Exception as e:
                    print('FAIL>', mri_file_path, '-->', png_file_path, ':', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('dicom_path', help='Full path to the mri file')
    parser.add_argument('png_path', help='Full path to the generated png file')
    
    args = parser.parse_args()
    if args.f:
        convert_folder(args.dicom_path, args.png_path)
    else:
        convert_file(args.dicom_path, args.png_path)
