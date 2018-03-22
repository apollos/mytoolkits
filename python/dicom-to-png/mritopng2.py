# -*- coding: utf-8 -*-
import argparse, os, shutil, time
from PIL import Image
import numpy as np
import pydicom as dicom

parser = argparse.ArgumentParser(
    description='''change all dicom image file in the "./renamed" directory tree.
The name of new target dir is png or jpg. DEFAULT: png, 8-bit (0-255) RGB image
with 256x256 matrix (256x256x3 image channel).
This code doesn't support color-map, only save gray scale image. This code
depends on python3, pydicom, os, numpy, shutil, time and PIL.''',
    usage='dcm2png.py [options]')
parser.add_argument('-S', '-16', action='store_true',
                    help=': use 16(sixteen)-bit scale, 0-66535. 16-bit image can only go with .png, gray scale (one-channel) image.  ')
parser.add_argument('-j', '-jpg', action='store_true',
                    help=': change dicom to jpg  ')
parser.add_argument('-g', '-gray', action='store_true',
                    help=': use gray scale, one channel  ')
parser.add_argument('-t', '-32', action='store_true',
                    help=': save with 32x32 (Thirty-two) imaging matrix  ')
parser.add_argument('-s', '-64', action='store_true',
                    help=': save with 64x64 (Sixty-four) imaging matrix  ')
parser.add_argument('-o', '-128', action='store_true',
                    help=': save with 128x128 (One two eight) imaging matrix  ')
parser.add_argument('-f', '-512', action='store_true',
                    help=': save with 512x512 (Five one two) imaging matrix  ')
parser.add_argument("-indir", nargs= 1, help=": name of dir_tree to collect dicom files, default: './renamed'  ")
parser.add_argument("-outdir", nargs= 1, help=": name of dir where png/jpg file is stored, default: 'png' or 'jpg'. If there is a directory with the same name, an error occurs.  ")
parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.21, Oct/21/2017')
args = parser.parse_args()

start = time.time()
data_type, reso = ['8-bit', 'png', 'RGB'], 256

if args.S:
    data_type[0, 1, 2] = '16-bit', 'png', 'gray_scale'
elif args.j:
    data_type[1] = 'jpg'
elif args.g:
    data_type[2] = 'gray_scale'
if args.t:
    reso = 32
if args.s:
    reso = 64
elif args.o:
    reso = 128
elif args.f:
    reso = 512

print('changing dicom to data type:', data_type)
n, total, sequence_list = 0, 0, []
try:
    in_dir = args.indir[0]
except:
    in_dir = 'renamed' # defaul target directory
try:
    out_dir = args.outdir[0]
except:
    out_dir = data_type[1] # 'png' or 'jpg'

print('in_dir is', in_dir)
print('out_dir is', out_dir)

for root, dirs, files in os.walk(in_dir):
    for file_name in files:
        try:
            file_name_, ext = os.path.splitext(file_name)
            if ext == '.dcm':
                total += 1
            else:
                pass
        except:
            pass

print('total of {} dicom files'.format(total))
n_verbose = total // 50 +1

if os.path.exists(out_dir):
    pass
else:
    os.mkdir(out_dir)
check_point = time.time()

for root, dirs, files in os.walk(in_dir):
    for file_name in files:
        try:
            file_name_, ext = os.path.splitext(file_name)
            if ext == '.dcm':
                root1, root2 = root.split("/", 1)
                save_root = out_dir + "/" + root2 # file path to save png/jpg file
                if os.path.exists(save_root):
                    pass
                else:
                    os.makedirs(save_root)
                file_path = root + "/" + file_name # file path to read dicom file
                try:
                    ds = dicom.read_file(file_path)
                    pix = ds.pixel_array
                    try:
                        pix = pix * ds.RescaleSlope + ds.RescaleIntercept
                    except:
                        pass
                    if data_type[0] == '16-bit':
                        if np.max(pix) - np.min(pix) != 0:
                            pix = (((pix - np.min(pix))/ (np.max(pix) - np.min(pix))
                                    ) * 65535).astype(np.uint16)
                        else:
                            pass
                        pix = pix.astype(np.uint16)
                        save_name = save_root + "/" + file_name_ + ".png"
                        if os.path.exists(save_name):
                            break
                        else:
                            Image.fromarray(np.uint16(pix)).convert('L').resize(
                                (reso, reso), Image.LANCZOS).save(save_name)
                    else: #data_type[0] == '8-bit'
                        if np.max(pix) - np.min(pix) != 0:
                            pix = (((pix - np.min(pix))/ (np.max(pix)-np.min(pix))
                                    ) * 255).astype(np.uint8)
                        else:
                            pass
                        if data_type[1] == 'jpg':
                            save_name = save_root + "/" + file_name_ + ".jpg"
                        else: #elif data_type[1] == 'png':
                            save_name = save_root + "/" + file_name_ + ".png"
                        if os.path.exists(save_name):
                            break
                        else:
                            if data_type[2] == 'gray_scale':
                                Image.fromarray(pix).convert('L').resize(
                                    (reso, reso), Image.LANCZOS).save(save_name)
                            else: #elif data_type[2] == 'RGB':
                                tmp_reso = [pix.shape[0], pix.shape[1]]
                                pix = np.reshape(pix, (tmp_reso[0]*tmp_reso[1], 1))
                                pix2 = np.append(pix, pix, axis = 1)
                                pix = np.append(pix2, pix, axis = 1)
                                pix = np.reshape(pix, (tmp_reso[0], tmp_reso[1], 3))
                                Image.fromarray(np.uint8(pix)).convert('L').resize(
                                    (reso, reso), Image.LANCZOS).save(save_name)
                except:
                    pass
            else:
                pass
        except:
            pass
        
        n += 1
        '''
        if (n % (n_verbose // 20) == 0 and n < n_verbose) or n % n_verbose == 0:
            elapsed_time = time.time() - check_point
            process_speed = n/elapsed_time
            est_total = (elapsed_time/n) * total
            if est_total < 600:
                print("{0}/{1} converted to {2}, {3:0.0f} files/sec. elapsed/est_total: {4:2.0f}/{5:2.0f} sec ".
                      format(n, total,  data_type[1], process_speed, elapsed_time, est_total))
            elif est_total < 4800:
                print("{0}/{1} converted to {2}, {3:0.0f} files/sec. elapsed/est_total: {4}/{5} min ".
                      format(n, total,  data_type[1], process_speed, elapsed_time//60, est_total//60))
            else:
                print("{0}/{1} converted to {2}, {3:0.0f} files/sec. elapsed/est_total: {4:2.1f}/{5:2.1f} hr ".
                      format(n, total,  data_type[1], process_speed, elapsed_time/3600, est_total/3600))
            print()
        '''


elapsed_time = time.time() - start
print("{0} cases processed with {1:2.0f} sec.".format(n, elapsed_time))
print("Finished!!")