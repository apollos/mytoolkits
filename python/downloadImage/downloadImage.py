import os
import argparse
import multiprocessing
import urllib
import urllib2
import threading
import cv2


def get_image_info(full_path):
    fd = open(full_path, 'r')
    image_list = []
    for line in fd:
        rst = line.strip().split()
        image_list.append(rst)
    return image_list


def download_image(image_info, output_path):
    image_id, url = image_info[0:2]
    useless, ext = os.path.splitext(url)
    if ext is None or ext == "":
        ext = ".jpg"
    filepath = os.path.join(output_path, image_id+ext)
    try:
        urllib.urlretrieve(url, filepath)
    except Exception:
        return None
    info = os.stat(filepath)
    if info.st_size > 10240:
        return filepath
    else:
        os.remove(filepath)
        return None


def crop_image(image_info, input_path, output_path):
    try:
        img = cv2.imread(input_path)
    except Exception:
        print("Failed to read %s" % input_path)
        return
    if img is None:
        print ("Read Image %s Error" % input_path)
        return
    image_id, url, left, top, right, bottom = image_info[0:6]

    crop = img[int(float(top)):int(float(bottom))+1,
               int(float(left)):int(float(right))+1]

    cv2.imwrite(os.path.join(output_path, image_id+".jpg"), crop)


def worker(image_list, original_output_path, crop_output_path):
    for image_info in image_list:
        saved_image = download_image(image_info, original_output_path)
        if saved_image is not None:
            crop_image(image_info, saved_image, crop_output_path)
        else:
            print("Download image %s failed" % image_info[1])


def main(input_path, output_path):
    """main function"""
    if not os.path.exists(input_path):
        print("%s does not exist" % input_path)
        return -1
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_file_list = os.listdir(input_path)
    max_thread_num = multiprocessing.cpu_count()
    for file_list in all_file_list:
        print("Processing %s file." % file_list)
        file_name, file_ext = os.path.splitext(file_list)
        file_list = os.path.join(input_path, file_list)
        if file_ext == ".txt":
            image_file_list = get_image_info(file_list)
        else:
            print("Unknown file: %s" % file_list)
            continue
        if not os.path.exists(os.path.join(output_path, file_name+"_original")):
            os.makedirs(os.path.join(output_path, file_name+"_original"))
        if not os.path.exists(os.path.join(output_path, file_name+"_crop")):
            os.makedirs(os.path.join(output_path, file_name+"_crop"))

        threads = []
        last_list_idx = 0
        for i in range(max_thread_num):
            t = threading.Thread(name=("worker_%d" % i),
                                 target=worker, args=(image_file_list[last_list_idx:int((i+1)*len(image_file_list)/max_thread_num)],
                                                      os.path.join(output_path, file_name+"_original"),
                                                      os.path.join(output_path, file_name+"_crop")))
            last_list_idx = int((i+1)*len(image_file_list)/max_thread_num)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='downloadImage.py is used to generate train and val file list')

    parser.add_argument('-i', "--input", action="store", help="Specify the input folder of the download list files",
                        type=str, required=True, dest="input_path")
    parser.add_argument('-o', "--out", action="store", help="Specify the output folder of the images",
                        type=str, required=True, dest="output_path")
    results = parser.parse_args()
    main(results.input_path, results.output_path)
