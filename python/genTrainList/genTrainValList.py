import os
import argparse
import random


def separate_file_list(full_file_list):
    if (full_file_list is not None) and len(full_file_list) > 0:
        random.shuffle(full_file_list)
        return full_file_list[:int(len(full_file_list)*0.8)], full_file_list[int(len(full_file_list)*0.8):]
    return [], []


def main(input_path, output_path):
    """main function"""
    if not os.path.exists(input_path):
        print("%s does not exist" % input_path)
        return -1
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    class_path_set = os.listdir(input_path)
    output_file_train = open("%s/train.txt" % output_path, 'w')
    output_file_val = open("%s/val.txt" % output_path, 'w')
    for class_path in class_path_set:
        class_path = os.path.join(input_path, class_path)
        if os.path.isdir(class_path):
            full_file_list = []
            file_set = os.listdir(class_path)
            for file_lst in file_set:
                file_name, file_ext = os.path.splitext(file_lst)
                if file_ext in (".jpg", ".JPEG", ".png"):
                    full_file_list.append(os.path.join(class_path, file_lst))
                else:
                    print("Unknown file: %s" % (os.path.join(class_path, file_lst)))
                    continue

            train_list, val_list = separate_file_list(full_file_list)
            for file_id in train_list:
                output_file_train.write(file_id+"\n")
            for file_id in val_list:
                output_file_val.write(file_id+"\n")
    output_file_train.close()
    output_file_val.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='genTrainValList.py is used to generate train and val file list')

    parser.add_argument('-i', "--input", action="store", help="Specify the input folder of the data",
                        type=str, required=True, dest="input_path")
    parser.add_argument('-o', "--out", action="store", help="Specify the output folder of the tran and val list",
                        type=str, required=True, dest="output_path")
    results = parser.parse_args()
    main(results.input_path, results.output_path)
