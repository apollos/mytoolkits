import os
import argparse
import random
import shutil


def separate_file_list(full_file_list):
    if (full_file_list is not None) and len(full_file_list) > 0:
        random.shuffle(full_file_list)
        return full_file_list[:int(len(full_file_list)*0.75)], full_file_list[int(len(full_file_list)*0.75):int(len(full_file_list)*0.95)],full_file_list[int(len(full_file_list)*0.95):]
    return [], [], []


def copyfile(source_file, output_path, type_name, class_name):
    target_path = os.path.join(output_path, type_name, class_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    shutil.copy2(source_file, target_path)


def write_list2file(fd, data_list, copy_flag, output_path, type_name, label_flag, label, class_path):
    for file_id in data_list:
        if label_flag:
            fd.write(file_id+" %d\n" % label)
        else:
            fd.write(file_id+"\n")
        if copy_flag:
            copyfile(file_id, output_path, type_name, class_path)


def main(input_path, output_path, copy_str_flag, label_flag):
    """main function"""
    if not os.path.exists(input_path):
        print("%s does not exist" % input_path)
        return -1
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if copy_str_flag.lower() == 'true':
        copy_flag = True
    else:
        copy_flag = False
    if label_flag.lower() == 'true':
        label_flag = True
    else:
        label_flag = False

    class_path_set = os.listdir(input_path)
    output_file_train = open("%s/train.txt" % output_path, 'w')
    output_file_val = open("%s/val.txt" % output_path, 'w')
    output_file_test = open("%s/test.txt" % output_path, 'w')
    synsets_dir_list = []
    for class_path in class_path_set:
        full_class_path = os.path.join(input_path, class_path)
        if os.path.isdir(full_class_path):
            synsets_dir_list.append(class_path)
            full_file_list = []
            file_set = os.listdir(full_class_path)
            for file_lst in file_set:
                file_name, file_ext = os.path.splitext(file_lst)
                if file_ext in (".jpg", ".JPEG", ".png", ".JPG"):
                    full_file_list.append(os.path.join(full_class_path, file_lst))
                else:
                    print("Unknown file: %s" % (os.path.join(full_class_path, file_lst)))
                    continue

            train_list, val_list, test_list = separate_file_list(full_file_list)
            write_list2file(output_file_train, train_list, copy_flag, output_path, "train", label_flag, len(synsets_dir_list)-1,
                            class_path)
            write_list2file(output_file_val, val_list, copy_flag, output_path, "val", label_flag, len(synsets_dir_list)-1,
                            class_path)
            write_list2file(output_file_test, test_list, copy_flag, output_path, "test", label_flag, len(synsets_dir_list)-1,
                            class_path)

    if label_flag:
        output_file_synsets = open("%s/synsets.txt" % output_path, 'w')
        output_file_synsets.write('\n'.join(class_path_set))
        output_file_synsets.close()
    output_file_train.close()
    output_file_val.close()
    output_file_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='genTrainValList.py is used to generate train and val file list')

    parser.add_argument('-i', "--input", action="store", help="Specify the input folder of the data",
                        type=str, required=True, dest="input_path")
    parser.add_argument('-o', "--out", action="store", help="Specify the output folder of the tran and val list",
                        type=str, required=True, dest="output_path")
    parser.add_argument('-c', "--copy", action="store", help="Specify whether copy the source data to output folder",
                        type=str, required=True, choices=['True', 'False'],  dest="copy_flag")
    parser.add_argument('-l', "--label", action="store", help="Specify whether add label after each data",
                        type=str, required=True, choices=['True', 'False'], dest="label_flag")
    results = parser.parse_args()
    main(results.input_path, results.output_path, results.copy_flag, results.label_flag)
