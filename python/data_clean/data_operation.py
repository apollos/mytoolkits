import argparse
import sys
import logging
import mylogs
import dataclean

reload(sys)
sys.setdefaultencoding('utf-8')

FLAGS = None
csv_file_ext = "csv"

logLevel = logging.DEBUG
recordLogs = mylogs.myLogs(logLevel)


def show_table_info(table_df):
    for dict_key in table_df.keys():
        if dict_key != "df" and table_df[dict_key] is not None and len(table_df[dict_key]) > 0:
            print("%s:" % dict_key)
            print(table_df[dict_key])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Set up the pre-trained graph.

    dataclean_handle = dataclean.dataclean(file_ext=FLAGS.file_ext.split(","), header_flag=FLAGS.header)
    table_list = dataclean_handle.load_datafile(FLAGS.input_files)
    if FLAGS.action == 'show':
        keys = table_list.keys()
        for key in keys:
            print ("***********************Table %s Information:**************************" % key)
            show_table_info(table_list[key])
    elif FLAGS.action == 'join':
        if FLAGS.join_key is None:
            recordLogs.logger.error("join key is not available")
            return
        out_df = dataclean_handle.join_tables(table_list, list(set(FLAGS.join_key.split(','))))
        print out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_files',
        type=str,
        default='',
        required=True,
        help='''Path to folders of csv or excel files, it can be single file, a folder include multiple files or a list \
             of multiple files'''
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['show', 'join', 'fill', 'numeral', 'delete', 'expand'],
        help="""\
      Actions for CVs:\n
      show: Display the head of all tables, Missing data columns, Type of columns.\n
      join: Join multiple CVs to one dataframe\n
      fill: Fill the missing data\n
      numera: Numeralization the string or chart columns
      """
    )
    parser.add_argument(
        '--header',
        type=str2bool,
        default="Yes",
        help="""Specify whether the csv file(s) including head or not.\
        Y or N"""
    )
    parser.add_argument(
        '--file_ext',
        type=str,
        default='csv, txt',
        help='File extension name. Support csv, txt, xls, xlsx'
    )
    parser.add_argument(
        '--join_key',
        type=str,
        help='Column name(s) as the join key. It can be a single value or a list which separated by comma'
    )
    parser.add_argument(
        '--output_files',
        type=str,
        default='merge_table.csv',
        help='Path to folders of output csv or excel files'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
