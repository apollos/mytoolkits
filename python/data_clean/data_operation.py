import argparse
import logging
import mylogs
import dataclean
import os


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

    dataclean_handle = dataclean.DataClean(file_ext=FLAGS.file_ext.split(","), header_flag=FLAGS.header)
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
        if FLAGS.output_file is not None:
            dataclean_handle.save_datafile(out_df, FLAGS.output_file)
        else:
            print(out_df)
    elif FLAGS.action == 'fill':
        if FLAGS.target_columns is None:
            recordLogs.logger.error("Target column name is not filled for fill action")
            return
        if FLAGS.target_values is None:
            recordLogs.logger.error("Value to be filled is None for fill action")
            return
        values = FLAGS.target_values.split(',')
        columns = FLAGS.target_columns.split(',')
        if len(values) != len(columns):
            recordLogs.logger.error("Target column number is not same as targt value number")
            return
        for idx in range(len(values)):
            table_list = dataclean_handle.fill_column_tables(table_list, columns[idx], values[idx])

        if table_list is not None:
            dirname = './'
            if FLAGS.output_file is not None:
                if os.path.isfile(FLAGS.output_file):
                    dirname = os.path.dirname(unicode(FLAGS.output_file, 'utf-8'))
                elif os.path.isdir(FLAGS.output_file):
                    dirname = unicode(FLAGS.output_file, 'utf-8')
            table_names = table_list.keys()
            for table_name in table_names:
                full_path = dirname+"/"+"fill_"+table_name
                dataclean_handle.save_datafile(table_list[table_name]["df"], full_path)
    elif FLAGS.action == 'factor':
        if FLAGS.target_columns is None:
            recordLogs.logger.error("Target column name is not filled for fill action")
            return
        columns = FLAGS.target_columns.split(',')

        for idx in range(len(columns)):
            table_list = dataclean_handle.factorize_column_tables(table_list, columns[idx])

        if table_list is not None:
            dirname = './'
            if FLAGS.output_file is not None:
                if os.path.isfile(FLAGS.output_file):
                    dirname = os.path.dirname(unicode(FLAGS.output_file, 'utf-8'))
                elif os.path.isdir(FLAGS.output_file):
                    dirname = unicode(FLAGS.output_file, 'utf-8')
            table_names = table_list.keys()
            for table_name in table_names:
                full_path = dirname+"/"+"factor_"+table_name
                dataclean_handle.save_datafile(table_list[table_name]["df"], full_path)
                full_path = dirname+"/"+"factor_dict_"+table_name
                dataclean_handle.save_dictfile(table_list[table_name]["factor"], full_path)
    elif FLAGS.action == 'delete':
        if FLAGS.target_columns is None:
            recordLogs.logger.error("Target column name is not filled for fill action")
            return
        columns = FLAGS.target_columns.split(',')

        for idx in range(len(columns)):
            table_list = dataclean_handle.delete_column_tables(table_list, columns[idx])

        if table_list is not None:
            dirname = './'
            if FLAGS.output_file is not None:
                if os.path.isfile(FLAGS.output_file):
                    dirname = os.path.dirname(unicode(FLAGS.output_file, 'utf-8'))
                elif os.path.isdir(FLAGS.output_file):
                    dirname = unicode(FLAGS.output_file, 'utf-8')
            table_names = table_list.keys()
            for table_name in table_names:
                full_path = dirname+"/"+"delete_"+table_name
                dataclean_handle.save_datafile(table_list[table_name]["df"], full_path)
    elif FLAGS.action == 'simple_expand':
        if FLAGS.join_key is None:
            recordLogs.logger.error("join_key column name is not filled for simple_expand action")
            return

        columns = FLAGS.join_key.split(',')

        table_list = dataclean_handle.expand_column_for_tabs(table_list, columns)
        if table_list is not None:
            dirname = './'
            if FLAGS.output_file is not None:
                if os.path.isfile(FLAGS.output_file):
                    dirname = os.path.dirname(unicode(FLAGS.output_file, 'utf-8'))
                elif os.path.isdir(FLAGS.output_file):
                    dirname = unicode(FLAGS.output_file, 'utf-8')
            table_names = table_list.keys()
            for table_name in table_names:
                full_path = dirname+"/"+"simple_expand_"+table_name
                dataclean_handle.save_datafile(table_list[table_name]["df"], full_path)
    elif FLAGS.action == 'expand_columnA_by_columnB':
        if FLAGS.join_key is None:
            recordLogs.logger.error("join_key column name is not filled for simple_expand action")
            return
        if FLAGS.expand_column_a_b is None:
            recordLogs.logger.error("expand_column_a_b is not filled for simple_expand action")
            return
        try:
            column_a, column_b = FLAGS.expand_column_a_b.split(',')
        except ValueError:
            recordLogs.logger.error("expand_column_a_b can only be 'column A, column B' format")
            return

        key_columns = FLAGS.join_key.split(',')
        table_list = dataclean_handle.expand_column_complex(table_list, key_columns, column_a, column_b)
        if table_list is not None:
            dirname = './'
            if FLAGS.output_file is not None:
                if os.path.isfile(FLAGS.output_file):
                    dirname = os.path.dirname(unicode(FLAGS.output_file, 'utf-8'))
                elif os.path.isdir(FLAGS.output_file):
                    dirname = unicode(FLAGS.output_file, 'utf-8')
            table_names = table_list.keys()
            for table_name in table_names:
                full_path = dirname+"/"+"simple_expand_"+table_name
                dataclean_handle.save_datafile(table_list[table_name]["df"], full_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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
        choices=['show', 'join', 'fill', 'factor', 'delete', 'simple_expand', 'expand_columnA_by_columnB',
                 'separate_tab_by_column'],
        help="""Actions for CVs:
      show: Display the head of all tables, Missing data columns, Type of columns.
      join: Join multiple CVs to one dataframe
      fill: Fill the missing data
      factor: factorize the string values to a factor list
      delete: delete specified columns from the tables
      simple_expand: automatically expand one columns to several according to the potential value to a dummy matrix
      expand_columnA_by_columnB: Expand one column A to several columns according to the specified column B values
      separate_tab_by_column: Generate new csv files by a column content
      """
    )
    parser.add_argument(
        '--header',
        type=str2bool,
        default="Yes",
        help="""Specify whether the csv file(s) including head or not. \
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
        '--output_file',
        type=str,
        help='Path to folders of output csv or excel files'
    )
    parser.add_argument(
        '--target_values',
        type=str,
        help='Fill in value to specified column, it can be a list'
    )
    parser.add_argument(
        '--target_columns',
        type=str,
        help='Column name to be filled/factor/delete, it can be a list'
    )
    parser.add_argument(
        '--expand_column_a_b',
        type=str,
        help='columnA, columnB for action expand_columnA_by_columnB'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
