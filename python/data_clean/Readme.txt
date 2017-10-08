python data_operation.py --help
usage: data_operation.py [-h] --input_files INPUT_FILES
                         [--action {show,join,fill,factor,delete,simple_expand,expand_columnA_by_columnB,separate_tab_by_column,merge}]
                         [--header HEADER] [--file_ext FILE_EXT]
                         [--join_key JOIN_KEY] [--output_file OUTPUT_FILE]
                         [--target_values TARGET_VALUES]
                         [--target_columns TARGET_COLUMNS]
                         [--expand_column_a_b EXPAND_COLUMN_A_B]
                         [--object_columns OBJECT_COLUMNS]

optional arguments:
  -h, --help            show this help message and exit
  --input_files INPUT_FILES
                        Path to folders of csv or excel files, it can be single file, a folder include multiple files or a list of multiple files
  --action {show,join,fill,factor,delete,simple_expand,expand_columnA_by_columnB,separate_tab_by_column,merge}
                        Actions for CVs:
                              show: Display the head of all tables, Missing data columns, Type of columns.
                              join: Join multiple CVs to one dataframe
                              fill: Fill the missing data
                              factor: factorize the string values to a factor list
                              delete: delete specified columns from the tables
                              simple_expand: automatically expand one columns to several according to the potential value to a dummy matrix
                              expand_columnA_by_columnB: Expand one column A to several columns according to the specified column B values
                              separate_tab_by_column: Generate new csv files by a column content

  --header HEADER       Specify whether the csv file(s) including head or not. Y or N
  --file_ext FILE_EXT   File extension name. Support csv, txt, xls, xlsx
  --join_key JOIN_KEY   Column name(s) as the join key. It can be a single value or a list which separated by comma
  --output_file OUTPUT_FILE
                        Path to folders of output csv or excel files
  --target_values TARGET_VALUES
                        Fill in value to specified column, it can be a list
  --target_columns TARGET_COLUMNS
                        Column name to be filled/factor/delete, it can be a list
  --expand_column_a_b EXPAND_COLUMN_A_B
                        columnA, columnB for action expand_columnA_by_columnB
  --object_columns OBJECT_COLUMNS
                        Specify the column names of object type for load csv or xls file