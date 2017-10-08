import pandas as pd
import os
import fnmatch
import mylogs
import collections
import numpy as np

default_file_ext = ["csv", "txt"]
excel_file_ext = ['xls', 'xlsx']
STATS_CLASS_NUM = 10


class DataClean:

    def __init__(self, file_ext=default_file_ext, header_flag=False):
        self.recordLogs = mylogs.myLogs()
        self.file_ext = file_ext
        self.header_flag = header_flag

    def read_content(self, filepath, object_column_names):
        file_df = None
        dtype_dict = {}
        if object_column_names is not None and len(object_column_names) > 0:
            for column_name in object_column_names:
                dtype_dict[column_name] = object
        if not os.path.exists(filepath):
            self.recordLogs.logger.error("%s does not exist" % filepath)
        if "csv" in self.file_ext or "txt" in self.file_ext:
            if not self.header_flag:
                file_df = pd.read_csv(filepath, header=None)
            else:
                if len(dtype_dict) > 0:
                    file_df = pd.read_csv(filepath, dtype=dtype_dict)
                else:
                    file_df = pd.read_csv(filepath)
        elif "xls" in self.file_ext or "xlsx" in self.file_ext:
            if not self.header_flag:
                file_df = pd.read_excel(filepath, header=None)
            else:
                if len(dtype_dict) > 0:
                    file_df = pd.read_excel(filepath, dtype=dtype_dict)
                else:
                    file_df = pd.read_excel(filepath, dtype=dtype_dict)
        return file_df

    def load_datafile(self, input_path, object_column_names, target_column_names):
        table_content = collections.defaultdict(dict)
        file_list = []
        if not os.path.exists(input_path):
            self.recordLogs.logger.error("%s does not exist" % input_path)
            return None
        else:
            if os.path.isfile(input_path):
                filename, ext_file = os.path.splitext(input_path)
                if ext_file.replace('.', '') in self.file_ext:
                    file_list.append(input_path)
                else:
                    self.recordLogs.logger.error("%s is not supported file" % input_path)
                    return None
            else:
                for root, dir_names, filenames in os.walk(input_path):
                    for file_ext in self.file_ext:
                        for filename in fnmatch.filter(filenames, '*.' + file_ext):
                            file_list.append(os.path.join(unicode(root, 'utf8'), unicode(filename, 'utf8')))
        self.recordLogs.logger.info("Total File number: %d" % len(file_list))
        for file_path in file_list:
            table_content[os.path.basename(file_path)]["df"] = self.read_content(file_path, object_column_names)
        self.check_df_dict(table_content, target_column_names)
        return table_content

    def save_datafile(self, dataframe, output_path):
        dirname = os.path.dirname(output_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if dirname != '' and not os.path.exists(dirname):
            try:
                os.mkdir(dirname)
            except OSError as e:
                self.recordLogs.logger.error("Create Path %s failed. Error Code %d" % (dirname, e.errno))
                return
        filename, extfile = os.path.splitext(output_path)
        extfile = extfile.replace('.', '')
        if extfile in default_file_ext:
            dataframe.to_csv(output_path, encoding='utf-8', index=False)
        elif extfile in excel_file_ext:
            dataframe.to_excel(output_path, encoding='utf-8', index=False)
        else:
            self.recordLogs.logger.error("Unrecognized file extension %s" % output_path)

    def save_dictfile(self, dict_data, output_path):
        dataframe = pd.DataFrame([dict_data])
        self.save_datafile(dataframe, output_path)

    def check_df(self, df_info, target_column_names):
        if df_info is None:
            self.recordLogs.logger.error("Empty or invalid data frame")

        '''check missing value'''
        missing_data_columns = []
        same_data_columns = []
        target_columns_stat_info = collections.defaultdict(dict)
        df_info["heads"] = []
        if self.header_flag:
            column_heads = list(df_info["df"].columns.values)
        else:
            column_heads = ["c%04d" % x for x in range(len(df_info["df"].columns))]
            '''force add a head for future use'''
            df_info["df"].columns = column_heads
        for column_name in column_heads:
            missing_flag = False
            if any(pd.isnull(df_info["df"][column_name])):
                missing_data_columns.append(column_name)
                missing_flag = True
            column_values = set(df_info["df"][column_name])
            if len(column_values) == 1 or (missing_flag and len(column_values) == 2):
                same_data_columns.append(column_name)
            value_list = df_info["df"][column_name].tolist()
            if column_name in target_column_names:
                if df_info["df"][column_name].dtype == 'O':
                    '''Calculate the ratio of each value in the whole list of the column'''
                    for column_value in column_values:
                        target_columns_stat_info[column_name][str(column_value)] = \
                            float(value_list.count(column_value))/len(value_list)
                elif df_info["df"][column_name].dtype == 'float64' or df_info["df"][column_name].dtype == 'int64':
                    if len(column_values) <= STATS_CLASS_NUM:
                        '''Calculate the ratio of each value in the whole list of the column'''
                        ratio_count = 0.0
                        for column_value in column_values:
                            target_columns_stat_info[column_name][str(column_value)] = \
                                float(value_list.count(column_value))/len(value_list)
                            ratio_count += target_columns_stat_info[column_name][str(column_value)]
                        if missing_flag:
                            target_columns_stat_info[column_name]['nan'] = 1 - ratio_count
                        else:
                            if ratio_count != 1:
                                self.recordLogs.logger.warning("Column [%s] Value Ratio is [%f]" % (column_name, ratio_count))
                    else:
                        self.recordLogs.logger.info("Value Number [%d] of Target Column [%s] exceed Threshold [%d], " +
                                                    "Calculate Stats Info"
                                                    % (len(column_values), column_name, STATS_CLASS_NUM))
                        target_columns_stat_info[column_name]['min'] = np.min(value_list)
                        target_columns_stat_info[column_name]['max'] = np.max(value_list)
                        target_columns_stat_info[column_name]['mean'] = np.mean(value_list)
                        target_columns_stat_info[column_name]['median'] = np.median(value_list)
                else:
                    self.recordLogs.logger.warn("Unknown type [%s] of Target Column [%s]"
                                                % (df_info["df"][column_name].dtype, column_name))
            df_info["heads"].append((column_name, df_info["df"][column_name].dtype))

        '''fill missing data columns'''
        df_info["missing"] = missing_data_columns
        '''fill same data columns'''
        df_info["sameData"] = same_data_columns
        '''fill shape'''
        df_info["shape"] = df_info["df"].shape
        '''fill statistic information'''
        df_info["stats"] = target_columns_stat_info
        '''fill different data type number'''
        dtypes_obj = df_info["df"].columns.to_series().groupby(df_info["df"].dtypes).groups
        dtype_keys = dtypes_obj.keys()
        df_info["dtypes"] = []
        for key in dtype_keys:
            df_info["dtypes"].append((key, len(dtypes_obj[key])))

    def check_df_dict(self, df_dict, target_column_names):
        if df_dict is None or len(df_dict.keys()) == 0:
            self.recordLogs.logger.error("Empty or invalid data frame Dict")

        for table_name in df_dict.keys():
            if df_dict[table_name]["df"].shape[0] == 0:
                del df_dict[table_name]
                continue
            self.check_df(df_dict[table_name], target_column_names)

    def merge_rows_same_key(self, df_info, different_columns, key_column_name):
        if df_info is None:
            self.recordLogs.logger.error("Wrong Table information")
            return None
        if different_columns is None or len(different_columns) == 0:
            self.recordLogs.logger.warn("different_columns is empty")
            return None
        new_df_info = df_info
        obj_df_columns = list(df_info[different_columns].select_dtypes(include=['object']).columns.values)
        d1 = pd.DataFrame()
        new_columns = []
        for column_name in obj_df_columns:
            self.recordLogs.logger.info("Automatically Change column %s to multiple columns" % column_name)
            if any(pd.isnull(df_info[column_name])):
                self.recordLogs.logger.error("There is NaN in column %s. Please first fill it and then merge rows" %
                                             column_name)
                return None
            d1 = df_info[column_name].apply(lambda x: column_name+'_'+'|'.join(pd.Series(x))).str.get_dummies()

            new_df_info = pd.concat([new_df_info, d1], axis=1)
            new_columns.extend(list(d1.columns.values))

            '''remove the old column'''
            new_df_info.drop(column_name, axis=1, inplace=True)
        '''must first deal with object type and then deal with int'''
        non_obj_df_columns = list(df_info[different_columns].select_dtypes(exclude=['object']).columns.values)
        '''shall we caculate the mean and update to one and remove others????'''
        '''Currently, it can not auotmatically process, need do it by specified method'''
        if len(non_obj_df_columns) > 0:
            self.recordLogs.logger.warn("Can not automatically expand the columns %s" % non_obj_df_columns)
            return None
        '''Start merge same key value rows'''
        '''
        row_keys = list(set(df_info[key_column_name]))
        for row_key in row_keys:
            df_info.loc[df_info[key_column_name] == row_key].apply(self.merge_row)
        '''
        '''generate agg_dict'''

        agg_dict = {}

        for column_name in list(new_columns):
            agg_dict[column_name] = max

        agg_obj_df_info = new_df_info.groupby([key_column_name]).agg(agg_dict)
        new_df_info.drop_duplicates([key_column_name], keep='first', inplace=True)

        for row in agg_obj_df_info.itertuples():
            """itertuples return result can not be used as dataframe
            we have to use index to seek it
            since use dict while groupby, we can also directly use idx from 0 to n
            we must search the location in agg_obj_df_info and then + 1"""

            for column_name in new_columns:
                new_df_info.set_value(new_df_info[key_column_name] == row[0], column_name,
                                      row[agg_obj_df_info.columns.get_loc(column_name)+1])
        return new_df_info

    def merge_rows_same_key_complex(self, df_info, key_column_name, columnA, columnB):
        if df_info is None:
            self.recordLogs.logger.error("Wrong Table information")
            return None
        if columnA is None or columnB is None:
            self.recordLogs.logger.warn("columnA or columnB is empty")
            return None
        new_df_info = df_info
        if df_info[columnA].dtypes == 'object':
            d1 = pd.DataFrame()
            self.recordLogs.logger.info("Expand column %s to multiple columns" % columnA)
            if any(pd.isnull(df_info[columnA])):
                self.recordLogs.logger.error("There is NaN in column %s. Please first fill it and then merge rows" %
                                             columnA)
                return None
            if any(pd.isnull(df_info[columnB])):
                self.recordLogs.logger.error("There is NaN in column %s. Please first fill it and then merge rows" %
                                             columnB)
                return None
            d1 = df_info[columnA].apply(lambda x: columnA+'_'+columnB+'_'+'|'.join(pd.Series(x))).str.get_dummies()
            '''d1 = d1.mul(df_info[columnB].values).where(d1.astype(bool))'''
            for d1_column in list(d1.columns.values):
                d1[d1_column] = d1[d1_column] * df_info[columnB].values
            new_df_info = pd.concat([new_df_info, d1], axis=1)
            '''remove the old column'''
            new_df_info.drop(columnA, axis=1, inplace=True)
            new_df_info.drop(columnB, axis=1, inplace=True)
        else:
            self.recordLogs.logger.warning("Expand column %s type is %s which is not good for expand" %
                                           (columnA, df_info[columnA].dtypes))
            return None
        '''Start merge same key value rows'''
        '''generate agg_dict'''
        agg_dict = {}
        new_columns = list(d1.columns.values)

        for column_name in list(new_columns):
            if new_df_info[column_name].dtype == "object":
                self.recordLogs.logger.warning("Merge Action may be not right in %s" % column_name)
                agg_dict[column_name] = max
            else:
                agg_dict[column_name] = sum
        agg_obj_df_info = new_df_info.groupby([key_column_name]).agg(agg_dict)
        new_df_info.drop_duplicates([key_column_name], keep='first', inplace=True)

        for row in agg_obj_df_info.itertuples():
            for column_name in new_columns:
                new_df_info.set_value(new_df_info[key_column_name] == row[0], column_name,
                                      row[agg_obj_df_info.columns.get_loc(column_name)+1])

        return new_df_info

    @staticmethod
    def check_redundant_row_by_column(table, key):
        different_columns_redundant_key = []
        columns = list(table.select_dtypes(include=['object']).columns.values)
        '''
        if len(index_set) != len(table[key]):
            for value in index_set:
                rst = table.loc[table[key] == value]
                for column_name in columns:
                    if column_name != key:
                        if len(set(rst[column_name])) != 1:
                            different_columns_redundant_key.append(column_name)
        return list(set(different_columns_redundant_key))
        '''
        for column_name in columns:
            if column_name != key:
                if len(set(table[column_name])) != 1:
                    different_columns_redundant_key.append(column_name)
        return list(set(different_columns_redundant_key))

    @staticmethod
    def seek_join_key(table, keys):
        column_names = list(table.columns.values)
        hit_column_name = None
        for key in keys:
            if key in column_names:
                hit_column_name = key
                break
        return hit_column_name

    def get_left_right_join_keys(self, left_table, right_table, keys):
        left_key = self.seek_join_key(left_table, keys)
        right_key = self.seek_join_key(right_table, keys)

        return left_key, right_key

    def simple_expand_column(self, df_info, key_column, table_name):
        """ check redundant"""
        if df_info is None or key_column is None:
            self.recordLogs.logger.error("Wrong Data Frame information or key column in Table %s" % table_name)
            return None
        key_column_name = self.seek_join_key(df_info, key_column)
        if key_column_name is None or len(key_column_name) == 0:
            self.recordLogs.logger.error("Wrong join_key %s for Table %s" % (key_column, table_name))
            return None
        redundant_columns = self.check_redundant_row_by_column(df_info, key_column_name)
        if redundant_columns is not None and len(redundant_columns) > 0:
            df_info = self.merge_rows_same_key(df_info, redundant_columns, key_column_name)
        return df_info

    def expand_column_for_tabs(self, df_dict, join_key):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if join_key is None or len(join_key) == 0:
            self.recordLogs.logger.error("Wrong join_key input")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            df_info = self.simple_expand_column(df_dict[table_name]["df"], join_key, table_name)
            if df_info is None:
                return None
            df_dict[table_name]["df"] = df_info
        return df_dict

    def expand_column_complex(self, df_dict, join_key, columnA, columnB):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if join_key is None or len(join_key) == 0:
            self.recordLogs.logger.error("Wrong join_key input")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            """ check redundant"""
            df_info = df_dict[table_name]["df"]
            if df_info is None:
                self.recordLogs.logger.error("Wrong Data Frame information in Table %s" % table_name)
                return None
            key_column_name = self.seek_join_key(df_info, join_key)
            if key_column_name is None:
                self.recordLogs.logger.error("Wrong join_key %s for Table %s" % (join_key, table_name))
                return None
            columns = list(df_info.columns.values)
            if columnA not in columns or columnB not in columns:
                self.recordLogs.logger.error("Column A %s or Column B %s is not in Table %s" %
                                             (columnA, columnB, table_name))
                return None
            df_info = self.merge_rows_same_key_complex(df_info, key_column_name, columnA, columnB)
            if df_info is None:
                return None
            df_dict[table_name]["df"] = df_info
        return df_dict

    def join_tables(self, df_dict, join_key):

        if df_dict is None or len(df_dict.keys()) <= 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if join_key is None or len(join_key) == 0:
            self.recordLogs.logger.error("Wrong join_key input")
            return None
        table_names = df_dict.keys()
        rst_table = None
        for table_name in table_names:
            df_info = self.simple_expand_column(df_dict[table_name]["df"], join_key, table_name)
            if df_info is None:
                return None
            if rst_table is None:
                rst_table = df_info
                continue
            left_on_key, right_on_key = self.get_left_right_join_keys(rst_table, df_info, join_key)
            try:
                if left_on_key is None or right_on_key is None:
                    self.recordLogs.logger.error("Can not find appropriate join keys in tables. Keys %s" % join_key)
                    return None
                if left_on_key != right_on_key:
                    df_info.rename(columns={right_on_key:left_on_key}, inplace=True)
                rst_table = pd.merge(rst_table, df_info, how="outer", on=left_on_key)
            except KeyError:
                self.recordLogs.logger.error("Merge Table %s and %s KeyError. Joint Key: %s" % join_key)
            return rst_table

    def fill_column_tables(self, df_dict, column_name, filled_value):

        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_name is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        if filled_value is None:
            self.recordLogs.logger.error("Wrong Filled Value")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            columns = list(df_dict[table_name]["df"].columns.values)
            if column_name not in columns:
                self.recordLogs.logger.warning("Column %s does not exist in table %s" % (column_name, table_name))
                continue
            df_dict[table_name]["df"][column_name].replace(np.NaN, filled_value, inplace=True)
        return df_dict

    def fill_column_tables_by_calculate(self, df_dict, joint_key, column_name):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_name is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        if joint_key is None:
            self.recordLogs.logger.error("Wrong Filled joint_key")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            columns = list(df_dict[table_name]["df"].columns.values)
            if column_name not in columns:
                self.recordLogs.logger.warning("Column %s does not exist in table %s" % (column_name, table_name))
                continue
            if joint_key not in columns:
                self.recordLogs.logger.error("Joint Key Column %s does not exist in table %s" % (column_name, table_name))
                return
            key_value_list = df_dict[table_name]["df"][joint_key].unique()
            for key_value in key_value_list:
                column_value_list = \
                    df_dict[table_name]["df"].loc[df_dict[table_name]["df"][joint_key] == key_value][column_name].tolist()
                if df_dict[table_name]["df"][column_name].dtype == 'O':
                    filled_value = collections.Counter(column_value_list).most_common(1)[0][0]
                else:
                    filled_value = np.mean(column_value_list)
                index_list = df_dict[table_name]["df"].loc[df_dict[table_name]["df"][joint_key] == key_value].index.tolist()
                for index in index_list:
                    df_dict[table_name]["df"].at[index, column_name] = filled_value

        return df_dict

    def factorize_column_tables(self, df_dict, column_name):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_name is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            columns = list(df_dict[table_name]["df"].columns.values)
            if column_name not in columns:
                self.recordLogs.logger.warning("Column %s does not exist in table %s" % (column_name, table_name))
                continue
            values = df_dict[table_name]["df"][column_name].drop_duplicates().values
            b = [x for x in df_dict[table_name]["df"][column_name].drop_duplicates().rank(method='dense')]
            column_dict = collections.defaultdict()
            column_dict[column_name] = dict(zip(b, values))
            df_dict[table_name]["factor"] = column_dict
            df_dict[table_name]["df"][column_name] = df_dict[table_name]["df"][column_name].rank(method='dense')

        return df_dict

    def delete_column_tables(self, df_dict, column_name):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_name is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            columns = list(df_dict[table_name]["df"].columns.values)
            if column_name not in columns:
                self.recordLogs.logger.warning("Column %s does not exist in table %s" % (column_name, table_name))
                continue
            df_dict[table_name]["df"].drop(column_name, axis=1, inplace=True)

        return df_dict

    def separate_tab_by_column(self, df_dict, column_name):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_name is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        table_names = df_dict.keys()
        new_df_dict = collections.defaultdict(dict)
        for table_name in table_names:
            columns = list(df_dict[table_name]["df"].columns.values)
            if len(set(column_name) & set(columns)) == 0:
                self.recordLogs.logger.warning("Column %s does not exist in table %s" % (column_name, table_name))
                continue
            """We can only pick up one column as the groupby key"""
            target_column_name = list(set(column_name) & set(columns))[0]
            new_df_group = df_dict[table_name]["df"].groupby(target_column_name)
            for name, group in new_df_group:
                name = name.replace('/', '-')
                new_df_dict[table_name][name] = group

        return new_df_dict

    def merge_columns(self, df_dict, column_list):
        if df_dict is None or len(df_dict.keys()) < 1:
            self.recordLogs.logger.error("Wrong Table Dict")
            return None
        if column_list is None:
            self.recordLogs.logger.error("Error Column Name")
            return None
        table_names = df_dict.keys()
        for table_name in table_names:
            for column_name in column_list:
                if df_dict[table_name]["df"][column_name].dtype != 'O':
                    self.recordLogs.logger.info("Chang Column %s type to Object" % column_name)
                    df_dict[table_name]["df"][column_name] = df_dict[table_name]["df"][column_name].astype('O')
            new_column_name = "_".join(column_list)
            df_dict[table_name]["df"][new_column_name] = \
                df_dict[table_name]["df"][column_list].apply(lambda x: '_'.join(x), axis=1)

        return df_dict