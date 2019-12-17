import time
import numpy as np
import pandas as pd
from darwinutils.io import get_weight_file_path
from darwinutils.log import get_task_logger
from dataprocess.preprocess.data_preprocess import DataPreprocess
import os
from functools import reduce
from functools import lru_cache
import json
from darwinutils.redis_help import get_data

logger = get_task_logger(__name__)
error_reason = ""

cold_model = None
cold_labels = None
cold_column_idx_map = None


def construct_last_rst(*args):
    last_rst = ()
    for arg_data in args:
        if arg_data is None:
            last_rst += ([],)
        else:
            last_rst += (arg_data,)
    return last_rst


def predict_warm_single(pre_data, model_id='None', weight_file_id='weight', pre_process_flag=True,
                        column_name_file="column_name_lst.json",
                        data_auto_preprocess_record_file='record_all.json', timeout_log_threshold=300):
    global error_reason
    error_reason = ""
    model = load_model(model_id=model_id, weight_file_id=weight_file_id)
    labels = load_label(model_id=model_id, weight_file_id=weight_file_id)
    column_name_lst = None
    if column_name_file is not None:
        column_name_json = load_json_file(column_name_file, model_id=model_id, weight_file_id=weight_file_id)
        if column_name_json is not None and isinstance(column_name_json, dict) \
                and column_name_json.get("feature_names") is not None and isinstance(
                column_name_json["feature_names"], list):
            column_name_lst = column_name_json["feature_names"]
        if column_name_lst is None:
            error_reason = "column name file {} content is not right.".format(column_name_file)
            last_rst = construct_last_rst(None, error_reason)
            raise StopIteration
        # build column_idx_map to speed up remaping when preprocessing data
        column_idx_map = {k: idx for idx, k in enumerate(column_name_lst)}
    logger.debug('column_name_lst = {}'.format(column_name_lst))

    # load data_preprocess_record
    data_auto_preprocess_record = None
    if data_auto_preprocess_record_file is not None:
        data_auto_preprocess_record = load_json_file(data_auto_preprocess_record_file, model_id=model_id,
                                                     weight_file_id=weight_file_id)
        if not (isinstance(data_auto_preprocess_record, dict) and data_auto_preprocess_record.get(
                'all_files') is not None and isinstance(data_auto_preprocess_record['all_files'], list)):
            logger.info('data automatic preprocess record file {} content is not right'.format(
                data_auto_preprocess_record_file))
    logger.debug('data_auto_preprocess_record = {}'.format(data_auto_preprocess_record))

    for s_list in pre_data:
        logger.debug('start warm node predict')
        split_record = []
        s_list_tmp = []
        if isinstance(s_list, (tuple, list)):
            failure_flag = False
            for s_lst_data in s_list:
                if isinstance(s_lst_data["pre_data"][0], str):
                    logger.info("Get data from redis with combinations")
                    # the data save in redis, the input is key but not data
                    redis_value = get_data(s_lst_data["pre_data"])
                    # the data format is not same as pipeline input since it is pure data
                    s_lst_data["pre_data"] = redis_value
                elif isinstance(s_lst_data["pre_data"][0], list) and isinstance(s_lst_data["pre_data"][1], list) and \
                        isinstance(s_lst_data["pre_data"][1][0], str):
                    logger.info("Get data from redis with combinations for tableWithHead mode")
                    # the data save in redis, the input is key but not data
                    redis_value = get_data(s_lst_data["pre_data"][1])
                    # the data format is not same as pipeline input since it is pure data
                    s_lst_data["pre_data"] = [s_lst_data["pre_data"][0], redis_value]

                if isinstance(s_lst_data["pre_data"][0], dict):
                    # split_record = list(map(lambda s: len(s["pre_data"]), s_list))
                    # s_list = reduce(lambda x, y: x+y["pre_data"], s_list,[])
                    split_record.append(len(s_lst_data["pre_data"]))
                    s_list_tmp.extend(s_lst_data["pre_data"])
                elif isinstance(s_lst_data["pre_data"][0], list):
                    '''
                    # tableWithHead type, it is list but not dict and each data[0] is head
                    split_record = list(map(lambda s: len(s["pre_data"][1]), s_list))
                    if pre_process_flag:
                        s_list = [d["pre_data"] for d in s_list]
                    else:
                        s_list = np.array(reduce(lambda x, y: x + y["pre_data"], s_list, []))
                    #we do not do merge above since data_preprocessing will do it. note, each pre_data may have different column head so
                    # we shall not merge now but in data_preprocessing
                    '''
                    split_record.append(len(s_lst_data["pre_data"][1]))
                    if pre_process_flag:
                        s_list_tmp.append(s_lst_data["pre_data"])
                    else:
                        s_list_tmp.extend(s_lst_data["pre_data"])
                else:
                    # unspported case
                    failure_flag = True
                    error_reason = "Unsupported data format received: {}".format(type(s_lst_data))
            if failure_flag:
                last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
                yield last_rst
                continue
            s_list = s_list_tmp
            if not pre_process_flag:
                s_list = np.array(s_list)
        else:
            if isinstance(s_list["pre_data"][0], bytes):
                logger.info("Get data from redis single")
                # the data save in redis, the input is key but not data
                redis_value = get_data(s_list["pre_data"])
                # the data format is not same as pipeline input since it is pure data
                s_list["pre_data"] = redis_value
            elif isinstance(s_list["pre_data"][0], list) and isinstance(s_list["pre_data"][1], list) and \
                    isinstance(s_list["pre_data"][1][0], bytes):
                logger.info("Get data from redis with sigle for tableWithHead mode")
                # the data save in redis, the input is key but not data
                redis_value = get_data(s_list["pre_data"][1])
                # the data format is not same as pipeline input since it is pure data
                s_list["pre_data"] = [s_list["pre_data"][0], redis_value]

            s_list = np.array(s_list["pre_data"])
            if isinstance(s_list[0], list):
                if pre_process_flag:
                    s_list = [s_list]
        start_time2 = 0
        end_time2 = 0
        if pre_process_flag:
            logger.debug('start data pre-processing')
            start_time2 = time.time()
            s_list_bk = s_list
            # preprocess
            # 1: using data_auto_preprocess_record reprocess the model expected data
            # 2: reorder and reshape data into model expected
            s_list = data_preprocessing(s_list, column_idx_map, data_auto_preprocess_record)
            end_time2 = time.time()
            if s_list is None:
                if split_record is not None and len(split_record) > 0:
                    last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
                else:
                    last_rst = construct_last_rst(None, error_reason)
                yield last_rst
                continue
        if model is None:
            if split_record is not None and len(split_record) > 0:
                last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
            else:
                last_rst = construct_last_rst(None, error_reason)
            yield last_rst
            continue
        """
        try:
            s_list = np.array(s_list, dtype=np.float32)
        except ValueError:
            logger.exception("Transfer table data to np array failed")
            error_reason = "Transfer table data to np array failed"
            if split_record is not None and len(split_record) > 0:
                last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
            else:
                last_rst = construct_last_rst(None, error_reason)
            yield last_rst
        """
        logger.debug('start data predict_warm')
        start_time = time.time()
        try:
            rst, predict_y, predict_prob, error_reason = model.predict_with_proba(x=s_list)
            if rst and predict_y is not None:
                if labels is not None:
                    predict_y = map_predict_y(labels, predict_y)
                if predict_prob is not None:
                    rst = np.column_stack([predict_y, predict_prob]).astype(np.str)
                else:
                    rst = np.reshape(predict_y, (-1, 1))
                last_rst = []
                if split_record is not None and len(split_record) > 0:
                    start_idx = 0
                    for split_num in split_record:
                        last_rst.append(construct_last_rst(rst[start_idx:start_idx + split_num], error_reason))
                        start_idx += split_num
                    logger.info("split_record: {}, rst shape: {}".format(split_record, np.shape(last_rst)))
                else:
                    last_rst = construct_last_rst(rst, error_reason)
            else:
                logger.error("model.predict_with_proba failure. Reason: {}".format(error_reason))
                if split_record is not None and len(split_record) > 0:
                    last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
                else:
                    last_rst = construct_last_rst(None, error_reason)
        except Exception as e:
            logger.exception(
                "model.predict_with_proba failure with Exception. Input shape {} and error {}".format(np.shape(s_list),
                                                                                                      str(e)))
            error_reason = "model.predict_with_proba failure. Input shape {} and error {}".format(np.shape(s_list),
                                                                                                  str(e))
            if split_record is not None and len(split_record) > 0:
                last_rst = [construct_last_rst(None, error_reason)] * len(split_record)
            else:
                last_rst = construct_last_rst(None, error_reason)
        end_time = time.time()
        logger.info('Data pre-processing time = {}s. Model predict_warm time = {}s'.format(end_time2 - start_time2,
                                                                                           end_time - start_time))
        if end_time2 - start_time2 >= timeout_log_threshold:
            logger.warning("Slow pre-processing spend {} for data shape:\n{}".format((end_time2 - start_time2),
                                                                                     np.shape(s_list_bk)))
        yield last_rst


def load_model(model_id='None', weight_file_id='weight'):
    global error_reason
    from modelmanager.model_service import load_sklearn_model
    from darwinutils.config import DARWIN_CONFIG
    import glob
    if model_id is not None:
        weight_file_path = get_weight_file_path(model_id=model_id, weight_file_id=weight_file_id)
    else:
        weight_file_path = weight_file_id  # in this case, it is for java case and weight_file_id is the full path
    model_dump_files = glob.glob(os.path.join(weight_file_path, '*{}'.format(DARWIN_CONFIG.model_dump_suffix_ext)))
    model_dump_files += glob.glob(os.path.join(weight_file_path, '*model'))
    _ = list(map(lambda _: logger.debug("candicate model_dump_files for ml_model_train_post: {}".format(_)),
                 enumerate(model_dump_files)))

    # load model
    model = None
    model_dump_file = None
    for model_dump_file in model_dump_files:
        if not model_dump_file.endswith(DARWIN_CONFIG.model_dump_suffix_ext):
            # TODO: this is for trace only and could be removed after all dump had been moved to standard dump file.
            logger.warning("Try loading model from non-standard dump file {}".format(model_dump_file))
        try:
            model = load_sklearn_model(model_dump_file, isTrain=False)
            if model is not None:
                break
        except:
            pass
    if model is None:
        logger.error("Load model for model_id {} and weight id {} failed".format(model_id, weight_file_id))
        error_reason = "Load model for model_id {} and weight id {} failed".format(model_id, weight_file_id)
    return model


def load_label(model_id='None', weight_file_id='None'):
    # The function shall only be invoked by predict but not evaluation
    if model_id is not None:
        weight_file_path = get_weight_file_path(model_id=model_id, weight_file_id=weight_file_id)
    else:
        weight_file_path = weight_file_id  # in this case, it is for java case and weight_file_id is the full path
    if os.path.exists(os.path.join(weight_file_path, "label_id_dict.txt")):
        with open(os.path.join(weight_file_path, "label_id_dict.txt"), 'r') as f:
            rst = f.readlines()
        label_name_lst = list(map(lambda s: s.strip(), rst))
        return label_name_lst
    return None


def load_json_file(json_file_name, model_id='None', weight_file_id='None'):
    # The function shall only be invoked by predict but not evaluation
    if model_id is not None:
        weight_file_path = get_weight_file_path(model_id=model_id, weight_file_id=weight_file_id)
    else:
        weight_file_path = weight_file_id  # in this case, it is for java case and weight_file_id is the full path
    file_path = os.path.join(weight_file_path, json_file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                rst = json.load(f)
            except:
                f.seek(0)
                logger.exception("Read file {} failed and raw info: \n {}".format(file_path, f.readlines()))
                return None
        return rst
    return None


def map_predict_y(labels, predict_y):
    if labels is not None:
        try:
            if predict_y is not None and isinstance(predict_y, np.ndarray):
                predict_y = predict_y.tolist()
            if isinstance(predict_y, list):
                if isinstance(predict_y[0], list):
                    # [[1,3],[2,5]]
                    y_pred_class_names = list(map(lambda s: list(map(lambda ss: labels[ss], s)), predict_y))
                elif isinstance(predict_y[0], np.ndarray):
                    tmp_v = predict_y[0].tolist()
                    if isinstance(tmp_v, list):
                        # [array(1,2),array(2, 5)]
                        y_pred_class_names = list(map(lambda s: list(map(lambda ss: labels[ss], s)), predict_y))
                    else:
                        # [array(1),array(2)]
                        y_pred_class_names = list(map(lambda s: labels[s], predict_y))
                else:
                    # [1, 2]
                    y_pred_class_names = list(map(lambda s: labels[s], predict_y))
            else:
                # 1
                y_pred_class_names = labels[predict_y]
            return y_pred_class_names
        except:
            logger.error("Map predict_y to real class name failed. \ny_pred: {}\nlabel_name_lst: {}\n".format(
                predict_y, labels
            ))
            return predict_y
    else:
        return predict_y


def data_preprocessing(s_list, column_idx_map, data_auto_preprocess_record=None):
    """
    preprocess data by reprocess data_auto_preprocess_record to the input data
    preprocess data by reordering and reshaping data to np.array with expected column orders.
    :params s_list: batched input pre_data
    :column_idx_map: pre-build expected <column, idx> map to help reordering pre_data
    """
    global error_reason

    # 1. Get keys and values
    # 2. Merge Keys and Values as pandas dataframe, keys is the head
    # 3. Get interested column name as new dataframe
    # 4. To numpy array
    if s_list is None or len(s_list) == 0:
        error_reason = "Wrong input data"
        logger.error(error_reason)
        return None
    if column_idx_map is None or len(column_idx_map) == 0:
        error_reason = "Wrong required column list with data preprocessing flag True"
        logger.error(error_reason)
        return None

    if isinstance(s_list[0], dict):
        # 把dict格式的输入，转换成list格式的输入，好统一后续处理
        s_list = [list(zip(*each_s_list.items())) for each_s_list in s_list]
        s_list = [[column_names, [values]] for column_names, values in s_list]

    elif isinstance(s_list[0], (list, np.ndarray)):
        pass

    else:
        error_reason = "Unsupported input data type: {}".format(type(s_list[0]))
        logger.error(error_reason)
        return None

    if data_auto_preprocess_record is not None:
        for idx, data_block in enumerate(s_list):
            df = pd.DataFrame(data=data_block[1], columns=data_block[0])
            dp = DataPreprocess(source_path=None, dataframe=df,
                                process_records=data_auto_preprocess_record.get('all_files'))
            df = dp()
            s_list[idx] = [list(df.columns), df.values]

    # 处理标准List[NewType("column_names", List[str]), NewType("values", Iterable[Iterable[Number]])]
    combined_s_list = []
    for each_s_list in s_list:
        try:
            # validate column_names first.
            column_names = each_s_list[0]
            # column_names = list(filter(lambda s: column_idx_map.get(s) is not None, column_names))
            column_names, column_indices = zip(*filter(lambda s: column_idx_map.get(s[0]) is not None,
                                                       map(lambda x: (x[1], x[0]), enumerate(column_names))))
            if len(column_names) != len(column_idx_map):
                error_reason = "Column head unmatched issue. Expect {} columns but get {} columns. Miss {}".format(
                    len(column_idx_map),
                    len(column_names),
                    set(column_idx_map.keys()).difference(column_names),
                )
                logger.error(error_reason)
                return None

            # reorder the column to match expected order
            col_locs = list(map(column_idx_map.get, column_names))
            col_locs = np.argsort(col_locs)

            # process values now
            values = np.array(each_s_list[1])
            # input value data must be able to convert to float
            if len(np.shape(values)) == 1:
                values = values[column_indices].astype(np.float32)
                values = values[col_locs]
            else:
                values = values[:, column_indices].astype(np.float32)
                values = values[:, col_locs]

            # values = np.array(values, dtype=np.object)
            # values = values[:, col_locs].astype(np.float32)

            # save a valid value set
            combined_s_list.append(values)

        except Exception as e:
            error_reason = "Data has string value issue for model and cannot transfered to float [{}]".format(e)
            logger.exception(error_reason)
            return None

    # combine valid and reordered value sets back to an array
    s_list = np.concatenate(combined_s_list, axis=0)

    return s_list


def predict_cold_single_java(
        pre_data,
        count,
        weight_path=None,
        pre_process_flag=True,
        column_name_file="column_name_lst.json",
        setup=False,
):
    """
    In java sdk. the input pre_data is very different, it is just a list list[0] is the head and list[1] is the data

    :param pre_data:
    :param count:
    :param weight_path:
    :param pre_process_flag:
    :param column_name_file:
    :param setup:
    :return:
    """
    # the function will deal with the input from customer caller. to keep same api as restful,
    # part of restful code add to here
    global error_reason
    global cold_model
    global cold_labels
    global cold_column_idx_map

    error_reason = ""
    if setup:
        # load model during setup
        cold_model = load_model(model_id=None, weight_file_id=weight_path)
        cold_labels = load_label(model_id=None, weight_file_id=weight_path)
        if column_name_file is not None:
            column_name_lst = None
            column_name_json = load_json_file(column_name_file, model_id=None, weight_file_id=weight_path)
            if isinstance(column_name_json, dict) and column_name_json.get("feature_names") is not None and isinstance(
                    column_name_json["feature_names"], list):
                column_name_lst = column_name_json["feature_names"]
            if column_name_lst is None:
                error_reason = "column name file {} content is not right.".format(column_name_file)
                last_rst = construct_last_rst(None, error_reason)
                return last_rst
        else:
            error_reason = "column_name_file should not be empty"
            last_rst = construct_last_rst(None, error_reason)
            return last_rst

        # only setup, return now
        if column_name_lst is not None:
            # build column_idx_map to speed up remaping when receiving data from java client
            cold_column_idx_map = {k: idx for idx, k in enumerate(column_name_lst)}
            return construct_last_rst(list(column_name_lst), error_reason)
        else:
            return construct_last_rst([], error_reason)
    else:
        model = cold_model
        labels = cold_labels
        column_idx_map = cold_column_idx_map

    logger.debug('start cold node predict')
    start_time2 = time.time()
    if pre_process_flag:
        logger.debug('start data pre-processing')
        data = data_preprocessing([pre_data], column_idx_map)
        if data is None:
            return construct_last_rst(None, error_reason)
    else:
        data = pre_data
    if len(data) != int(count):
        error_reason = "Data len {} != count {}".format(len(data), count)
        return construct_last_rst(None, error_reason)
    end_time2 = time.time()

    logger.debug('start data predict_cold_single_java')
    start_time = time.time()
    try:
        rst, predict_y, predict_prob, error_reason = model.predict_with_proba(x=data)
        if rst and predict_y is not None:
            if labels is not None:
                predict_y = map_predict_y(labels, predict_y)
            if predict_prob is not None:
                rst = np.column_stack([predict_y, predict_prob]).astype(np.str)
            else:
                rst = np.reshape(predict_y, (-1, 1))
            last_rst = construct_last_rst(rst, str(error_reason))
        else:
            error_reason = "model.predict_with_proba failure. Reason: {}".format(error_reason)
            return_code = 500
            logger.error(error_reason)
            last_rst = construct_last_rst(None, error_reason, return_code)
    except Exception:
        logger.exception("model.predict_with_proba failure with Exception. Input shape {}".format(np.shape(data)))
        error_reason = "model.predict_with_proba failure. Input shape {}".format(np.shape(data))
        return_code = 500
        last_rst = construct_last_rst(None, error_reason, return_code)
    end_time = time.time()
    logger.debug('Data pre-processing time = {}s. Model predict_cold time = {}s'.format(end_time2 - start_time2,
                                                                                        end_time - start_time))
    if end_time2 - start_time2 >= 0.03:
        logger.warning("Slow pre-processing spend {} for data:\n{}".format((end_time2 - start_time2), pre_data))
    return last_rst
