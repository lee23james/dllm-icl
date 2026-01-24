import json
import os
from typing import List

import faiss
import more_itertools
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
#专门针对数据处理进行
#这里应该有模型适配,但是我目前还没开始做(等之后要用到我就开始进行适配)


#这段代码有问题,因为它的query默认是放在最后一位的,如果之后我要训练,那么我应该进行区分
#因为我这里考虑了位置,所以之后我区分训练集和验证集的时候应该区分,
def data_split(generated_data, train_ratio):
    # 获得有多少条test数据
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]["id_list"]
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少test数据用于训练 剩下部分用于监督val loss
    train_data_len = int(train_ratio * test_dataset_len)
    train_idx_set = set(sorted(list(test_dataset_id_set))[:train_data_len])
    val_idx_set = test_dataset_id_set - train_idx_set

    train_data_list = list()
    val_data_list = list()
    train_data_score = list()
    val_data_score = list()
    #这里是按一组icd进行配置的,一组icd包含一个query和一群candidates,但是这里默认query放在最后一个位置
    #之后我可能要进行标记,每一个shot都有一个query标识
    for d in generated_data:
        for i in range(len(generated_data[d]["id_list"])):
            query_idx = generated_data[d]["id_list"][i][-1]
            if int(query_idx) in train_idx_set:
                train_data_list.append(generated_data[d]["id_list"][i])
                train_data_score.append(generated_data[d]["score_list"][i])
            elif int(query_idx) in val_idx_set:
                val_data_list.append(generated_data[d]["id_list"][i])
                val_data_score.append(generated_data[d]["score_list"][i])
            else:
                raise ValueError()

    print(f"the train size {len(train_data_list)}, the test size {len(val_data_list)}")

    train_data = {
        "icd_seq": train_data_list,
        "icd_score": train_data_score,
    }
    val_data = {
        "icd_seq": val_data_list,
        "icd_score": val_data_score,
    }
    return train_data, val_data

