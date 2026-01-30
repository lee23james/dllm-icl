import json
import os
from typing import List, Tuple

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


def beam_filter(score_list: List[float], id_list: List[List[int]], beam_size: int) -> Tuple[List[float], List[List[int]]]:
    """
    Beam filter：保留top beam_size个序列
    
    Args:
        score_list: 分数列表
        id_list: 序列ID列表（每个元素是一个序列的ID列表）
        beam_size: 要保留的序列数量
    
    Returns:
        (filtered_score_list, filtered_id_list): 过滤后的分数和序列列表
    """
    if len(score_list) == 0:
        return [], []
    
    if len(score_list) != len(id_list):
        logger.warning(f"Score list length ({len(score_list)}) != id list length ({len(id_list)}), using min length")
        min_len = min(len(score_list), len(id_list))
        score_list = score_list[:min_len]
        id_list = id_list[:min_len]
    
    # 按分数排序（降序）
    paired = list(zip(score_list, id_list))
    paired.sort(key=lambda x: x[0], reverse=True)
    
    # 保留top beam_size个
    filtered_paired = paired[:beam_size]
    
    # 解包
    filtered_score_list = [score for score, _ in filtered_paired]
    filtered_id_list = [id_seq for _, id_seq in filtered_paired]
    
    return filtered_score_list, filtered_id_list


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

