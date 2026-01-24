import os
from typing import Dict, List, Optional, Union

import hydra
import more_itertools
import torch
from loguru import logger
from transformers import AutoProcessor

#这里开始计算分数,需要参考的主要是之前做的东西
#这里是打分函数,我想用来进行分数的计算
from lever_lm.load_ds_utils import load_hf_ds,load_gsm8k_ds

#cfg是任务配置,方便进行查询
def load_ds(cfg,split=None):
    if cfg.task.task_name == "gsm8k":
        ds=load_gsm8k_ds(
            version=cfg.dataset.version,
            train_path=cfg.dataset.train_path,
            val_path=cfg.dataset.val_path
        )
    else:
        try:
            ds=load_hf_ds(cfg.dataset.hf_ds)
        except Exception as e:
            raise ValueError(f"dataset load fail with error: {e}")
    return ds

#需要模型和数据集分别进行计算,这里计算的是互信息
#先把主要的逻辑写了,之后再进行补充
@torch.no_grad()
def get_info_score(
    interface,
    choosed_icd_seq_list: List,  # 插入前的序列（当前已选好的ICD+query）
    candidate_data_list: List,  # 所有要插入的候选数据列表
    position_list: List,  # 所有插入位置列表（与candidate_data_list一一对应）
    batch_size: int,
    split_token: Optional[str] = None,
) -> torch.Tensor:
    """
    批量计算InfoScore分数
    
    Args:
        interface: 用于计算分数的接口
        choosed_icd_seq_list: 当前序列（插入前的序列），是一个列表，包含所有数据字典
        candidate_data_list: 所有要插入的候选数据列表，每个元素是一个数据字典
        position_list: 所有插入位置列表，与candidate_data_list一一对应
        batch_size: 批处理大小
        split_token: 分隔符（可选）
    
    Returns:
        scores: torch.Tensor, shape=(len(candidate_data_list),)，包含所有候选-位置组合的分数
    """
    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    #这里先照抄它们的,等我之后进行了适配再进行更改
    #==========================================
    # 改进
    #==========================================
    #这里可以直接用我们之前的模型,直接进行处理,之前用的处理的计算置信度的式子
    
    # TODO: 实现InfoScore计算逻辑
    # 对于每个(candidate_data, position)组合：
    #   1. 构建插入后的序列
    #   2. 计算插入前后的置信度差异
    #   3. 返回InfoScore
    
    # 临时返回：返回随机分数（需要后续实现）
    num_scores = len(candidate_data_list)
    scores = torch.randn(num_scores, dtype=torch.float32)
    return scores
    
