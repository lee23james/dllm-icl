import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig

#这里开始去寻找适配每一个问题的候选数据集
from lever_lm.utils import beam_filter
#在这里实现info_score
# 导入评分函数（用户自己实现）
# 注意：get_info_score需要用户自己实现，这里只是占位
try:
    from utils import get_info_score
except ImportError:
    try:
        from lever_lm.utils import get_info_score
    except ImportError:
        # 临时占位：返回随机分数（用户需要实现真实的get_info_score）
        #这里有很严中的问题,我并有写的是getinfo的函数
        def get_info_score(*args, **kwargs):
            import torch
            candidate_data_list = kwargs.get("candidate_data_list", [])
            num = len(candidate_data_list) if candidate_data_list else 1
            logger.warning(f"Using placeholder get_info_score, returning random scores for {num} candidates")
            return torch.randn(num, dtype=torch.float32)


def find_query_position(id_seq: List[int], candidateidx2data: Dict) -> int:
    """
    找到query在序列中的位置
    
    Args:
        id_seq: 包含ID的序列
        candidateidx2data: ID到数据的映射字典
    
    Returns:
        query在序列中的位置索引，如果没找到返回-1
    """
    for i, idx in enumerate(id_seq):
        #相当于candidate-set中记录的是所有idx与数据的对应关系(包括数据的所有状态)
        if idx in candidateidx2data and candidateidx2data[idx].get("isquery", 0) == 1:
            return i
    # 如果序列中只有一个元素，且是test_data，那么它就是query
    if len(id_seq) == 1:
        return 0
    return -1


def get_insert_positions(id_seq: List[int], query_pos: int, metric: str) -> List[int]:
    """
    根据metric参数获取所有可能的插入位置
    
    Args:
        id_seq: 当前序列
        query_pos: query在序列中的位置
        metric: "no_order" 或 "order"
    
    Returns:
        所有可能的插入位置列表
    """
    if metric == "no_order":
        # 只在两侧插入：query之前（最左侧）或query之后（最右侧）
        positions = []
        # query之前的位置（最左侧）
        positions.append(0)
        # query之后的位置（最右侧）
        if len(id_seq) > 0:
            positions.append(len(id_seq))
        # 去重并排序
        return sorted(list(set(positions)))
    elif metric == "order":
        # 在所有可能位置插入：包括中间和两侧
        # 可以插入的位置：0, 1, 2, ..., len(id_seq)
        # 这意味着可以在任意两个元素之间插入，包括最前面和最后面
        return list(range(len(id_seq) + 1))
    else:
        raise ValueError(f"Unknown metric: {metric}. Must be 'no_order' or 'order'")

#这个是为每一个test的例子从candidate集合中选择合适的ICD,构建出序列,并进行分数计算
@torch.inference_mode()
def generate_single_sample_icd(
    interface,  # BaseInterface类型，用于计算分数
    test_data: Dict,  # 这个后期一定要进行标注,标出这个是否是query
    cfg: DictConfig,
    candidate_set: Dataset,
    metric: str = "no_order",  # "no_order" 或 "order"
):
    """
    为每个测试的query找到合适的ICD（In-Context Demonstration）
    、
    Args:
        interface: 用于计算分数的接口
        test_data: 测试数据，必须包含"idx"字段，且isquery=1
        cfg: 配置对象，需要包含few_shot_num, beam_size, scorer等
        candidate_set: 候选数据集
        metric: "no_order"表示不关注顺序（只在两侧插入），"order"表示关注顺序（所有位置插入）
    
    Returns:
        包含id_list和score_list的字典
    """
    #测试集的id,默认测试集的id并没有进入candidata_set中
    test_data_id = test_data["idx"]
    # 标记test_data为query
    test_data["isquery"] = 1
    
    # 构建ID到数据的映射表
    # 注意：虽然名字叫candidateidx2data，但实际上需要包含所有可能出现在序列中的数据
    # 包括候选ICD和query本身，因为代码需要通过ID统一访问这些数据
    candidateidx2data = {data["idx"]: data for data in candidate_set}
    # 必须把test_data也加入映射，因为：
    # 1. test_data_id会出现在test_data_id_list中（初始序列）
    # 2. 代码需要通过candidateidx2data[idx]获取序列中每个ID对应的数据
    # 3. find_query_position需要通过映射查找query位置
    #这个维护的是id与数据的关系
    candidateidx2data[test_data_id] = test_data
    
    # 初始化：只包含query的序列
    test_data_id_list = [[test_data_id]]
    
    for step in range(cfg.few_shot_num):
        # 每一轮构建的时候都会选择新的数据集
        new_test_data_id_list = []  # 这个是新的数据集
        new_test_score_list = []  # 这个是记录分数的
        
        for test_data_id_seq in test_data_id_list:
            # 避免添加重复的结果，将已经添加的进行过滤
            filtered_candidateidx2data = candidateidx2data.copy()
            # 过滤掉已经使用的ICD和query（因为query已经在序列中，不应该再被插入）
            #相当于在函数内进行所有可能的情况讨论,同时
            for idx in test_data_id_seq:
                if idx in filtered_candidateidx2data:
                    # 过滤掉所有已经在序列中的元素（包括ICD和query）
                    filtered_candidateidx2data.pop(idx)
            
            # 找到query的位置,对于当前的ICD是一个固定的数字
            query_pos = find_query_position(test_data_id_seq, candidateidx2data)
            if query_pos == -1:
                logger.warning(f"Query not found in sequence {test_data_id_seq}, skipping...")
                continue
            
            # 根据metric获取所有可能的插入位置,返回的是插入的位置,注意这里0是开头len()是结尾
            insert_positions = get_insert_positions(test_data_id_seq, query_pos, metric)
            
            # 构建当前已选好的ICD+测试样本的序列（用于计算分数）
            # 当前的candidateidx2data包括query和所有候选ICD,test_data_id_seq只有当前的ICD和query,还没加新ICD
            current_seq_data = [candidateidx2data[idx] for idx in test_data_id_seq]
            
            # 对每个候选ICD和每个插入位置进行评分,排序
            filtered_idx_list = sorted(list(filtered_candidateidx2data.keys()))
            
            if len(filtered_idx_list) == 0:
                logger.warning("No more candidates available, stopping early")
                break
            
            # 构建所有(候选ID, 插入位置)的组合列表
            # 用于批量处理和索引映射: (ICD_idx, position) -> index
            candidate_position_pairs = []
            candidate_data_list = []
            position_list = []
            
            for candidate_idx in filtered_idx_list:
                for insert_pos in insert_positions:
                    candidate_position_pairs.append((candidate_idx, insert_pos))
                    candidate_data_list.append(filtered_candidateidx2data[candidate_idx])
                    position_list.append(insert_pos)
            
            if len(candidate_position_pairs) == 0:
                logger.warning("No candidate-position pairs to evaluate, skipping this step")
                continue
            
            # 批量计算分数（需要根据cfg.scorer调用相应的评分函数）
            try:
                if cfg.scorer == "infoscore":
                    # 尝试从不同位置导入get_info_score
                    # try:
                    #     from lever_lm.utils import get_info_score
                    # except ImportError:
                    #     raise ImportError("get_info_score not found. Please implement or adjust import path.")
                    
                    # 批量调用get_info_score，传入所有候选数据和位置
                    # get_info_score应该返回一个tensor，包含所有(候选,位置)组合的分数
                    # scores shape: (len(candidate_position_pairs),)
                    #在这里面就是不断处理这两种组合的情况得到的所有分数
                    scores = get_info_score(
                        interface,
                        choosed_icd_seq_list=current_seq_data,  # 插入前的序列
                        #下面两个构成插入顺序,但是并不一定是连续的,存在一个转化表candidate_position_pairs,将二维转化为一维,并且根据一维的指标知道真实的idx是哪一个
                        candidate_data_list=candidate_data_list,  # 所有要插入的候选数据列表
                        position_list=position_list,  # 所有插入位置列表
                        batch_size=cfg.batch_size,
                        split_token=cfg.task.split_token,
                    )
                    
                    # scores应该是一个tensor，shape为(len(candidate_position_pairs),)
                    if not isinstance(scores, torch.Tensor):
                        scores = torch.tensor(scores, dtype=torch.float32)
                    
                    # 确保scores是一维tensor
                    scores = scores.flatten()
                    
                    if len(scores) != len(candidate_position_pairs):
                        logger.error(f"Score length mismatch: expected {len(candidate_position_pairs)}, got {len(scores)}")
                        continue
                    
                    # 选出最高的InfoScore,这里的indices也是跳段段
                    #局部筛选
                    topk_scores, indices = scores.topk(cfg.beam_size)
                    
                    # 将索引转换为列表
                    indices = indices.tolist()
                    topk_scores = topk_scores.tolist()
                    
                    # 根据索引找到对应的(候选ID, 插入位置)组合，并构建新序列
                    #这个idx对应的应该是candidate_position_pairs,不是真实idx
                    for idx, score in zip(indices, topk_scores):
                        #把(候选ID,插入位置)转化为以一种统一的尺度进行衡量
                        candidate_idx, insert_pos = candidate_position_pairs[idx]
                        # 构建新的序列,并进行插入
                        new_seq_ids = test_data_id_seq.copy()
                        new_seq_ids.insert(insert_pos, candidate_idx)
                        # 分数和序列一一对应
                        new_test_data_id_list.append(new_seq_ids)
                        new_test_score_list.append(score)
                        
                else:
                    raise ValueError(f"Unknown scorer: {cfg.scorer}. Must be 'infoscore' or 'cider'")
                    
            except Exception as e:
                logger.error(f"Error computing scores: {e}")
                continue
        
        #对每一步的上一步剩下的beam_size的发散的所有分支进行汇总
        # Beam filter: 保留top beam_size个序列
        if len(new_test_score_list) > 0:
            new_test_score_list, new_test_data_id_list = beam_filter(
                new_test_score_list, new_test_data_id_list, cfg.beam_size
            )
            test_data_id_list = new_test_data_id_list
        else:
            logger.warning("No new sequences generated, stopping early")
            break
    
    return {
        test_data_id: {
            "id_list": test_data_id_list,
            "score_list": new_test_score_list
        }
    }

    