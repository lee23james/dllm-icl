import os
from typing import Dict, List, Optional, Union, Tuple

import hydra
import more_itertools
import torch
from loguru import logger
from transformers import AutoProcessor
from open_mmicl.interface import LLaDAInterface
from open_mmicl.interface import BaseInterface
#这里开始计算分数,需要参考的主要是之前做的东西
#这里是打分函数,我想用来进行分数的计算
from lever_lm.load_ds_utils import load_hf_ds,load_gsm8k_ds

#cfg是任务配置,方便进行查询
#
def load_ds(cfg,split=None):
    """
    加载数据集
    
    Args:
        cfg: 配置对象
        split: 要加载的split名称（如 "train"），如果为None，则根据任务类型决定
    
    Returns:
        加载的数据集
    """
    if cfg.task.task_name == "gsm8k":
        # 使用方式2：只加载单个split
        if split == "train":
            data_path = cfg.dataset.train_path
        elif split == "test" or split == "validation":
            data_path = cfg.dataset.get("test_path") or cfg.dataset.get("val_path")
            if data_path is None:
                raise ValueError("test_path or val_path must be provided for test/validation split")
        else:
            # 默认加载train
            data_path = cfg.dataset.train_path
            split = "train"
        
        ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=data_path,
            split=split
        )
    else:
        try:
            ds=load_hf_ds(cfg.dataset.hf_ds)
        except Exception as e:
            raise ValueError(f"dataset load fail with error: {e}")
    return ds

#需要模型和数据集分别进行计算,这里计算的是互信息
#使用蒙特卡洛方法计算InfoScore
#这里稍微有点问题,就是,这里接受的输入是各个位置的吗,还是其他的,就是需不需要外部调用的时候再传递进来另外一个位置的信息根据这个信息进行判定
@torch.no_grad()
def get_info_score(
    interface,
    choosed_icd_seq_list: List[Dict],  # 插入前的序列（当前已选好的ICD+query）
    candidate_data_list: List[Dict],   # 所有要插入的候选数据列表
    position_list: List[int],          # 所有插入位置列表（与candidate_data_list一一对应）
    batch_size: int = 1,
    split_token: Optional[str] = None,
    mc_num: int = 128,                 # Monte Carlo采样次数
    cfg_scale: float = 0.0,            # CFG scale
) -> torch.Tensor:
    """
    批量计算InfoScore分数（反向插入语义）
    
    适用于双向 ICL（Diffusion LLM）：query 可以在序列中间，左右都可以有 ICD
    
    调用流程：
    1. 从 choosed_icd_seq_list 中根据 isquery=1 定位 query
    2. 将序列拆分为 (left_icd_list, query, right_icd_list)
    3. 计算插入前的 baseline 分数
    4. 对每个 (candidate_data, position) 组合：
       - 反向插入：position 越小，插入位置越靠后
       - 构建新的 (prompt_left, answer, prompt_right)
       - 调用 interface.compute_log_likelihood 计算分数
       - InfoScore = score_after - score_before
    
    Args:
        interface: BaseInterface 子类（如 LLaDAInterface），提供 compute_log_likelihood 方法
        choosed_icd_seq_list: 当前序列（插入前），每个元素是 Dict，其中一个有 isquery=1
        candidate_data_list: 候选 ICD 数据列表，每个元素是 Dict
                            支持同一个 candidate 出现多次以测试不同位置
        position_list: 插入位置列表，与 candidate_data_list 一一对应
                       **反向插入语义**：
                       - position=0: 插入到序列末尾（最远）
                       - position=1: 插入到倒数第二个位置
                       - position=k: 插入到倒数第 k+1 个位置
                       
                       公式：insert_pos = len(choosed_icd_seq_list) - position
        batch_size: 批处理大小（目前固定为 1）
        split_token: ICD 之间的分隔符（默认 "\n\n"）
        mc_num: Monte Carlo 采样次数（默认 128）
        cfg_scale: CFG scale（默认 0.0）
    
    Returns:
        scores: torch.Tensor, shape=(len(candidate_data_list),)
                每个元素是对应 (candidate, position) 的 InfoScore
    
    示例1（双向 ICL - query 在中间）：
        choosed_icd_seq_list = [icd1, icd2, query, icd3, icd4]  # 长度=5
        
        candidate_data_list = [new, new, new]
        position_list = [0, 1, 2]
        结果：
          - position=0 → insert_pos=5 → [icd1, icd2, query, icd3, icd4, new]（末尾）
          - position=1 → insert_pos=4 → [icd1, icd2, query, icd3, new, icd4]（倒数第二）
          - position=2 → insert_pos=3 → [icd1, icd2, query, new, icd3, icd4]（倒数第三）
    
    示例2（只有 query）：
        choosed_icd_seq_list = [query]  # 长度=1
        
        candidate_data_list = [new, new]
        position_list = [0, 1]
        结果：
          - position=0 → insert_pos=1 → [query, new]（末尾）
          - position=1 → insert_pos=0 → [new, query]（开头）
    """
    # 1. 根据 isquery 定位 query 的位置，并分割序列
    query_idx = None
    for i, item in enumerate(choosed_icd_seq_list):
        if item.get("isquery", 0) == 1:
            query_idx = i
            break
    
    if query_idx is None:
        raise ValueError("No query found in choosed_icd_seq_list (no item with isquery=1)")
    
    query = choosed_icd_seq_list[query_idx]
    
    # 检查query是否包含答案信息（answer 字段或 q_a 字段）
    if "answer" not in query and "q_a" not in query:
        raise ValueError("query must contain 'answer' or 'q_a' field for Monte Carlo score calculation")
    
    # 根据 query 位置分割序列：
    # - left_icd_list: query 前面的 ICD
    # - query: query 本身
    # - right_icd_list: query 后面的 ICD（如果有）
    left_icd_list = [item for item in choosed_icd_seq_list[:query_idx] if item.get("isquery", 0) == 0]
    right_icd_list = [item for item in choosed_icd_seq_list[query_idx+1:] if item.get("isquery", 0) == 0]
    
    # 2. 准备 tokenizer & 辅助函数（基于 BaseInterface 提供的通用能力）
    tokenizer = interface.tokenizer
    device = interface.device
    sep = split_token if split_token is not None else "\n\n"

    # 优先使用 Interface 上的 PromptTemplate（统一使用配置中的 prompt_template）
    pt = getattr(interface, "pt", None)

    if pt is not None:
        def build_prompt_parts(left_icds: List[Dict], query_sample: Dict, right_icds: List[Dict]) -> Tuple[str, str, Optional[str]]:
            """
            使用 PromptTemplate 构建打分用的三个部分：
            - left_text: left ICD + query 的 question + "<answer>\\n"（不含答案主体）
            - answer_text: query 的答案文本（位于 <answer> 与 </answer> 之间）
            - right_text:  "</answer>" + 右侧 ICD（如果有）

            要求 prompt_template 形如：
                question: <Q>\\n<answer>\\n<A>\\n</answer>
            """
            # 1) 使用模板生成左侧 ICD 的完整文本
            left_icd_texts: List[str] = []
            for s in left_icds:
                icd_text = pt.generate_ice_item(s)
                left_icd_texts.append(icd_text.strip())

            # 2) 使用模板生成 query 的完整文本（包含 <answer> 和 </answer>）
            full_query = pt.generate_ice_item(query_sample)
            if "<answer>" not in full_query or "</answer>" not in full_query:
                raise ValueError(
                    "Prompt template used for scoring must contain <answer> and </answer> tags, "
                    f"got: {full_query[:100]}"
                )

            # 拆分成：prefix（含 question 和到 <answer> 为止）、answer_block、suffix（</answer> 之后）
            before, rest = full_query.split("<answer>", 1)
            answer_block, after = rest.split("</answer>", 1)

            # prefix：保持到 `<answer>\n`，用于 left_text
            # 这里显式补回 "<answer>" 以及一个换行，保证格式为：
            # question: Q\n<answer>\n
            prefix = before.rstrip() + "\n<answer>\n"

            # answer_text：A_query（去掉多余空白）
            answer_text = answer_block.strip()

            # suffix："</answer>" 后面的内容（通常为空），我们统一用固定的关闭标签
            closing_tag = "\n</answer>"

            # 3) left_text = 左侧 ICD + prefix
            left_parts = left_icd_texts + [prefix.strip()]
            left_text = sep.join(left_parts)

            # 4) right_text = "</answer>" + 右侧 ICD（如果有）
            right_text = closing_tag
            if right_icds:
                right_icd_texts: List[str] = []
                for s in right_icds:
                    icd_text = pt.generate_ice_item(s)
                    right_icd_texts.append(icd_text.strip())
                right_icd_block = sep.join(right_icd_texts)
                right_text = f"{closing_tag}{sep}{right_icd_block}"

            return left_text, answer_text, right_text

    else:
        # 兼容旧逻辑：使用硬编码的 GSM8K 风格模板
        def format_icd(sample: Dict) -> str:
            """将单个 ICD 样本格式化成统一的文本表示。"""
            if "q_a" in sample:
                return sample["q_a"]
            if "question" in sample and "answer" in sample:
                return f"question: {sample['question']}\n<answer>\n{sample['answer']}\n</answer>"
            raise ValueError(f"Unknown ICD format: {sample.keys()}")

        def format_query_prefix(query_sample: Dict) -> str:
            """格式化 query 的 question 部分（不含答案）。"""
            if "question" in query_sample:
                return f"question: {query_sample['question']}\n<answer>\n"
            elif "q_a" in query_sample:
                qa_text = query_sample["q_a"]
                if "<answer>" in qa_text:
                    return qa_text.split("<answer>")[0] + "<answer>\n"
                return qa_text
            else:
                raise ValueError(f"Unknown query format: {query_sample.keys()}")

        def extract_answer_text(query_sample: Dict) -> str:
            """从 query 中提取答案文本，优先使用 answer 字段，其次解析 q_a。"""
            if "answer" in query_sample:
                return str(query_sample["answer"])
            if "q_a" in query_sample:
                qa_text = query_sample["q_a"]
                if "<answer>" in qa_text and "</answer>" in qa_text:
                    return qa_text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
                if "<answer>" in qa_text:
                    return qa_text.split("<answer>", 1)[1].strip()
                return qa_text
            raise ValueError("query_sample must contain 'answer' or 'q_a' to extract answer text")

        def build_prompt_parts(left_icds: List[Dict], query_sample: Dict, right_icds: List[Dict]) -> Tuple[str, str, Optional[str]]:
            """
            根据 left ICD、query、right ICD 构建 prompt 的三个部分：
            - left_text:  left ICD + query(不含答案) 部分
            - answer_text:  query 的答案文本
            - right_text:  right ICD 部分（如果有）
            """
            # 格式化 left ICD
            left_icd_parts = [format_icd(s) for s in left_icds]
            
            # query 的 question 部分
            query_prefix = format_query_prefix(query_sample)
            
            # 组合 left 部分
            left_parts = left_icd_parts + [query_prefix]
            left_text = sep.join(left_parts)
            
            # 答案文本
            answer_text = extract_answer_text(query_sample)
            
            # 格式化 right ICD（在 answer 之后补上 </answer> 即使没有右侧内容）
            closing_tag = "\n</answer>"
            right_text = closing_tag
            if right_icds:
                right_icd_parts = [format_icd(s) for s in right_icds]
                right_icd_block = sep.join(right_icd_parts)
                right_text = f"{closing_tag}{sep}{right_icd_block}"
            
            return left_text, answer_text, right_text

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer(text)["input_ids"]
        return torch.tensor(ids, device=device)

    # 3. 计算插入前的分数（使用 BaseInterface 的 compute_log_likelihood）
    left_before_text, answer_text, right_before_text = build_prompt_parts(left_icd_list, query, right_icd_list)
    prompt_left_before = encode(left_before_text)   # 1D: 左侧 prompt
    answer_ids = encode(answer_text)                # 1D: 答案 tokens
    prompt_right_before = encode(right_before_text) if right_before_text else None  # 1D: 右侧 prompt（可选）

    score_before = interface.compute_log_likelihood(
        prompt_left=prompt_left_before,
        answer=answer_ids,
        prompt_right=prompt_right_before,
        mc_num=mc_num,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
    )
    
    # ========== 4. 对每个 (candidate, position) 计算 InfoScore ==========
    scores: List[float] = []
    
    for candidate_data, position in zip(candidate_data_list, position_list):
        # 4.1 构建插入后的序列
        # position 语义（反向插入）：
        # - position=0: 插入到序列末尾（最远）
        # - position=1: 插入到倒数第二个位置
        # - position=k: 插入到倒数第 k+1 个位置
        # 
        # 公式：insert_pos = len(choosed_icd_seq_list) - position
        # 
        # 示例：choosed_icd_seq_list = [icd1, icd2, query, icd3, icd4]（长度=5）
        #   position=0 → insert_pos=5 → [icd1, icd2, query, icd3, icd4, new]
        #   position=1 → insert_pos=4 → [icd1, icd2, query, icd3, new, icd4]
        #   position=2 → insert_pos=3 → [icd1, icd2, query, new, icd3, icd4]
        
        # 构建新序列：在原始序列的反向 position 位置插入 candidate
        new_seq = choosed_icd_seq_list.copy()
        
        # 反向插入：position 越小，插入位置越靠后
        total_length = len(new_seq)
        insert_pos = total_length - position
        
        # 确保插入位置合法
        if insert_pos < 0:
            insert_pos = 0
        elif insert_pos > total_length:
            insert_pos = total_length
        
        new_seq.insert(insert_pos, candidate_data)
        
        # 重新拆分序列为 left, query, right
        # 找到 query 的新位置（可能因为插入而改变）
        new_query_idx = None
        for i, item in enumerate(new_seq):
            if item.get("isquery", 0) == 1:
                new_query_idx = i
                break
        
        if new_query_idx is None:
            logger.error("Query not found after insertion, skipping...")
            scores.append(0.0)
            continue
        
        # 拆分新序列
        left_icd_after = [item for item in new_seq[:new_query_idx] if item.get("isquery", 0) == 0]
        right_icd_after = [item for item in new_seq[new_query_idx+1:] if item.get("isquery", 0) == 0]
        query_after = new_seq[new_query_idx]
        
        # 4.2 构建插入后的 prompt 部分
        left_after_text, _, right_after_text = build_prompt_parts(
            left_icd_after, query_after, right_icd_after
        )
        prompt_left_after = encode(left_after_text)
        prompt_right_after = encode(right_after_text) if right_after_text else None
        
        # 4.3 计算插入后的分数
        score_after = interface.compute_log_likelihood(
            prompt_left=prompt_left_after,
            answer=answer_ids,
            prompt_right=prompt_right_after,
            mc_num=mc_num,
            batch_size=batch_size,
            cfg_scale=cfg_scale,
        )
        
        # 4.4 计算 InfoScore = 插入后 - 插入前
        infoscore = score_after - score_before
        scores.append(infoscore)
        
        logger.debug(
            f"Candidate at position {position} (insert_pos={insert_pos}): "
            f"score_after={score_after:.4f}, infoscore={infoscore:.4f}"
        )
    
    return torch.tensor(scores, dtype=torch.float32)
    