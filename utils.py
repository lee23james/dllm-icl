import json
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
from lever_lm.load_ds_utils import load_hf_ds, load_gsm8k_ds, load_mmlu_ds, load_subj_ds

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
    elif cfg.task.task_name == "mmlu":
        # MMLU 任务：使用本地 parquet + load_mmlu_ds（会自动展开 choices 并添加 idx / isquery）
        if split == "train":
            train_path = cfg.dataset.train_path
            ds = load_mmlu_ds(
                version=cfg.dataset.version,
                data_path=train_path,
                split="train",
            )
        elif split in ("test", "validation"):
            test_path = cfg.dataset.get("test_path") or cfg.dataset.get("val_path")
            if test_path is None:
                raise ValueError("test_path or val_path must be provided for test/validation split in MMLU")
            ds = load_mmlu_ds(
                version=cfg.dataset.version,
                data_path=test_path,
                split="validation",
            )
        else:
            # 默认加载 train
            train_path = cfg.dataset.train_path
            ds = load_mmlu_ds(
                version=cfg.dataset.version,
                data_path=train_path,
                split="train",
            )
    elif cfg.task.task_name == "subj":
        label_mapping = cfg.task.get("label_mapping") or cfg.dataset.get("label_mapping")
        if label_mapping is None:
            label_mapping = {"0": "subjective", "1": "objective"}
        if split == "train":
            data_path = cfg.dataset.train_path
            ds = load_subj_ds(
                data_path=data_path,
                split="train",
                label_mapping=label_mapping,
            )
        elif split in ("test", "validation"):
            data_path = cfg.dataset.get("test_path") or cfg.dataset.get("val_path")
            if data_path is None:
                raise ValueError("test_path or val_path must be provided for test/validation split in subj")
            ds = load_subj_ds(
                data_path=data_path,
                split="validation",
                label_mapping=label_mapping,
            )
        else:
            data_path = cfg.dataset.train_path
            ds = load_subj_ds(
                data_path=data_path,
                split="train",
                label_mapping=label_mapping,
            )
    else:
        try:
            # 其他 HF 数据集：可选展开 choices 数组
            expand_choices = (cfg.task.task_name == "mmlu")
            ds = load_hf_ds(cfg.dataset.hf_ds, expand_choices=expand_choices)
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
    output_column: Optional[str] = None,  # 用于从 query 中提取答案字段（如 "answer"）
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
        if output_column is None:
            raise ValueError(
                "output_column must be provided when using PromptTemplate for InfoScore scoring "
                "(e.g., 'answer' as defined in task.output_column)."
            )

        def build_prompt_parts(
            left_icds: List[Dict],
            query_sample: Dict,
            right_icds: List[Dict],
        ) -> Tuple[str, str, Optional[str]]:
            """
            通用打分模板（适用于 MMLU / GSM8K 等）：
            - 上部分 left_text  = 前面的 ICD（完整 prompt） + 当前 query 的“无答案版 prompt”
              （通过 PromptTemplate.generate_text_for_embedding 去掉 output_column 字段）
            - 中间部分 answer_text = query_sample[output_column]（答案本身）
            - 下部分 right_text  = 后面的 ICD（完整 prompt），如果没有则为 None
            """
            # 1) 左侧 ICD：使用完整的 ICD prompt
            left_icd_texts: List[str] = []
            for s in left_icds:
                icd_text = pt.generate_ice_item(s)
                left_icd_texts.append(icd_text.strip())

            # 2) Query 上半部分：使用“无答案版” prompt
            if output_column not in query_sample:
                raise ValueError(
                    f"output_column '{output_column}' not found in query_sample keys: {list(query_sample.keys())}"
                )
            query_upper = pt.generate_text_for_embedding(
                query_sample,
                output_column=output_column,
            ).strip()

            # 拼接左侧完整文本：前面的 ICD + query 上半部分
            left_parts = left_icd_texts + [query_upper]
            left_text = sep.join(left_parts)

            # 3) 中间答案部分：直接从 query_sample[output_column] 读取
            answer_val = query_sample[output_column]
            answer_text = str(answer_val)

            # 4) 右侧 ICD：如果存在，则使用完整 ICD prompt，否则为 None
            right_text = None
            if right_icds:
                right_icd_texts: List[str] = []
                for s in right_icds:
                    icd_text = pt.generate_ice_item(s)
                    right_icd_texts.append(icd_text.strip())
                right_text = sep.join(right_icd_texts)

            return left_text, answer_text, right_text

    else:
        # 没有 PromptTemplate 时，退回到老的 GSM8K 风格硬编码逻辑
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

        def build_prompt_parts(
            left_icds: List[Dict],
            query_sample: Dict,
            right_icds: List[Dict],
        ) -> Tuple[str, str, Optional[str]]:
            """
            兼容旧逻辑：根据 left ICD、query、right ICD 构建 prompt 的三个部分：
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
        # 打分详情已去掉默认 debug 输出，避免刷屏；需要时可把下方取消注释或设置 hydra.verbose=DEBUG
        # logger.debug(
        #     f"Candidate at position {position} (insert_pos={insert_pos}): "
        #     f"score_after={score_after:.4f}, infoscore={infoscore:.4f}"
        # )
    
    return torch.tensor(scores, dtype=torch.float32)

#将原本放到中间的json放到后面
def fix_icd_json_query_last(json_path: str, out_path: Optional[str] = None) -> str:
    """
    将「query 在中间」的 ICD 生成 JSON 转为「query 在最后」的版本，并保存为新文件（不覆盖原文件）。
    
    原 JSON 格式：每个 key 为 anchor（query）的 id；value 含 id_list（每条为索引序列，其中一项为 query）
    和 score_list。序列中 query 可能在中间，例如 [icd1, icd2, query, icd3, icd4]。
    
    本函数对每条 id_list 中的序列重排为：保持其他 id 相对顺序不变，把 query（即 key 对应的 id）
    移到序列末尾，即 [icd1, icd2, icd3, icd4, query]。score_list 及其余字段不变。
    
    Args:
        json_path: 原始 JSON 文件路径（如 generated_data/xxx-mc_num:1-coarse_k:200-lambda:0.1.json）
        out_path: 输出路径；若为 None，则在原路径的 .json 前插入 "-fix"，得到 xxx-fix.json
    
    Returns:
        实际写入的路径（out_path 或自动生成的 -fix 路径）
    
    Example:
        fix_icd_json_query_last("generated_data/mmlu-...-lambda:0.1.json")
        -> 生成 "generated_data/mmlu-...-lambda:0.1-fix.json"，原文件不改动
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if out_path is None:
        if not json_path.endswith(".json"):
            out_path = json_path.rstrip() + "-fix.json"
        else:
            out_path = json_path[:-5] + "-fix.json"
    
    fixed = {}
    for anchor_key, record in data.items():
        anchor_id = int(anchor_key)
        id_list = record.get("id_list", [])
        score_list = record.get("score_list", [])
        new_id_list = []
        for seq in id_list:
            # 保持相对顺序：所有不等于 query 的 id 按原序，最后接 query
            others = [x for x in seq if int(x) != anchor_id]
            query_at_end = others + [anchor_id]
            new_id_list.append(query_at_end)
        fixed[anchor_key] = {
            "id_list": new_id_list,
            "score_list": score_list,
        }
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    
    logger.info(f"fix_icd_json_query_last: 已写入 query 置尾的 JSON -> {out_path}（原文件未修改）")
    return out_path

#之前已经生成的pt文件可以用到第一个分类器中,这里就可以直接把之前的pt文件转化成一个字典,方便第二阶段进行相关训练
def sampler_cache_to_embedding_dict(
    cache_path: str,
    out_path: Optional[str] = None,
    map_location: str = "cpu",
) -> str:
    """
    将 sampler 的 Qwen 特征缓存（(N, D) 的 tensor）转成 train.py 可用的 embedding_dict（{i: vector}）。

    Sampler 的 cache 文件（如 generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt）
    存的是 shape=(N, 2560) 的 tensor，第 i 行对应 train_ds[i] 的文本 embedding。
    train.py 的 _load_embedding_dict 期望的是 dict[int, Tensor]，本函数完成转换并另存。

    Args:
        cache_path: sampler 特征缓存 .pt 路径（内容为 tensor (N, D)）
        out_path: 输出 .pt 路径；若为 None，则在 cache_path 同目录下生成 xxx_embedding_dict.pt
        map_location: torch.load 的 map_location（默认 "cpu"）

    Returns:
        实际写入的路径。

    Example:
        sampler_cache_to_embedding_dict(
            "generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt",
            "generated_icd_data/qwen3_embeddings_mmlu_from_cache.pt",
        )
        然后在 train 时设 embedding_path=generated_icd_data/qwen3_embeddings_mmlu_from_cache.pt
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Sampler cache not found: {cache_path}")

    cache = torch.load(cache_path, map_location=map_location)
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Expected tensor in {cache_path}, got {type(cache)}")

    embedding_dict = {i: cache[i] for i in range(cache.shape[0])}

    if out_path is None:
        base, ext = os.path.splitext(cache_path)
        out_path = base + "_embedding_dict" + (ext or ".pt")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(embedding_dict, out_path)
    logger.info(
        f"sampler_cache_to_embedding_dict: 已转换 {cache.shape[0]} 条 -> {out_path}"
    )
    return out_path


def get_lever_lm_path(ckpt_dir: str, default_cpk_key: str = "min_vl") -> str:
    """
    从 checkpoint 目录获取 LeverLM 权重路径。

    Args:
        ckpt_dir: checkpoint 目录，如 generated_icd_data/model_cpk/mmlu/debug
        default_cpk_key: "min_vl" | "min_tl" | "last"，优先使用 min_vl（val_loss 最优）

    Returns:
        checkpoint 文件路径
    """
    import glob
    if default_cpk_key == "min_vl":
        pattern = os.path.join(ckpt_dir, "min_vl-*.ckpt")
    elif default_cpk_key == "min_tl":
        pattern = os.path.join(ckpt_dir, "min_tl-*.ckpt")
    elif default_cpk_key == "last":
        path = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(path):
            return path
        pattern = os.path.join(ckpt_dir, "last*.ckpt")
    else:
        raise ValueError(f"default_cpk_key must be min_vl|min_tl|last, got {default_cpk_key}")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")
    return max(candidates, key=os.path.getmtime)


def init_lever_lm(
    cfg,
    ckpt_path: str,
    train_ds,
    embedding_path: str,
    device: str = "cuda:0",
):
    """
    初始化训练好的 LeverLM 模型，用于 ICD 生成。

    Args:
        cfg: 配置（需含 train.lever_lm, train.lever_lm.input_dim 等）
        ckpt_path: checkpoint 路径
        train_ds: 训练集（用于 index_ds_size）
        embedding_path: Qwen embedding 缓存路径（与训练时一致）
        device: 设备

    Returns:
        lever_lm: GPT2LeverLM 实例，已加载权重
        embedding_dict: {idx: tensor} 供 LeverLMRetriever 使用（可选）
    """
    from train import _load_embedding_dict, LeverLM
    import hydra

    index_ds_size = len(train_ds)
    embedding_dict = _load_embedding_dict(embedding_path)
    input_dim = cfg.train.lever_lm.get("input_dim", 2560)
    emb_lookup = torch.zeros(index_ds_size + 3, input_dim, dtype=torch.float32)
    for i in range(index_ds_size):
        if i in embedding_dict:
            v = embedding_dict[i]
            v = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
            emb_lookup[i] = v.squeeze().float()

    lever_lm = hydra.utils.instantiate(
        cfg.train.lever_lm,
        index_ds_size=index_ds_size,
        embedding_lookup=emb_lookup,
    )
    model = LeverLM(
        lever_lm,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-3),
        warm_steps=cfg.get("warm_steps", 0.05),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    lever_lm = model.lever_lm.to(device)
    return lever_lm, embedding_dict
