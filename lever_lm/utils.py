import json
import os
from typing import List, Tuple, Dict, Any, Optional

import faiss
import more_itertools
import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
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


def format_mmlu_prompt(item: Dict[str, Any]) -> str:
    """
    将 MMLU 数据项格式化为纯文本 Prompt（仅包含 Question + Options，不含 Answer）
    
    注意：此函数已弃用，建议使用 PromptTemplate.generate_text_for_embedding() 或
    TextSimQwenMMRSampler._format_text() 方法，它们与 PromptTemplate 逻辑保持一致。
    
    Args:
        item: MMLU 数据项，包含 'question' 和 'choices' 字段
    
    Returns:
        格式化后的文本字符串
    """
    question = item.get("question", "").strip()
    choices = item.get("choices", [])
    
    if not choices:
        # 如果没有 choices，尝试从其他字段获取
        choices = item.get("options", [])
    
    # 构建选项文本
    options_text = ""
    if len(choices) >= 4:
        options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    elif len(choices) > 0:
        option_labels = ["A", "B", "C", "D", "E", "F"][:len(choices)]
        options_text = "\n".join([f"{label}. {choice}" for label, choice in zip(option_labels, choices)])
    
    # 组合问题和选项
    prompt = f"{question}\n{options_text}".strip()
    return prompt


@torch.inference_mode()
def encode_text_qwen(
    text_list: List[str],
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "cuda:0",
    batch_size: int = 64,
    normalize: bool = True,
) -> torch.Tensor:
    """
    使用 Qwen-Embedding 模型对文本列表进行编码
    
    Args:
        text_list: 文本列表
        model_name: Qwen 模型名称，推荐使用 Qwen3-Embedding-4B 或 Qwen/Qwen2.5-0.5B-Instruct
        device: 设备字符串（如 "cuda:0"）
        batch_size: 批处理大小
        normalize: 是否对向量进行 L2 归一化
    
    Returns:
        features: torch.Tensor, shape=(N, D)，其中 N 是文本数量，D 是向量维度
    """
    # 简化实现：强制使用 SentenceTransformer 加载 Qwen3-Embedding。
    # 不做 transformers 版本的硬限制，只有在实际出现 qwen3 相关错误时再提示升级。
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:  # pragma: no cover - 极老环境
        import importlib_metadata  # type: ignore

    # 检查 sentence-transformers 是否可用（仍按官方建议做版本检查）
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            st_ver = importlib_metadata.version("sentence-transformers")
        except Exception:
            st_ver = "0.0.0"
    except ImportError as e:
        raise ImportError(
            "需要安装 sentence-transformers>=2.7.0 才能使用 Qwen3-Embedding 作为 SentenceTransformer。\n"
            "请执行: pip install 'sentence-transformers>=2.7.0'"
        ) from e

    def _parse_ver(v: str) -> tuple:
        parts = v.split(".")
        return tuple(int(p) for p in parts[:3] if p.isdigit())

    if _parse_ver(st_ver) < _parse_ver("2.7.0"):
        raise RuntimeError(
            f"Qwen3-Embedding 需要 sentence-transformers>=2.7.0，当前为 {st_ver}。"
            "请执行: pip install 'sentence-transformers>=2.7.0'"
        )

    try:
        import os

        logger.info(f"Loading SentenceTransformer model: {model_name} on {device}")

        # 临时关闭 tqdm 进度条，避免在加载权重时打印 "Loading weights: 100% ..."。
        prev_tqdm_disable = os.environ.get("TQDM_DISABLE", None)
        os.environ["TQDM_DISABLE"] = "1"
        try:
            # 只设置 padding_side，不传 attn_implementation 等，使用库默认行为
            model = SentenceTransformer(
                model_name,
                device=device,
                tokenizer_kwargs={"padding_side": "left"},
            )
        finally:
            # 恢复环境变量
            if prev_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm_disable

        logger.info(
            f"Encoding {len(text_list)} texts with SentenceTransformer, "
            f"batch_size={batch_size}..."
        )
        embeddings = model.encode(
            text_list,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,  # 关闭编码时的进度条输出
        )
    except Exception as e:
        msg = str(e)
        # 如果底层是 transformers 不认识 qwen3 之类的问题，再提示升级 transformers
        if "qwen3" in msg.lower() or "KeyError: 'qwen3'" in msg or "KeyError(\"qwen3" in msg:
            raise RuntimeError(
                "检测到底层 transformers 对 `qwen3` 架构不兼容，"
                "建议在当前环境中执行: pip install 'transformers>=4.51.0'"
            ) from e
        raise

    features = embeddings.to(torch.float32).cpu()
    logger.info(f"Encoded {len(text_list)} texts into features with shape {features.shape}")
    return features


def recall_and_rerank_mmr(
    query_vecs: torch.Tensor,
    candidate_vecs: torch.Tensor,
    top_k: int = 64,
    coarse_k: int = 200,
    lambda_param: float = 0.1,
    exclude_indices: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    两阶段检索：先用相似度粗筛，再用 MMR 细筛
    
    Args:
        query_vecs: Query 向量，shape=(N_query, D)
        candidate_vecs: 候选向量池，shape=(N_candidate, D)
        top_k: 最终返回的 top-k 数量
        coarse_k: 粗筛阶段的 top-k 数量（应该 >= top_k）
        lambda_param: MMR 的 lambda 参数，控制相似度和多样性的平衡
                      lambda=0: 只考虑相似度
                      lambda=1: 只考虑多样性
        exclude_indices: 需要排除的索引列表（例如 anchor 本身）
    
    Returns:
        selected_indices: List[List[int]]，每个 query 对应的 top-k 候选索引列表
    """
    device = query_vecs.device
    query_vecs = query_vecs.to(torch.float32)
    candidate_vecs = candidate_vecs.to(torch.float32)
    
    # 确保向量已归一化（用于余弦相似度）
    query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
    candidate_vecs = torch.nn.functional.normalize(candidate_vecs, p=2, dim=1)
    
    N_query = query_vecs.shape[0]
    N_candidate = candidate_vecs.shape[0]
    
    # 排除索引集合
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    # 计算相似度矩阵：query_vecs @ candidate_vecs.T
    # shape: (N_query, N_candidate)
    similarity_matrix = query_vecs @ candidate_vecs.t()
    
    # 对每个 query 进行两阶段检索
    selected_indices_list = []
    
    for q_idx in range(N_query):
        sim_scores = similarity_matrix[q_idx]  # shape: (N_candidate,)
        
        # 阶段1：粗筛 - 选择 top coarse_k 个最相似的
        _, coarse_indices = torch.topk(sim_scores, k=min(coarse_k, N_candidate), dim=0)
        coarse_indices = coarse_indices.cpu().tolist()
        
        # 排除指定索引
        coarse_indices = [idx for idx in coarse_indices if idx not in exclude_set]
        
        if len(coarse_indices) == 0:
            # 如果没有候选，返回空列表
            selected_indices_list.append([])
            continue
        
        # 如果粗筛结果少于 top_k，直接返回
        if len(coarse_indices) <= top_k:
            selected_indices_list.append(coarse_indices[:top_k])
            continue
        
        # 阶段2：MMR 细筛
        # 从 coarse_indices 中选择 top_k 个，同时考虑相似度和多样性
        query_vec = query_vecs[q_idx:q_idx+1]  # shape: (1, D)
        coarse_candidate_vecs = candidate_vecs[coarse_indices]  # shape: (coarse_k, D)
        coarse_sim_scores = sim_scores[coarse_indices].cpu().numpy()  # shape: (coarse_k,)
        
        # MMR 算法
        selected = []
        remaining = list(range(len(coarse_indices)))
        
        # 第一步：选择相似度最高的
        first_idx = np.argmax(coarse_sim_scores)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # 迭代选择剩余 top_k-1 个
        for _ in range(min(top_k - 1, len(remaining))):
            if len(remaining) == 0:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for cand_idx in remaining:
                # 相似度分数（与 query 的相似度）
                sim_score = coarse_sim_scores[cand_idx]
                
                # 多样性分数：与已选样本的最大相似度（越小越好）
                if len(selected) > 0:
                    selected_vecs = coarse_candidate_vecs[selected]  # shape: (len(selected), D)
                    cand_vec = coarse_candidate_vecs[cand_idx:cand_idx+1]  # shape: (1, D)
                    max_sim_to_selected = (selected_vecs @ cand_vec.t()).max().item()
                else:
                    max_sim_to_selected = 0.0
                
                # MMR 分数：λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
                # 越大越好
                mmr_score = lambda_param * sim_score - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = cand_idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        # 将索引映射回原始候选池
        final_indices = [coarse_indices[idx] for idx in selected]
        selected_indices_list.append(final_indices)
    
    return selected_indices_list

