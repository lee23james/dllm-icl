"""
InsertionSelector 训练数据：从 Stage1 JSON 解析样本列表，划分 train/val；
并提供与 sampler 一致的 ceval/cmmlu 文本格式化（供预计算嵌入使用）。
方案 A：Dataset 仅索引预计算好的 .pt 嵌入矩阵。
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset

from loguru import logger


def format_ceval_cmmlu_sample(sample: Dict[str, Any]) -> str:
    """
    将 C-Eval/C-MMLU 单条样本格式化为用于 Qwen 嵌入的文本（与 text_sim_qwen_mmr_sampler 的
    MMLU/ceval 分支一致：question + A. choice_a ... D. choice_d）。
    """
    for key in ("question", "choice_a", "choice_b", "choice_c", "choice_d"):
        if key not in sample:
            raise ValueError(
                f"Sample missing required key '{key}' for ceval/cmmlu formatting. "
                f"Keys: {list(sample.keys())}"
            )
    question = str(sample["question"]).strip()
    options_text = (
        f"A. {sample['choice_a']}\n"
        f"B. {sample['choice_b']}\n"
        f"C. {sample['choice_c']}\n"
        f"D. {sample['choice_d']}"
    )
    return f"{question}\n{options_text}".strip()


def format_subj_sample(sample: Dict[str, Any]) -> str:
    """
    将 Subj 单条样本格式化为用于 Qwen 嵌入的文本。
    必须使用映射后的 answer（subjective/objective），与 task 模板一致。
    """
    for key in ("sentence", "answer"):
        if key not in sample:
            raise ValueError(
                f"Sample missing required key '{key}' for subj formatting. "
                f"Keys: {list(sample.keys())}"
            )
    sentence = str(sample["sentence"]).strip()
    answer = str(sample["answer"]).strip()
    return f"Input: {sentence}\nType: {answer}".strip()


def parse_json_to_samples(
    json_path: str | Path,
) -> List[Tuple[int, List[int], int, float]]:
    """
    从 Stage1 生成的 JSON 解析出所有训练样本。

    JSON 结构：{ "anchor_id": { "id_list": [ [id1, id2, ..., anchor_id], ... ], "score_list": [ ... ] } }
    支持 few_shot=3（序列长 4：3 ICD + anchor）或 few_shot=4（序列长 5：4 ICD + anchor）。
    anchor 在序列中的 0-based 位置即插入位置标签，与模型 gap 索引一致：
    label 0 = 插在第一个 ICD 前，1 = 插在 1/2 之间，…，n = 插在第 n 个 ICD 后。

    Returns:
        List of (anchor_idx, ordered_icd_indices, label, score).
        anchor_idx / ordered_icd_indices 为题库 parquet 行号（0-based）；
        label 为 0-based 插入位置，与 InsertionSelector 的 logits 维度 0..num_gaps-1 一一对应。
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Tuple[int, List[int], int, float]] = []
    for anchor_id_str, entry in data.items():
        anchor_id = int(anchor_id_str)
        id_list = entry.get("id_list", [])
        score_list = entry.get("score_list", [])
        if not id_list:
            continue
        n = min(len(id_list), len(score_list)) if score_list else len(id_list)
        for i in range(n):
            seq = id_list[i]
            score = float(score_list[i]) if i < len(score_list) else 0.0
            # 支持序列长 4（3 ICD + anchor）或 5（4 ICD + anchor）
            if len(seq) not in (4, 5):
                continue
            try:
                pos = seq.index(anchor_id)
            except ValueError:
                continue
            ordered_icd = [x for x in seq if x != anchor_id]
            if len(ordered_icd) not in (3, 4):
                continue
            samples.append((anchor_id, ordered_icd, pos, score))

    logger.info(f"Parsed {len(samples)} samples from {path}")
    return samples


def train_val_split(
    samples: List[Tuple[int, List[int], int, float]],
    val_ratio: float = 0.2,
    seed: int = 42,
    by_anchor: bool = True,
) -> Tuple[
    List[Tuple[int, List[int], int, float]],
    List[Tuple[int, List[int], int, float]],
]:
    """
    划分训练集与验证集。by_anchor=True 时按 anchor 划分，避免同一 anchor 同时出现在 train 和 val。
    """
    rng = random.Random(seed)
    if by_anchor:
        anchor_to_samples: Dict[int, List[Tuple[int, List[int], int, float]]] = {}
        for s in samples:
            anchor_idx = s[0]
            anchor_to_samples.setdefault(anchor_idx, []).append(s)
        anchors = list(anchor_to_samples.keys())
        rng.shuffle(anchors)
        n_val = max(1, int(len(anchors) * val_ratio))
        val_anchors = set(anchors[-n_val:])
        train_samples = []
        val_samples = []
        for a, lst in anchor_to_samples.items():
            if a in val_anchors:
                val_samples.extend(lst)
            else:
                train_samples.extend(lst)
    else:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(samples) * val_ratio))
        val_indices = set(indices[-n_val:])
        train_samples = [samples[i] for i in indices if i not in val_indices]
        val_samples = [samples[i] for i in indices if i in val_indices]

    logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}")
    return train_samples, val_samples


class InsertionSelectorDataset(Dataset):
    """
    方案 A：仅从预计算的 [N_train, emb_dim] .pt 中按样本下标索引，得到 query_embed 与 icd_embeds。
    不加载 Qwen，训练时只读缓存。
    支持 few_shot=3（icd_embeds [3, emb_dim]，label 0～3）与 few_shot=4（[4, emb_dim]，label 0～4）。
    """

    def __init__(
        self,
        samples: List[Tuple[int, List[int], int, float]],
        embedding_cache_path: Union[str, Path],
    ):
        """
        Args:
            samples: List of (anchor_idx, ordered_icd_indices, label, score).
            embedding_cache_path: Path to .pt of shape [N_train, emb_dim].
        """
        self.samples = samples
        path = Path(embedding_cache_path)
        if not path.is_absolute():
            # 若为相对路径，假定相对于当前工作目录；调用方通常从项目根运行
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding cache not found: {path}. Run scripts/precompute_insertion_embedding.py first."
            )
        self.embeddings = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(self.embeddings, torch.Tensor) or self.embeddings.dim() != 2:
            raise ValueError(
                f"Expected embedding cache to be 2D tensor, got {type(self.embeddings)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_idx, ordered_icd_indices, label, _score = self.samples[idx]
        # label 与 JSON 中 pos=seq.index(anchor_id) 一致：0-based 插入位置（0=第一个 ICD 前，n=第 n 个 ICD 后）
        query_embed = self.embeddings[anchor_idx].clone()
        icd_embeds = self.embeddings[ordered_icd_indices].clone()  # [3, emb_dim] 或 [4, emb_dim]
        return {
            "query_embed": query_embed,
            "icd_embeds": icd_embeds,
            "labels": torch.tensor(label, dtype=torch.long),
        }
