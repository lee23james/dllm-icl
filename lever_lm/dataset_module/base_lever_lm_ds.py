#这里替换成了Qwen的,进行离线推理,同时把N多模态换成了纯NLP
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class BaseLeverLMDataset(Dataset):
    """
    适配纯 NLP、离线 Qwen-Embedding 的基础 Dataset。

    与原版多模态 CLIP 版本的区别：
    - 不再从 `index_ds` 中读取图片 / 文本然后在线编码；
    - 改为在 __init__ 里传入预先算好的 `embedding_dict`，
      __getitem__ 直接查表拿向量，避免在线调用 Qwen-Embedding。

    约定：
    - data 结构与原来一致：
        data[\"icd_seq\"]: List[List[int]]，每条为 [icd_1, icd_2, ..., query_id]
        data[\"icd_score\"]: List[float]，与 icd_seq 一一对应的分数
    - embedding_dict: Dict[int, Union[torch.Tensor, list]]
        key   = query_id（即 icd_seq 中最后那个 id）
        value = 预计算好的向量（Qwen-Embedding-4B 的 embedding）
    """

    def __init__(
        self,
        data: Dict,
        embedding_dict: Dict[int, Union[torch.Tensor, List[float]]],
        threshold: float = 0.0,
        reverse_seq: bool = False,
        emb_dtype: torch.dtype = torch.float32,
        emb_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.reverse_seq = reverse_seq

        # 预计算好的 query 向量表
        self.embedding_dict = embedding_dict
        self.emb_dtype = emb_dtype
        self.emb_device = emb_device

        # 保存每条样本的 ICD 序列（不含 query 本身）和对应的 query id
        self.icd_idx_seq_list: List[List[int]] = []
        self.query_id_list: List[int] = []

        icd_seq_list = data["icd_seq"]
        icd_score_list = data["icd_score"]

        for icd_seq, icd_score in zip(icd_seq_list, icd_score_list):
            # 按阈值过滤低分样本
            if icd_score < self.threshold:
                continue

            # 约定 icd_seq 的最后一个 id 是 query 的 id
            if not icd_seq:
                continue

            query_id = int(icd_seq[-1])
            idx_list = icd_seq[:-1]

            if self.reverse_seq:
                idx_list = list(reversed(idx_list))

            # 仅保留我们在 embedding_dict 里有向量的样本
            if query_id not in self.embedding_dict:
                # 如果你希望严格检查，可以改成 raise KeyError
                continue

            self.icd_idx_seq_list.append(list(idx_list))
            self.query_id_list.append(query_id)

    def __len__(self) -> int:
        return len(self.query_id_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        返回：
        - query_embedding: 预计算好的 Qwen embedding（Tensor）
        - icd_seq_idx:     ICD 序列的 index（不含 query，自左到右，或反转后）
        """
        icd_seq_idx = self.icd_idx_seq_list[index]
        query_id = self.query_id_list[index]

        # 查表拿到预计算好的向量
        emb = self.embedding_dict[query_id]
        dtype = self.emb_dtype if isinstance(self.emb_dtype, torch.dtype) else torch.float32
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=dtype)
        else:
            if emb.dtype != dtype:
                emb = emb.to(dtype=dtype)

        if self.emb_device is not None:
            emb = emb.to(device=self.emb_device)

        icd_seq_idx_tensor = torch.tensor(icd_seq_idx, dtype=torch.long)
        #这里一定要多加考虑
        ## 旧：直接生成了完整的训练序列 add_sp_token_seq_idx = [BOS, QUERY] + 
        # icd_seq_idx + [EOS] return { "icd_seq_idx": tensor(add_sp_token_seq_idx) }
        return {
            "query_embedding": emb,
            "icd_seq_idx": icd_seq_idx_tensor,
        }

