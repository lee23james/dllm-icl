from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import os

import torch
from datasets import Dataset
from loguru import logger

from lever_lm.utils import encode_text_qwen


class QwenTopkRetriever:
    """
    Top-k retriever based on Qwen-Embedding.

    - Index corpus: `index_ds` (typically the train set)
    - Query: encoded by the same Qwen-Embedding model
    - Returns the most similar `nshot` indices for each query
    - If `reverse_seq=True`, the returned order is from less similar to most similar,
      so the most similar example appears last.
    """

    def __init__(
        self,
        index_ds: Dataset,
        qwen_model_path: str,
        query_text_extractor: Callable[[Dict[str, Any]], str],
        nshot: int = 4,
        device: str = "cuda:0",
        encode_batch_size: int = 64,
        feature_cache: Optional[str] = None,
        overwrite_cache: bool = False,
        reverse_seq: bool = True,
        forbidden_indices: Optional[Set[int]] = None,
        query_task_description: Optional[str] = None,
    ) -> None:
        """
        Args:
            index_ds: dataset to be used as retrieval corpus (usually the full train set)
            qwen_model_path: Qwen-Embedding model path/name
            query_text_extractor: function mapping sample -> text for embedding
            nshot: number of ICD examples to retrieve per query
            device: device string, e.g. "cuda:0"
            encode_batch_size: batch size when encoding the index corpus
            feature_cache: optional path to cached embeddings for index_ds;
                           if present, will be loaded instead of recomputing
            overwrite_cache: if True, ignore existing cache and recompute
            reverse_seq: whether to reverse the final order so that
                         the most similar example appears last
            forbidden_indices: global set of indices that must never appear in ICD
            query_task_description: optional task description used to wrap query text as
                                    "Instruct: ...\\nQuery: <text>"; if None, use raw text
        """
        self.index_ds = index_ds
        self.qwen_model_path = qwen_model_path
        self.query_text_extractor = query_text_extractor
        self.nshot = int(nshot)
        self.device = torch.device(device)
        self.encode_batch_size = int(encode_batch_size)
        self.feature_cache = feature_cache
        self.overwrite_cache = bool(overwrite_cache)
        self.reverse_seq = bool(reverse_seq)
        self.forbidden_indices: Set[int] = set(forbidden_indices or set())
        self.index_ds_size = len(index_ds)
        # Default generic task description if not provided
        if query_task_description is None:
            self.query_task_description = (
                "Given a query, retrieve in-context examples that are most helpful for solving the query"
            )
        else:
            self.query_task_description = str(query_task_description)

        logger.info(
            "QwenTopkRetriever initialized: "
            f"nshot={self.nshot}, index_ds_size={self.index_ds_size}, "
            f"reverse_seq={self.reverse_seq}, "
            f"feature_cache={self.feature_cache}, "
            f"forbidden_size={len(self.forbidden_indices)}"
        )

        # Pre-compute / load embeddings for the index corpus
        self.index_features: torch.Tensor = self._encode_corpus()

    def _load_features_from_cache(self, path: str) -> Optional[torch.Tensor]:
        """Try to load features from cache; return None on failure."""
        try:
            if not os.path.exists(path):
                return None
            logger.info(f"Loading Qwen features cache from {path} ...")
            features = torch.load(path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            logger.info(f"Loaded cached features with shape {tuple(features.shape)}")
            return features
        except Exception as e:  # defensive
            logger.warning(f"Failed to load feature cache from {path}: {e}")
            return None

    @torch.inference_mode()
    def _encode_corpus(self) -> torch.Tensor:
        """
        Encode `index_ds` into embeddings, shape (N, D).
        If `feature_cache` is provided and exists, it will be reused.
        """
        features: Optional[torch.Tensor] = None

        if self.feature_cache and not self.overwrite_cache:
            features = self._load_features_from_cache(self.feature_cache)

        if features is None:
            logger.info(
                f"Encoding index_ds (size={self.index_ds_size}) with Qwen-Embedding model "
                f"'{self.qwen_model_path}' ..."
            )
            texts: List[str] = []
            for i in range(self.index_ds_size):
                sample = self.index_ds[i]
                text = self.query_text_extractor(sample)
                if not isinstance(text, str):
                    text = str(text)
                texts.append(text)

            features = encode_text_qwen(
                text_list=texts,
                model_name=self.qwen_model_path,
                device=str(self.device),
                batch_size=self.encode_batch_size,
                normalize=True,
            )

            if self.feature_cache:
                try:
                    os.makedirs(os.path.dirname(self.feature_cache), exist_ok=True)
                    logger.info(f"Saving Qwen features cache to {self.feature_cache} ...")
                    torch.save(features, self.feature_cache)
                except Exception as e:  # defensive
                    logger.warning(f"Failed to save feature cache to {self.feature_cache}: {e}")

        # Move features to target device
        features = features.to(self.device).to(torch.float32)
        return features

    def _format_query_text(self, sample: Dict[str, Any]) -> str:
        """Build the final query text (with optional Instruct prefix)."""
        base_text = self.query_text_extractor(sample)
        if not isinstance(base_text, str):
            base_text = str(base_text)

        if self.query_task_description:
            return f"Instruct: {self.query_task_description}\nQuery:{base_text}"
        return base_text

    def _postprocess_indices(self, sorted_indices: List[int], extra_exclude: Optional[List[int]]) -> List[int]:
        """
        Filter, truncate and optionally reverse the index list.
        - Exclude global `forbidden_indices` and `extra_exclude`
        - Keep at most `nshot`
        - If `reverse_seq=True`, reverse so that the most similar is last
        """
        exclude_set: Set[int] = set(extra_exclude or [])
        exclude_set |= self.forbidden_indices

        picked: List[int] = []
        for idx in sorted_indices:
            if idx in exclude_set:
                continue
            picked.append(int(idx))
            if len(picked) >= self.nshot:
                break

        if len(picked) < self.nshot:
            logger.warning(
                f"Only picked {len(picked)} candidates for nshot={self.nshot} "
                f"(after excluding {len(exclude_set)} indices)."
            )

        if self.reverse_seq:
            picked = list(reversed(picked))

        return picked

    @torch.inference_mode()
    def retrieve(
        self,
        test_sample: Dict[str, Any],
        exclude_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Retrieve top-k similar indices for a single test sample.
        """
        query_text = self._format_query_text(test_sample)

        query_feat = encode_text_qwen(
            text_list=[query_text],
            model_name=self.qwen_model_path,
            device=str(self.device),
            batch_size=1,
            normalize=True,
        )
        query_feat = query_feat.to(self.device).to(torch.float32)  # (1, D)

        # Cosine similarity (vectors are already normalized in encode_text_qwen)
        sim_scores = torch.matmul(query_feat, self.index_features.t()).squeeze(0)  # (N,)

        # Sort indices from high similarity to low
        sorted_indices = torch.argsort(sim_scores, descending=True).tolist()

        return self._postprocess_indices(sorted_indices, exclude_indices)

    @torch.inference_mode()
    def retrieve_batch(
        self,
        test_samples: List[Dict[str, Any]],
        exclude_indices_list: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """
        Batch version of `retrieve`.
        """
        if not test_samples:
            return []

        if exclude_indices_list is None:
            exclude_indices_list = [[] for _ in range(len(test_samples))]
        elif len(exclude_indices_list) != len(test_samples):
            raise ValueError(
                f"exclude_indices_list length ({len(exclude_indices_list)}) "
                f"!= test_samples length ({len(test_samples)})"
            )

        query_texts: List[str] = []
        for sample in test_samples:
            query_texts.append(self._format_query_text(sample))

        query_feats = encode_text_qwen(
            text_list=query_texts,
            model_name=self.qwen_model_path,
            device=str(self.device),
            batch_size=self.encode_batch_size,
            normalize=True,
        )
        query_feats = query_feats.to(self.device).to(torch.float32)  # (B, D)

        # Similarity matrix: (B, N)
        sim_matrix = torch.matmul(query_feats, self.index_features.t())

        # For each query row, sort and post-process
        results: List[List[int]] = []
        for row, extra_exclude in zip(sim_matrix, exclude_indices_list):
            sorted_indices = torch.argsort(row, descending=True).tolist()
            picked = self._postprocess_indices(sorted_indices, extra_exclude)
            results.append(picked)

        return results