"""
SUBJ 评估指标

假设：
- 通过 load_ds(cfg, split=...) 加载后的样本，answer 字段已经是
  label_mapping 映射后的文本，比如 "subjective" / "objective"
- 评估规则：只要真实标签文本（忽略大小写、前后空格）出现在模型生成的文本里，
  就认为预测正确（与原 eval_utils.py 里 eval_emotion 的思路一致）
"""

from typing import Dict, List, Any


class SubjMetrics:
    """
    SUBJ 任务的评估指标（简单基于包含关系的准确率）
    """

    @staticmethod
    def normalize_label(label: str) -> str:
        """
        标准化标签：去掉前后空格，小写化
        """
        return (label or "").strip().lower()

    @staticmethod
    def is_correct(generated_text: str, ground_truth: str) -> bool:
        """
        判断预测是否正确。

        Args:
            generated_text: 模型生成的完整文本
            ground_truth: 真实标签文本（如 'subjective' / 'objective'）

        Returns:
            bool: 是否预测正确
        """
        gt = SubjMetrics.normalize_label(ground_truth)
        if not gt:
            return False

        text = (generated_text or "").lower()
        return gt in text

    @staticmethod
    def evaluate_single(
        generated_text: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        评估单个样本，返回与 GSM8K/MMLU 相同风格的结构
        """
        correct = SubjMetrics.is_correct(generated_text, ground_truth)

        return {
            "predicted": generated_text,    # 这里不做结构化抽取，直接保留原文
            "ground_truth": ground_truth,
            "is_correct": correct,
            "generated_text": generated_text,
        }

    @staticmethod
    def evaluate_batch(
        generated_texts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        批量评估，输出 accuracy / correct_count / total_count
        """
        assert len(generated_texts) == len(ground_truths), \
            f"Length mismatch: {len(generated_texts)} vs {len(ground_truths)}"

        results: List[Dict[str, Any]] = []
        correct_count = 0

        for gen_text, gt in zip(generated_texts, ground_truths):
            r = SubjMetrics.evaluate_single(gen_text, gt)
            results.append(r)
            if r["is_correct"]:
                correct_count += 1

        total = len(results)
        accuracy = correct_count / total if total > 0 else 0.0

        return {
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total,
        }