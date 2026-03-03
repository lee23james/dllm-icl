"""
TREC 评估指标（6 类问句类型分类）

标签集合（文本形式）：
- description
- entity
- expression
- human
- location
- number

评测规则（exact-match）：
- 从模型生成的文本中抽取上述标签之一作为预测标签 predicted
- 对真实标签 ground_truth 做相同的标准化
- 若 predicted == ground_truth，则认为预测正确
"""

from typing import Dict, List, Any, Tuple


class TrecMetrics:
    """
    TREC 任务的评估指标（基于标签精确匹配的准确率）
    """

    # 允许的标签集合（文本形式）
    LABELS: Tuple[str, ...] = (
        "description",
        "entity",
        "expression",
        "human",
        "location",
        "number",
    )

    @staticmethod
    def _normalize(text: str) -> str:
        """
        标准化文本：小写 + 去掉首尾空格 + 压缩中间空白
        """
        if not text:
            return ""
        text = text.strip().lower()
        # 将连续空白压缩为单个空格
        return " ".join(text.split())

    @classmethod
    def extract_label(cls, generated_text: str) -> str:
        """
        从模型生成的文本中抽取一个 TREC 标签。

        策略：
        - 统一小写
        - 对每个标签，取最后一次出现的位置
        - 在所有标签中选择“位置最大”的那一个（最靠后的匹配）
        - 若没有任何标签出现，则返回空字符串
        """
        if not generated_text:
            return ""

        text_lower = generated_text.lower()
        best_label = ""
        best_pos = -1

        for label in cls.LABELS:
            pos = text_lower.rfind(label)
            if pos == -1:
                continue
            # 简单边界检查：前后不是字母/数字，防止匹配到更长单词的一部分
            before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
            after_idx = pos + len(label)
            after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
            if not (before_ok and after_ok):
                continue
            if pos > best_pos:
                best_pos = pos
                best_label = label

        return best_label

    @classmethod
    def evaluate_single(
        cls,
        generated_text: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        评估单个样本。

        Args:
            generated_text: 模型生成的完整文本
            ground_truth: 真实标签文本（如 'description' / 'entity' / ...）
        """
        pred_label = cls.extract_label(generated_text)
        gt_label = cls._normalize(ground_truth)

        is_correct = False
        if pred_label:
            is_correct = cls._normalize(pred_label) == gt_label

        return {
            "predicted": pred_label,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "generated_text": generated_text,
        }

    @classmethod
    def evaluate_batch(
        cls,
        generated_texts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        批量评估，输出 accuracy / correct_count / total_count
        """
        assert len(generated_texts) == len(ground_truths), (
            f"Length mismatch: {len(generated_texts)} vs {len(ground_truths)}"
        )

        results: List[Dict[str, Any]] = []
        correct_count = 0

        for gen_text, gt in zip(generated_texts, ground_truths):
            r = cls.evaluate_single(gen_text, gt)
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

