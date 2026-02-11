"""
MMLU 评估指标

假设：
- 通过 `load_mmlu_ds` 加载后的样本，`answer` 字段已经是 "A"/"B"/"C"/"D"
- 模型生成的文本中，只要包含正确选项字母（如 "A" 或 "A." 或 "Answer: A" 等）即认为预测正确
"""

import re
from typing import Dict, List, Any


class MMLUMetrics:
    """
    MMLU 任务的评估指标
    """

    @staticmethod
    def extract_choice(text: str) -> str:
        """
        从生成文本中提取模型预测的选项（A/B/C/D）

        规则（按优先级）：
        1. 匹配像 "Answer: A"、"答案是 C" 这类带前缀的形式（大小写不敏感）
        2. 匹配独立的大写字母选项（A/B/C/D），如单独一行的 "A."、"(B)"、"C )"
        3. 如果找不到，就返回空字符串，让上层逻辑用更宽松的包含判断
        """
        # 1) 明确带前缀的答案形式（Answer: A / answer is B / 选项 C 等）
        # 常见英文形式
        prefix_pattern = re.compile(
            r"(?:answer\s*(?:is|:)?\s*|\boption\s*[:\-]?\s*|\bchoice\s*[:\-]?\s*)"
            r"([ABCD])\b",
            flags=re.IGNORECASE,
        )
        m = prefix_pattern.search(text)
        if m:
            return m.group(1).upper()

        # 2) 退而求其次：找独立的大写 A/B/C/D（带简单标点）
        #   例如：
        #   "A.", "(B)", "C )", " D "
        standalone_pattern = re.compile(r"\b([ABCD])\b")
        m2 = standalone_pattern.search(text)
        if m2:
            return m2.group(1).upper()

        # 找不到就返回空串，交给上层作宽松包含判断
        return ""

    @staticmethod
    def is_correct(generated_text: str, ground_truth: str) -> bool:
        """
        判断预测是否正确。

        Args:
            generated_text: 模型生成的完整文本
            ground_truth: 真实答案字母（"A"/"B"/"C"/"D"），来自样本的 answer 字段

        Returns:
            bool: 是否预测正确
        """
        truth = (ground_truth or "").strip().upper()
        if truth not in {"A", "B", "C", "D"}:
            # 如果 ground_truth 不在 {A,B,C,D}，直接返回 False，避免误判
            return False

        # 先尝试结构化提取
        pred_choice = MMLUMetrics.extract_choice(generated_text)
        if pred_choice:
            return pred_choice == truth

        # 如果结构化提取失败，使用宽松规则：
        # 只要生成文本中包含正确选项字母（忽略大小写），就视为正确。
        # 注意：这里可能会有少量噪声（比如长文本里恰好有一个 "A"），
        # 但在常见 MMLU 格式下通常是可以接受的。
        pattern = re.compile(rf"\b{re.escape(truth)}\b", flags=re.IGNORECASE)
        return bool(pattern.search(generated_text))

    @staticmethod
    def evaluate_single(generated_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        评估单个样本

        Args:
            generated_text: 模型生成的完整文本
            ground_truth: 真实答案字母（"A"/"B"/"C"/"D"）

        Returns:
            dict: {
                "predicted": <提取出的预测选项或空串>,
                "ground_truth": <标准答案字母>,
                "is_correct": <是否正确>,
                "generated_text": <原始生成文本>,
            }
        """
        pred_choice = MMLUMetrics.extract_choice(generated_text)
        truth = (ground_truth or "").strip().upper()
        correct = MMLUMetrics.is_correct(generated_text, truth)

        return {
            "predicted": pred_choice,
            "ground_truth": truth,
            "is_correct": correct,
            "generated_text": generated_text,
        }

    @staticmethod
    def evaluate_batch(
        generated_texts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        批量评估
        """
        assert len(generated_texts) == len(
            ground_truths
        ), f"Length mismatch: {len(generated_texts)} vs {len(ground_truths)}"

        results: List[Dict[str, Any]] = []
        correct_count = 0

        for gen_text, gt in zip(generated_texts, ground_truths):
            r = MMLUMetrics.evaluate_single(gen_text, gt)
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

