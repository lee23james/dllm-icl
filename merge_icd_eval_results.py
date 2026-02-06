#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并多个 shard 评估结果，计算整体准确率。

设计思路：
- 每个 shard 评估一次，会产生一个 result.txt（由 eval_gsm8k 写入）
- 该文件中包含一行 "Total Accuracy:X"
- shard 的样本数 = 该 shard 对应 JSON 里 anchor 的数量（len(dict))

整体准确率 = sum(accuracy_i * n_i) / sum(n_i)

用法示例：

1) 两个 shard：
   python3 merge_icd_eval_results.py \\
       --json_files shard0.json shard1.json \\
       --result_files shard0_result.txt shard1_result.txt

注意：
- json_files 和 result_files 的顺序必须一一对应。
"""

import argparse
import json
from typing import List, Tuple


def parse_accuracy_from_result(result_path: str) -> float:
    """
    从 result.txt 中解析最后一个 Total Accuracy 的数值。
    """
    last_acc = None
    with open(result_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Total Accuracy:"):
                try:
                    last_acc = float(line.split("Total Accuracy:")[-1].strip())
                except ValueError:
                    continue
    if last_acc is None:
        raise ValueError(f"未在结果文件中找到 Total Accuracy: {result_path}")
    return last_acc


def get_num_anchors(json_path: str) -> int:
    """
    返回 JSON 中 anchor 的数量（即 top-level key 的数量）。
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 格式异常（顶层不是字典）: {json_path}")
    return len(data)


def merge_results(
    json_files: List[str],
    result_files: List[str],
) -> Tuple[float, List[Tuple[str, float, int]]]:
    """
    合并多个 (json, result) 对应的准确率。

    Returns:
        overall_accuracy, [(json_path, acc_i, n_i), ...]
    """
    if len(json_files) != len(result_files):
        raise ValueError("json_files 和 result_files 数量不一致")

    per_shard: List[Tuple[str, float, int]] = []
    total_weighted = 0.0
    total_n = 0

    for j_path, r_path in zip(json_files, result_files):
        n_i = get_num_anchors(j_path)
        acc_i = parse_accuracy_from_result(r_path)
        per_shard.append((j_path, acc_i, n_i))
        total_weighted += acc_i * n_i
        total_n += n_i

    if total_n == 0:
        raise ValueError("总样本数为 0，无法计算整体准确率")

    overall = total_weighted / total_n
    return overall, per_shard


def main():
    parser = argparse.ArgumentParser(description="合并多个 shard 的 ICD 评估结果，计算整体准确率")
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        required=True,
        help="shard JSON 文件列表（与 result_files 一一对应）",
    )
    parser.add_argument(
        "--result_files",
        type=str,
        nargs="+",
        required=True,
        help="对应的 result.txt 文件列表（与 json_files 一一对应）",
    )
    args = parser.parse_args()

    overall, per_shard = merge_results(args.json_files, args.result_files)

    print("=" * 80)
    print("Shard 评估结果：")
    print("=" * 80)
    for j_path, acc_i, n_i in per_shard:
        print(f"JSON: {j_path}")
        print(f"  样本数: {n_i}")
        print(f"  准确率: {acc_i:.4f}")
        print("-" * 80)

    print("\n整体结果：")
    print(f"  总样本数: {sum(n for _, _, n in per_shard)}")
    print(f"  加权整体准确率: {overall:.4f}")


if __name__ == "__main__":
    main()

