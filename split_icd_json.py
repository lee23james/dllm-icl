#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 anchor 切分 ICD 结果 JSON，方便评估阶段多进程 / 多 GPU 并行。

示例：
  python3 split_icd_json.py \
    --input generated_icd_data/generated_data/xxx-sample_num:200-mc_num:128.json \
    --output_dir generated_icd_data/generated_data/shards \
    --num_shards 2
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def split_icd_json(
    input_path: str,
    output_dir: str,
    num_shards: int,
) -> List[Tuple[str, int, int]]:
    """
    将 ICD JSON 按 anchor 数量切成 num_shards 份。

    Returns:
        [(输出文件路径, 起始下标, 结束下标), ...]
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with open(input_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    # 按 anchor id 排序（按数值排序更直观）
    keys = sorted(data.keys(), key=lambda k: int(k))
    total = len(keys)
    if total == 0:
        raise ValueError(f"No anchors found in JSON: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    base = total // num_shards
    remainder = total % num_shards

    shards_info: List[Tuple[str, int, int]] = []
    start = 0
    base_name = os.path.basename(input_path).rsplit(".json", 1)[0]

    for shard_idx in range(num_shards):
        shard_size = base + (1 if shard_idx < remainder else 0)
        end = start + shard_size
        shard_keys = keys[start:end]

        shard_data = {k: data[k] for k in shard_keys}
        out_name = f"{base_name}_shard{shard_idx}_{start}_{end}.json"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w") as f:
            json.dump(shard_data, f, ensure_ascii=False, indent=2)

        shards_info.append((out_path, start, end))
        start = end

    return shards_info


def main():
    parser = argparse.ArgumentParser(description="按 anchor 切分 ICD JSON，用于多进程 / 多 GPU 评估")
    parser.add_argument("--input", type=str, required=True, help="输入 ICD JSON 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_shards", type=int, default=2, help="切分份数（一般等于 GPU 数量）")
    args = parser.parse_args()

    shards = split_icd_json(args.input, args.output_dir, args.num_shards)

    print("切分完成：")
    for path, start, end in shards:
        print(f"  {path}  (anchors[{start}:{end}), 共 {end - start} 条)")


if __name__ == "__main__":
    main()

