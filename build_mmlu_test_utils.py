"""
从 MMLU 训练集中随机选出一批 **完全未参与 Stage1 JSON 的样本**，
作为单独的测试子集。

排除逻辑（更严格）：
- 不仅排除 JSON 的每个字段名（anchor/query 本身的 idx），
- 还排除所有出现在任意 id_list 里的编号（作为 ICD candidate 出现过的 idx）。

结果保存为一个简单的 JSON，里面只存这些 query 的 idx，
后续可以在 icl_inference.py 里按 idx 过滤出对应样本进行测试。
"""

import os
import json
import random
from datasets import load_dataset

# 1) 读 Stage1 生成的 JSON（LeverLM 训练用，大规模 random_sampler 结果）
STAGE1_JSON = (
    "generated_icd_data/generated_data/"
    "mmlu-mmlu-llada-random_sampler-scorer_infoscore-construct_order_no_order-"
    "beam_size_5-few_shot_3-candidate_num_64-sample_num_5000-mc_num_1-fix.json"
)

with open(STAGE1_JSON, "r", encoding="utf-8") as f:
    generated_data = json.load(f)

# JSON 里所有已经参与过 Stage1 的样本编号：
# 1) 字段名本身（anchor/query 的 idx）
used_ids_from_keys = {int(k) for k in generated_data.keys()}
# 2) 所有 id_list 里的每一个编号（无论是 query 还是 ICD，都视为“已用过”）
used_ids_from_lists = {
    int(x)
    for anchor in generated_data.values()
    for seq in anchor["id_list"]
    for x in seq
}
used_ids = used_ids_from_keys | used_ids_from_lists

print(
    f"[build_mmlu_test_utils] used_ids_from_keys={len(used_ids_from_keys)}, "
    f"used_ids_from_lists={len(used_ids_from_lists)}, merged_used_ids={len(used_ids)}"
)

# 2) 读 MMLU 训练集（auxiliary_train parquet）
MMLU_TRAIN_PATH = "/home/lzh/llada-icl/data/MMLU/auxiliary_train/train-00000-of-00001.parquet"
ds = load_dataset("parquet", data_files={"train": MMLU_TRAIN_PATH})["train"]

# 如果 parquet 里没有 idx，就自己加一列（与 load_mmlu_ds 中的约定保持一致）
if "idx" not in ds.column_names:
    ds = ds.add_column("idx", list(range(len(ds))))

all_train_ids = set(ds["idx"])
print(f"[build_mmlu_test_utils] all_train_ids in MMLU train: {len(all_train_ids)}")

# 3) 从“完全没出现在 Stage1 JSON（键或任意 id_list 元素）里的训练样本”中随机挑 N 个
N_TEST = 500
candidate_ids = sorted(all_train_ids - used_ids)
sample_size = min(N_TEST, len(candidate_ids))

if sample_size == 0:
    raise RuntimeError("No available candidates: all train ids are already used in Stage1 JSON.")

random.seed(42)
picked_ids = random.sample(candidate_ids, sample_size)

print(
    f"[build_mmlu_test_utils] candidates={len(candidate_ids)}, "
    f"picked={len(picked_ids)} (N_TEST={N_TEST})"
)

# 4) 存成一个简单的 JSON，里面放：
#    - query_ids: 作为 ICL 测试 query 的 idx 列表
#    - forbidden_ids: 所有在 Stage1 JSON 中出现过的 idx（用于 RandRetriever 避免选为 ICD）
OUT_PATH = "generated_icd_data/cache/test_500.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query_ids": picked_ids,
            "forbidden_ids": sorted(used_ids),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"saved subset ids (len={len(picked_ids)}) to {OUT_PATH}")