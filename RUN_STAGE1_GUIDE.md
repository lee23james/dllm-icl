# Stage 1 运行指南

## 快速开始

### 1. 小批量测试（推荐先运行）

```bash
# 最简单的运行方式
./run_stage1.sh -s

# 或者指定更多参数
./run_stage1.sh -s sampler.candidate_num=20 few_shot_num=3
```

**配置：**
- test_sample数量：5
- 候选池大小：10
- ICD数量：2
- beam_size：1

**预计时间：** 几分钟（取决于模型大小和GPU）

---

### 2. 中等规模测试

```bash
./run_stage1.sh -m

# 调整候选池大小
./run_stage1.sh -m sampler.candidate_num=50
```

**配置：**
- test_sample数量：50
- 候选池大小：32
- ICD数量：3
- beam_size：2

**预计时间：** 10-30分钟

---

### 3. 大规模运行（完整数据生成）

```bash
./run_stage1.sh -l

# 使用多GPU加速
./run_stage1.sh -l use_multi_gpu=true gpu_ids=[0,1,2,3]
```

**配置：**
- test_sample数量：5000
- 候选池大小：64
- ICD数量：4
- beam_size：2

**预计时间：** 数小时（取决于GPU数量和速度）

---

### 4. 关注顺序的完整配置

```bash
./run_stage1.sh -o
```

**配置：**
- test_sample数量：1000
- 候选池大小：64
- ICD数量：4
- beam_size：2
- metric：order（所有位置插入，更慢但更精确）

---

### 5. 自定义配置

```bash
# 完全自定义
./run_stage1.sh -c \
    sampler.anchor_sample_num=100 \
    sampler.candidate_num=32 \
    few_shot_num=3 \
    beam_size=2 \
    metric=no_order

# 或者直接提供参数（不使用预设）
./run_stage1.sh \
    sampler.anchor_sample_num=10 \
    sampler.candidate_num=20 \
    few_shot_num=2 \
    beam_size=1
```

---

## 常用参数说明

### 核心参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `sampler.anchor_sample_num` | test_sample数量 | 5000 | `sampler.anchor_sample_num=100` |
| `sampler.candidate_num` | 候选池大小 | 64 | `sampler.candidate_num=32` |
| `few_shot_num` | 要选择的ICD数量 | 4 | `few_shot_num=3` |
| `beam_size` | Beam Search大小 | 2 | `beam_size=1` |
| `metric` | 插入策略 | "no_order" | `metric=order` |

### 性能参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `mc_num` | Monte Carlo采样次数 | 128 | `mc_num=32`（更快但精度降低） |
| `batch_size` | 批处理大小 | 1 | `batch_size=1` |
| `use_multi_gpu` | 是否使用多GPU | false | `use_multi_gpu=true` |
| `gpu_ids` | GPU ID列表 | [0] | `gpu_ids=[0,1,2,3]` |

### 其他参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `sampler.overwrite` | 是否覆盖缓存 | false | `sampler.overwrite=true` |
| `output_dir` | 输出目录 | "./generated_icd_data" | `output_dir="./my_results"` |

---

## 使用示例

### 示例1：快速验证逻辑

```bash
# 最小配置，快速验证
./run_stage1.sh -s \
    sampler.anchor_sample_num=2 \
    sampler.candidate_num=5 \
    few_shot_num=1 \
    beam_size=1 \
    mc_num=32
```

### 示例2：调整候选池大小

```bash
# 使用更大的候选池
./run_stage1.sh -m sampler.candidate_num=100
```

### 示例3：加快速度（降低精度）

```bash
# 减少Monte Carlo采样次数
./run_stage1.sh -m mc_num=32

# 或者使用更小的beam_size
./run_stage1.sh -m beam_size=1
```

### 示例4：提高精度（降低速度）

```bash
# 增加Monte Carlo采样次数
./run_stage1.sh -m mc_num=256

# 或者使用order策略（所有位置插入）
./run_stage1.sh -m metric=order
```

### 示例5：多GPU运行

```bash
# 使用4个GPU
./run_stage1.sh -l \
    use_multi_gpu=true \
    gpu_ids=[0,1,2,3] \
    sleep_time=10
```

---

## 输出文件

### 结果文件

**位置：** `./generated_icd_data/gsm8k_icd_results.json`

**格式：**
```json
{
  "anchor_idx_1": {
    "id_list": [
      [id1, id2, query_id, id3],
      [id4, query_id, id5],
      ...
    ],
    "score_list": [score1, score2, ...]
  },
  "anchor_idx_2": {
    ...
  }
}
```

### 日志文件

**位置：** `./outputs/YYYY-MM-DD_HH-MM-SS/generate_data.log`

**内容：**
- 数据集加载信息
- Sampler生成信息
- 每个anchor的处理进度
- 错误信息（如果有）

### 缓存文件

**位置：** `./cache/gsm8k/`

**文件：**
- `gsm8k-anchor_sample_num:XXX.json` - anchor集合缓存
- `gsm8k-random_sampler-anchor_sample_num:XXX:YYY.json` - 候选集缓存

---

## 常见问题

### Q1: 如何查看运行进度？

```bash
# 实时查看日志
tail -f ./outputs/*/generate_data.log

# 或者查看最新的日志文件
ls -t ./outputs/*/generate_data.log | head -1 | xargs tail -f
```

### Q2: 如何中断运行？

按 `Ctrl+C` 中断，已处理的结果会保存在输出文件中。

### Q3: 如何从断点继续？

多GPU模式下，每个rank会保存自己的进度，重新运行会自动从断点继续。

单GPU模式下，需要手动处理（目前不支持断点续传）。

### Q4: 如何清理缓存？

```bash
# 删除所有缓存
rm -rf ./cache/gsm8k/

# 或者只删除候选集缓存
rm ./cache/gsm8k/gsm8k-random_sampler-*.json
```

### Q5: 内存不足怎么办？

```bash
# 减小候选池大小
./run_stage1.sh -s sampler.candidate_num=10

# 减小beam_size
./run_stage1.sh -s beam_size=1

# 减少Monte Carlo采样次数
./run_stage1.sh -s mc_num=32
```

---

## 性能优化建议

### 1. 加快速度

- ✅ 减小 `mc_num`（如32或64）
- ✅ 减小 `beam_size`（如1）
- ✅ 使用 `metric=no_order`（只在两侧插入）
- ✅ 使用多GPU（`use_multi_gpu=true`）

### 2. 提高精度

- ✅ 增加 `mc_num`（如256或512）
- ✅ 增加 `beam_size`（如4或8）
- ✅ 使用 `metric=order`（所有位置插入）
- ✅ 增加 `sampler.candidate_num`（更大的候选池）

### 3. 平衡速度和精度

- ✅ `mc_num=128`（默认值）
- ✅ `beam_size=2`（默认值）
- ✅ `metric=no_order`（默认值）
- ✅ `sampler.candidate_num=64`（默认值）

---

## 下一步

Stage 1完成后，可以使用生成的结果进行Stage 2（ICL推理）：

```bash
python icl_inference.py
```

或者查看结果文件：

```python
import json
with open("./generated_icd_data/gsm8k_icd_results.json", "r") as f:
    results = json.load(f)
    print(f"处理了 {len(results)} 个anchor")
    print(f"第一个anchor的序列数: {len(results[list(results.keys())[0]]['id_list'])}")
```
