# Lever-LM 训练逻辑详解

本文档详细说明 Lever-LM（ICD 序列生成模型）的训练流程、数据处理与计算逻辑。

---

## 1. 训练目标

**任务**：给定一个 query（待回答的测试样本），预测与之最匹配的 ICD（In-Context Demonstration）序列。

**输入**：query 的 text embedding（预计算的 Qwen-Embedding 向量）  
**输出**：ICD 的 token id 序列，即 `[id_1, id_2, ..., id_k]`，每个 id 对应训练集中一个样本的索引。

**模型**：GPT2 风格的因果语言模型，将「给定 query 预测 ICD 序列」建模为自回归 next-token 预测。

---

## 2. 数据流概览

```
Stage1 JSON (generated_data)     Qwen Embedding Cache (.pt)
         │                                │
         ▼                                ▼
  fix_icd_json_query_last          _load_embedding_dict
  (query 置尾)                     (N×D tensor → {id: vec})
         │                                │
         ▼                                │
  data_split (train/val 9:1)              │
         │                                │
         ▼                                ▼
  BaseLeverLMDataset ◄────────── embedding_dict
  (icd_seq, query_embedding)
         │
         ▼
  make_collate_fn
  (构造 [BOS, QUERY, icd_1..k, EOS])
         │
         ▼
  GPT2LeverLM.forward
  (wte + query_emb + icd_emb → LM)
         │
         ▼
  Cross-Entropy Loss (next-token prediction)
```

---

## 3. 输入数据格式

### 3.1 Stage1 生成的 JSON

来自 `generate_data_main.py` 或类似流程，格式为：

```json
{
  "63564": {
    "id_list": [
      [63441, 59232, 63564, 36949, 32884],
      [20010, 59232, 63564, 36949, 32884],
      ...
    ],
    "score_list": [0.125, 0.0625, 0.0, ...]
  },
  "74886": { ... }
}
```

- **key**：anchor（query）的 id，即训练集中某样本的索引
- **id_list**：多条 ICD 序列，每条为 `[icd_1, icd_2, ..., query_id]`，query 可能在中间或末尾
- **score_list**：每条序列的分数（如 InfoScore、MMR 等）

### 3.2 fix_icd_json_query_last

若 JSON 中 query 在序列中间（如 `[icd1, icd2, query, icd3, icd4]`），需先统一为 query 在末尾：

- 输入：`xxx.json`
- 输出：`xxx-fix.json`
- 规则：对每条 `id_list[i]`，将等于 anchor_id 的项移到末尾，其余保持相对顺序

例如：`[icd1, icd2, query, icd3, icd4]` → `[icd1, icd2, icd3, icd4, query]`

### 3.3 data_split

将 fix 后的 JSON 按 **query_id** 划分为 train/val：

1. 收集所有 query_id：`test_dataset_id_set = {v[-1] for ...}`
2. 按 `train_ratio`（默认 0.9）划分：前 90% 的 query 用于 train，后 10% 用于 val
3. 输出：
   - `train_data = {"icd_seq": [...], "icd_score": [...]}`
   - `val_data = {"icd_seq": [...], "icd_score": [...]}`

每条 `icd_seq[i]` 形如 `[icd_1, icd_2, ..., query_id]`，`icd_score[i]` 为对应分数。

### 3.4 embedding_dict

从 `embedding_path` 加载预计算的 Qwen embedding：

- **格式 1**：`(N, D)` tensor，按行转为 `{i: tensor[i]}`
- **格式 2**：`{id: vector}` 的 .pt/.pkl

其中 `N` = 训练集样本数，`D` = 2560（Qwen3-Embedding-4B 输出维度）。

---

## 4. Dataset：BaseLeverLMDataset

### 4.1 输入

- `data`：`{"icd_seq": [...], "icd_score": [...]}`（来自 data_split）
- `embedding_dict`：`{query_id: embedding}`

### 4.2 处理逻辑

1. 遍历 `icd_seq` 与 `icd_score`
2. 过滤：`icd_score >= threshold`（默认 0）
3. 约定：`icd_seq` 最后一个 id 为 `query_id`，前面为 ICD id 列表
4. 若 `query_id not in embedding_dict`，跳过
5. 保存：`icd_idx_seq_list`（ICD 序列）、`query_id_list`（query id）

### 4.3 __getitem__ 输出

```python
{
    "query_embedding": emb,      # [D] 来自 embedding_dict[query_id]
    "icd_seq_idx": tensor([icd_1, icd_2, ..., icd_k])  # [K]，不含 query
}
```

---

## 5. Collate：make_collate_fn

将 batch 内样本拼成完整 token 序列。

### 5.1 词表与特殊 token

| ID 范围 | 含义 |
|---------|------|
| 0 .. index_ds_size-1 | content id（训练集样本） |
| index_ds_size | BOS |
| index_ds_size + 1 | EOS |
| index_ds_size + 2 | QUERY placeholder |

### 5.2 序列构造

对每个样本：

```
full = [BOS, QUERY_PLACEHOLDER] + icd_ids + [EOS]
```

即：`[bos_id, query_placeholder_id, icd_1, icd_2, ..., icd_k, eos_id]`

### 5.3 输出

```python
{
    "query_embedding": [B, D],      # batch 内每个 query 的 embedding
    "icd_seq_idx": [B, L],         # 完整序列（含 padding）
    "attention_mask": [B, L],      # 1=有效，0=padding
    "labels": [B, L]               # 同 icd_seq_idx，padding 位置为 -100
}
```

---

## 6. 模型：GPT2LeverLM

### 6.1 结构

- **backbone**：GPT2LMHeadModel（n_embd=512, n_layer=2, n_head=8）
- **input_adapter**：`Linear(2560, 2048) → ReLU → Linear(2048, 512)`，将 Qwen 2560 维映射到 512 维
- **embedding_lookup**：`[vocab_size, 2560]`，每行对应一个 content id 的预计算 embedding

### 6.2 forward 计算流程

设 `icd_seq_idx` 形状为 `[B, L]`，`query_embedding` 为 `[B, 2560]`。

#### Step 1：基础 embedding

```python
inputs_embeds = wte(icd_seq_idx)  # [B, L, 512]
```

每个位置先由 token id 通过 `wte` 得到可学习向量。

#### Step 2：叠加 query 特征（position 1）

```python
query_feat = input_adapter(query_embedding)   # [B, 512]
inputs_embeds[:, 1] += query_feat
```

position 1 为 QUERY 占位符，叠加 query 的 text embedding（经 adapter 投影）。

#### Step 3：叠加 ICD 特征（position 2..L-1）

```python
icd_embeds = embedding_lookup[icd_seq_idx]    # [B, L, 2560]
icd_feats = input_adapter(icd_embeds)        # [B, L, 512]
inputs_embeds[:, 2:] += icd_feats[:, 2:]
```

对 content id 位置，叠加对应样本的 text embedding。BOS、EOS 等特殊 token 的 `embedding_lookup` 行为为 0，不改变其表示。

#### Step 4：LM 前向与 loss

```python
output = lm_model(inputs_embeds=inputs_embeds, attention_mask=..., labels=labels)
```

标准因果 LM：对每个位置预测下一个 token，`labels` 中 padding 为 -100，不参与 loss。

### 6.3 各位置含义

| 位置 | Token | 可学习 wte | Query 特征 | ICD 特征 | 预测目标 |
|------|-------|------------|------------|----------|----------|
| 0 | BOS | ✓ | ✗ | ✗ | 预测 QUERY |
| 1 | QUERY | ✓ | ✓ | ✗ | 预测 icd_1 |
| 2..K+1 | icd_1..icd_K | ✓ | ✗ | ✓ | 预测下一个 ICD 或 EOS |
| K+2 | EOS | ✓ | ✗ | ✗ | - |

---

## 7. embedding_lookup 的构建

在 `main()` 中，根据 `embedding_dict` 构建 `embedding_lookup`：

```python
emb_lookup = torch.zeros(index_ds_size + 3, 2560)
for i in range(index_ds_size):
    if i in embedding_dict:
        emb_lookup[i] = embedding_dict[i].squeeze().float()
# 行 index_ds_size, index_ds_size+1, index_ds_size+2 保持为 0
```

- 行 0..index_ds_size-1：对应训练集样本的 Qwen embedding
- 行 index_ds_size..index_ds_size+2：BOS/EOS/QUERY，保持 0（这些位置主要靠 wte 和 query 特征）

---

## 8. 训练配置

- **优化器**：AdamW，lr=1e-4，weight_decay=1e-3
- **学习率**：cosine schedule + warmup（5% 步数）
- **Checkpoint**：
  - `min_tl-*`：训练 loss 最低
  - `min_vl-*`：验证 loss 最低
  - `last.ckpt`：最后一个 epoch

---

## 9. 输出文件

| 路径 | 内容 |
|------|------|
| `generated_icd_data/model_cpk/mmlu/debug/` | 模型 checkpoint |
| `generated_icd_data/wandb_logs/` | wandb 日志 |
| `generated_icd_data/hydra_output/` | Hydra 配置与日志 |
| `generated_icd_data/generated_data/*-fix.json` | query 置尾后的 JSON |

---

## 10. 推理（generation）

给定 `query_embedding` 和 `shot_num`：

1. 初始化：`icd_seq_idx = [BOS, QUERY_PLACEHOLDER]`
2. 自回归生成 `shot_num` 个 token：
   - 调用 `forward` 得到 logits
   - 取最后一个位置的 logits，mask 掉特殊 token 和已用 id
   - argmax 得到下一个 token，拼接到序列
3. 返回 `[BOS, QUERY, id_1, ..., id_shot_num]`
