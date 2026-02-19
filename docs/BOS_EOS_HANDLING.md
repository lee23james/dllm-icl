# BOS 与 EOS 处理说明

本文档详细说明本代码库中 BOS（序列开始）和 EOS（序列结束）的分配、使用与计算方式。

---

## 1. Token ID 分配

词表结构（`vocab_size = index_ds_size + 3`）：

| Token ID | 含义 | 用途 |
|----------|------|------|
| 0 .. index_ds_size-1 | content id | 训练集中样本的索引 |
| index_ds_size | BOS | 序列开始 |
| index_ds_size + 1 | EOS | 序列结束 |
| index_ds_size + 2 | QUERY placeholder | query 占位符 |

**注意**：`train.py` 的 collate 与 `gpt2_lever_lm.py` 的 GPT2Config 对 BOS/EOS 的语义约定一致，均为：
- `bos_id = index_ds_size`
- `eos_id = index_ds_size + 1`

---

## 2. 序列构造（Collate）

在 `train.py` 的 `make_collate_fn` 中：

```python
bos_id = index_ds_size
eos_id = index_ds_size + 1
query_placeholder_id = index_ds_size + 2

# 每个样本的序列
full = [bos_id, query_placeholder_id] + icd_ids + [eos_id]
# 即：[BOS, QUERY, icd_1, icd_2, ..., icd_k, EOS]
```

- **position 0**：BOS，表示序列开始
- **position 1**：QUERY placeholder，用于注入 query embedding
- **position 2..K+1**：ICD id
- **position K+2**：EOS，表示序列结束

### 2.1 Padding

```python
input_ids = torch.full((len(seqs), max_len), eos_id, dtype=torch.long)
```

不足 `max_len` 的位置用 **EOS** 填充，`attention_mask` 中对应为 0。

### 2.2 Labels

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100
```

padding 位置的 label 设为 -100，不参与 loss 计算。

---

## 3. 模型中的 Embedding 处理

### 3.1 基础 Embedding（所有位置）

```python
inputs_embeds = self.lm_model.transformer.wte(icd_seq_idx)
```

每个位置（含 BOS、EOS）都通过 `wte` 得到可学习 embedding。

### 3.2 Query 特征（仅 position 1）

```python
inputs_embeds[:, 1] += query_feat
```

只对 position 1（QUERY）叠加 query embedding，BOS 和 EOS 不参与。

### 3.3 ICD 特征（position 2..L-1）

```python
icd_embeds = self.embedding_lookup[icd_seq_idx]
icd_feats = self.input_adapter(icd_embeds)
inputs_embeds[:, 2:] += icd_feats[:, 2:]
```

对 position 2 及之后的位置叠加 ICD text embedding。

**BOS（position 0）**：
- 不在 `inputs_embeds[:, 2:]` 范围内
- 只使用 `wte(BOS_id)`，不叠加 query 或 ICD 特征

**EOS（position L-1）**：
- 在 `inputs_embeds[:, 2:]` 范围内
- `embedding_lookup[EOS_id]` 为 0（特殊 token 对应行未填充）
- `icd_feats[:, L-1] = adapter(0)`，可能非零（adapter 有 bias）
- 因此 EOS 的表示为：`wte(EOS_id) + adapter(0)`

### 3.4 汇总

| 位置 | Token | 可学习 wte | Query 特征 | ICD 特征 |
|------|-------|------------|------------|----------|
| 0 | BOS | ✓ | ✗ | ✗ |
| 1 | QUERY | ✓ | ✓ | ✗ |
| 2..K+1 | ICD | ✓ | ✗ | ✓ |
| K+2 | EOS | ✓ | ✗ | adapter(0) |

---

## 4. embedding_lookup 中的 BOS/EOS

```python
emb_lookup = torch.zeros(index_ds_size + 3, 2560)
for i in range(index_ds_size):
    if i in embedding_dict:
        emb_lookup[i] = embedding_dict[i]
# 行 index_ds_size, index_ds_size+1, index_ds_size+2 保持为 0
```

- 行 `index_ds_size`（BOS）：0
- 行 `index_ds_size+1`（EOS）：0
- 行 `index_ds_size+2`（QUERY）：0（该位置用 query_embedding，不查表）

BOS 和 EOS 在 embedding_lookup 中均为 0，因此不会从该表引入额外语义。

---

## 5. GPT2Config 中的 BOS/EOS

```python
config = GPT2Config(
    ...
    eos_token_id=index_ds_size,
    bos_token_id=index_ds_size + 1,
)
```

这里与 collate 的约定相反：GPT2Config 中 `eos_token_id` 对应 `index_ds_size`，`bos_token_id` 对应 `index_ds_size+1`。  
实际序列构造仍按 collate 的约定：position 0 用 `index_ds_size`（BOS），末尾用 `index_ds_size+1`（EOS）。GPT2Config 主要用于 GPT2 内部（如生成时的停止条件），对当前 forward 的 inputs_embeds 无影响。

---

## 6. 推理（generation）

```python
icd_seq_idx = [[bos_token_id, query_placeholder_id]] * batch_size
for _ in range(shot_num):
    next_token = logits.argmax(dim=-1)
    icd_seq_idx = torch.cat([icd_seq_idx, next_token], dim=1)
# 返回 [BOS, QUERY, id_1, ..., id_shot_num]
```

- 以 `[BOS, QUERY]` 为起始序列
- 自回归生成 `shot_num` 个 ICD id
- 生成时通过 `logits[:, index_ds_size:] = -inf` 禁止生成 BOS/EOS/QUERY
- 不显式生成 EOS，序列长度由 `shot_num` 固定

---

## 7. Loss 中的 BOS/EOS

因果 LM 的 next-token 预测：

- position 0（BOS）→ 预测 position 1（QUERY）
- position 1（QUERY）→ 预测 position 2（icd_1）
- ...
- position K+1（icd_k）→ 预测 position K+2（EOS）

因此：
- BOS 作为输入参与预测 QUERY
- EOS 作为预测目标，由最后一个 ICD 位置预测

padding 位置（label=-100）不参与 loss。

---

## 8. 小结

| 维度 | BOS | EOS |
|------|-----|-----|
| Token ID | index_ds_size | index_ds_size + 1 |
| 序列位置 | 0 | 末尾 |
| 可学习 wte | ✓ | ✓ |
| Query 特征 | ✗ | ✗ |
| ICD 特征 | ✗ | adapter(0) |
| embedding_lookup | 0 | 0 |
| Padding 填充 | - | 使用 EOS |
| 预测角色 | 作为输入 | 作为预测目标 |
