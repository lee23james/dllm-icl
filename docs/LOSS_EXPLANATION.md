# LeverLM 训练 Loss 详解

## 1. Loss 类型与公式

### 1.1 任务定义

LeverLM 做的是 **Causal Language Modeling（因果语言建模）**：给定前文 token，预测下一个 token。

序列格式：`[BOS, QUERY_placeholder, icd_1, icd_2, ..., icd_k, EOS]`

- **输入**：位置 0 到 L-1 的 token
- **预测目标**：位置 1 到 L 的 token（即“下一个 token”）
- **词表大小**：`V = index_ds_size + 3`（例如 99842 + 3 = 99845）

### 1.2 数学公式

使用 **Cross-Entropy Loss**（交叉熵损失）：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i \in \text{valid}} \log p(y_i \mid y_{<i})
$$

其中：
- \( N \)：参与 loss 计算的 token 数量（不含 `labels=-100` 的位置）
- \( y_i \)：第 \( i \) 个位置的真实 token id
- \( p(y_i \mid y_{<i}) \)：模型在给定前文 \( y_{<i} \) 时，预测 \( y_i \) 的 softmax 概率

等价形式：

$$
\mathcal{L} = \frac{1}{N} \sum_{i} \left[ \log \sum_{v=1}^{V} e^{z_{i,v}} - z_{i,y_i} \right]
$$

其中 \( z_{i,v} \) 是位置 \( i \) 对词表第 \( v \) 个 token 的 logit。

### 1.3 代码实现

- **train.py**：`output = self.lever_lm(**batch)`，取 `output["loss"]`
- **gpt2_lever_lm.py**：调用 `self.lm_model(inputs_embeds=..., labels=labels)`，内部是 HuggingFace 的 `GPT2LMHeadModel`
- **labels**：与 `input_ids` 相同，padding 位置为 `-100`（不参与 loss）
- **内部实现**：对 labels 做 shift，用 `CrossEntropyLoss(ignore_index=-100)` 计算 loss

---

## 2. 为什么 Loss 数值偏大？

### 2.1 词表很大

词表大小 \( V \approx 99845 \)。若模型接近均匀预测，每个 token 的 loss 约为：

$$
-\log \frac{1}{V} = \log V \approx \log 99845 \approx \mathbf{11.51}
$$

因此：
- **初始 loss ~11**：接近随机猜测，符合预期
- **val_loss ~7–8**：比随机好，但仍有较大不确定性
- **train_loss ~1.4**：训练集上预测较准

### 2.2 数值范围参考

| Loss 值 | 含义（近似） |
|---------|--------------|
| ~11.5   | 接近均匀分布，\( p \approx 1/V \) |
| ~7–8    | 平均 \( p \approx e^{-7} \sim e^{-8} \approx 0.0009 \sim 0.0003 \) |
| ~6.9    | \( p \approx 0.001 \) |
| ~4.6    | \( p \approx 0.01 \) |
| ~2.3    | \( p \approx 0.1 \) |
| ~1.4    | \( p \approx 0.25 \) |
| ~0.7    | \( p \approx 0.5 \) |
| 0       | 完全正确，\( p = 1 \) |

### 2.3 与分类任务的对比

- 二分类：随机 loss ≈ 0.69
- 10 类：随机 loss ≈ 2.3
- 100 类：随机 loss ≈ 4.6
- **99845 类**：随机 loss ≈ **11.5**

词表越大，随机 baseline 的 loss 越高，这是正常现象。

---

## 3. 序列与 Labels 构造

### 3.1 序列结构

```
input_ids:  [BOS, QUERY_ph, icd_1, icd_2, ..., icd_k, EOS]
labels:     [BOS, QUERY_ph, icd_1, icd_2, ..., icd_k, EOS]  (padding 处为 -100)
```

模型在内部做 shift，实际预测关系为：

- 位置 0 的 logits → 预测位置 1（QUERY_placeholder）
- 位置 1 的 logits → 预测位置 2（icd_1）
- …
- 位置 L-1 的 logits → 预测位置 L（EOS）

### 3.2 哪些位置参与 Loss

- `labels != -100` 的位置参与 loss
- padding 位置 `labels = -100` 被忽略
- 若序列无 padding，则除最后一个预测位置外，其余位置都参与（具体取决于 HuggingFace 的 shift 实现）

---

## 4. 总结

| 问题 | 回答 |
|------|------|
| Loss 公式 | Cross-Entropy：\( \mathcal{L} = -\frac{1}{N}\sum \log p(y_i \mid y_{<i}) \) |
| 为什么偏大 | 词表 V≈99845，随机 baseline ≈ 11.5；7–8 表示比随机好很多 |
| 初始 ~11 | 接近 \( \log V \)，符合随机猜测 |
| train_loss 下降 | 模型在训练集上学会预测 ICD 序列 |
| val_loss 上升 | 过拟合，泛化变差 |

**结论**：在当前词表规模下，loss 在 7–11 之间是合理的；更应关注 train_loss 与 val_loss 的差距，而不是绝对数值。
