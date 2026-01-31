# Stage1 详细流程说明

## 当前运行配置

```
进程: PID 3199063
配置参数:
  - sampler.anchor_sample_num=20    # 20个anchor（测试样本）
  - sampler.candidate_num=20         # 每个anchor有20个候选ICD
  - few_shot_num=4                   # 每个序列选择4个ICD
  - beam_size=5                      # Beam Search保留5个最优序列
  - metric=no_order                   # 不关注顺序（只在query两侧插入）
  - mc_num=20                        # 蒙特卡洛采样20次（实际可能用128，见下文）
  - batch_size=1                     # 批处理大小为1
```

---

## 整体流程概览

```
1. 数据准备
   ↓
2. 对每个anchor（测试样本）:
   ├─ 初始化: 序列 = [query]
   ├─ 迭代 few_shot_num 次（4次）:
   │   ├─ 对当前每个序列（beam_size个）:
   │   │   ├─ 过滤已使用的候选
   │   │   ├─ 构建所有(候选ICD, 插入位置)组合
   │   │   ├─ 批量计算InfoScore
   │   │   └─ 选择top beam_size个组合
   │   └─ 更新序列列表
   └─ 保存结果
```

---

## 详细步骤解析

### 步骤1: 数据准备

```python
# generate_data_main.py
train_ds = load_ds(cfg, split="train")  # 加载训练集
sampler_result = sampler(train_ds)      # 随机采样
  - anchor_set: 20个测试样本的索引
  - candidate_set_idx: {anchor_idx: [20个候选ICD索引]}
```

### 步骤2: 对每个anchor处理

#### 2.1 初始化序列

```python
# generate_data.py:121
test_data_id_list = [[test_data_id]]  # 初始序列只包含query
# 例如: [[5118]]  # 5118是query的idx
```

#### 2.2 Beam Search迭代（4轮）

**第1轮迭代（选择第1个ICD）:**

```python
当前序列: [[5118]]  # 只有query

1. 过滤候选:
   - 候选池: 20个候选ICD（排除query本身）
   - 插入位置: [0, 1]  # metric=no_order时，只有2个位置
     - position=0: 插入到query后面 → [query, ICD]
     - position=1: 插入到query前面 → [ICD, query]

2. 构建组合:
   - 20个候选 × 2个位置 = 40个组合
   - 例如: (ICD_6553, position=0), (ICD_6553, position=1), ...

3. 批量计算InfoScore（见下方详细说明）

4. 选择top 5个组合（beam_size=5）
   - 例如: 
     - [6553, 5118] score=25.1
     - [5118, 1282] score=24.8
     - [3320, 5118] score=24.5
     - [5118, 333] score=24.2
     - [1403, 5118] score=23.9

5. 更新序列列表:
   test_data_id_list = [
       [6553, 5118],
       [5118, 1282],
       [3320, 5118],
       [5118, 333],
       [1403, 5118]
   ]
```

**第2轮迭代（选择第2个ICD）:**

```python
当前序列: 5个序列（beam_size=5）

对每个序列:
1. 过滤候选:
   - 排除已使用的ICD和query
   - 例如序列[6553, 5118]:
     - 排除: 6553, 5118
     - 剩余候选: 18个

2. 插入位置:
   - 序列长度=2，有3个位置: [0, 1, 2]
     - position=0: 末尾 → [6553, 5118, new_ICD]
     - position=1: 中间 → [6553, new_ICD, 5118]
     - position=2: 开头 → [new_ICD, 6553, 5118]

3. 构建组合:
   - 18个候选 × 3个位置 = 54个组合（对每个序列）

4. 批量计算InfoScore

5. 全局选择top 5个组合（从所有序列的所有组合中选择）
   - 例如:
     - [6553, 1282, 5118] score=28.5
     - [2159, 3320, 5118] score=28.2
     - [4186, 6553, 5118] score=27.9
     - [6553, 5118, 333] score=27.6
     - [3320, 1403, 5118] score=27.3

6. 更新序列列表（5个新序列）
```

**第3、4轮迭代:** 类似，每次选择1个新ICD，最终得到5个包含4个ICD的序列。

---

## InfoScore计算详解（核心部分）

### InfoScore定义

```
InfoScore = log_likelihood(插入后) - log_likelihood(插入前)
```

### 计算流程

#### 1. 构建Prompt的三个部分

```python
# utils.py:185 build_prompt_parts

输入: 
  - left_icds: query左侧的ICD列表
  - query: query样本（包含question和answer）
  - right_icds: query右侧的ICD列表

输出:
  - left_text: "ICD1\n\nICD2\n\nquestion: ...\n<answer>\n"
  - answer_text: "真实的答案文本"
  - right_text: "\n</answer>\n\nICD3\n\nICD4"
```

**示例:**

```python
# 插入前（baseline）
left_icds = [ICD_6553]
query = {question: "Q1", answer: "A1"}
right_icds = []

left_text = "question: Q6553\n<answer>\nA6553\n</answer>\n\nquestion: Q1\n<answer>\n"
answer_text = "A1"
right_text = "\n</answer>"

# 插入后（插入ICD_1282到position=1）
left_icds = [ICD_6553, ICD_1282]  # 新增ICD_1282
query = {question: "Q1", answer: "A1"}
right_icds = []

left_text = "question: Q6553\n<answer>\nA6553\n</answer>\n\nquestion: Q1282\n<answer>\nA1282\n</answer>\n\nquestion: Q1\n<answer>\n"
answer_text = "A1"
right_text = "\n</answer>"
```

#### 2. Tokenize

```python
prompt_left = tokenizer(left_text)["input_ids"]    # shape: (L_left,)
answer = tokenizer(answer_text)["input_ids"]       # shape: (L_answer,)
prompt_right = tokenizer(right_text)["input_ids"]  # shape: (L_right,)
```

#### 3. 计算Log-Likelihood（蒙特卡洛方法）

```python
# utils.py:225
score = interface.compute_log_likelihood(
    prompt_left=prompt_left,
    answer=answer,
    prompt_right=prompt_right,
    mc_num=20,  # 蒙特卡洛采样次数
    batch_size=1,
    cfg_scale=0.0,
)
```

---

## 蒙特卡洛计算详解（核心算法）

### 函数调用链

```
compute_log_likelihood (llada_interface.py:245)
  ↓
get_log_likelihood (src/eval_likelihood.py:53)
  ↓
forward_process (src/eval_likelihood.py:7)  # 每次采样调用
get_logits (src/eval_likelihood.py:35)        # 每次采样调用
```

### 详细步骤

#### 步骤1: 构建完整序列

```python
# src/eval_likelihood.py:78
seq = torch.concatenate([prompt_left, answer, prompt_right])
# shape: (1, L_total)
# 例如: [token1, token2, ..., answer_token1, answer_token2, ..., tokenN]

# 标记answer位置
answer_start = len(prompt_left)
answer_end = answer_start + len(answer)
answer_index = [False, ..., True, True, ..., False]  # answer位置为True
```

#### 步骤2: 重复batch_size次

```python
seq = seq.repeat((batch_size, 1))  # shape: (batch_size, L_total)
# batch_size=1时，shape: (1, L_total)
```

#### 步骤3: 蒙特卡洛采样循环

```python
# mc_num=20, batch_size=1
# 循环次数: 20 // 1 = 20次

for _ in range(20):
    # 步骤3.1: forward_process - 添加噪声（部分mask）
    perturbed_seq, p_mask = forward_process(seq, answer_index, mask_id)
    
    # 步骤3.2: 前向传播
    logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
    
    # 步骤3.3: 计算loss
    loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
    loss = loss.sum() / batch_size
    
    loss_.append(loss.item())

# 步骤3.4: 平均
return - sum(loss_) / len(loss_)  # 返回log-likelihood
```

---

## forward_process详解（噪声添加过程）

### 函数签名

```python
def forward_process(batch, answer_index, mask_id):
    """
    Args:
        batch: shape (batch_size, seq_len) - 完整序列
        answer_index: shape (seq_len,) - answer位置的布尔索引
        mask_id: mask token的ID (126336)
    
    Returns:
        perturbed_seq: 添加噪声后的序列（部分answer被替换为mask_id）
        p_mask: 每个位置的mask概率
    """
```

### 详细步骤

#### 步骤1: 确定要mask的answer token数量

```python
target_len = answer_index.sum().item()  # answer的token数量
# 例如: answer有50个tokens

k = torch.randint(1, target_len + 1, ())  # 随机选择1到target_len之间的数
# 例如: k = 30，表示要mask 30个tokens
```

#### 步骤2: 为batch中每个样本生成不同的mask数量

```python
# batch_size=1时，x = [k]
# batch_size>1时，x会在[k, k+1, ..., k+batch_size-1]之间均匀分布
x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b)).long()
x = ((x - 1) % target_len) + 1  # 确保在[1, target_len]范围内
# 例如: x = [30]  # batch_size=1
```

#### 步骤3: 随机选择要mask的位置

```python
# 创建索引
indices = torch.arange(target_len, device=device).repeat(b, 1)
# shape: (batch_size, target_len)
# 例如: [[0, 1, 2, ..., 49]]  # batch_size=1

# 确定哪些位置要mask
is_mask = indices < x.unsqueeze(1)
# 例如: x=30, is_mask = [True, True, ..., True(30个), False, False, ...]

# 随机打乱mask位置
for i in range(b):
    is_mask[i] = is_mask[i][torch.randperm(target_len)]
# 例如: is_mask = [False, True, False, True, ..., True, False]
#       随机打乱，但仍然是30个True
```

#### 步骤4: 映射到完整序列

```python
# 创建完整序列的mask索引
is_mask_full = torch.zeros(b, l, dtype=torch.bool, device=device)
# shape: (batch_size, seq_len)

# 将answer位置的mask映射到完整序列
for i in range(b):
    is_mask_full[i, answer_index] = is_mask[i]

# 例如:
# answer_index = [False, ..., True(50个), False, ...]
# is_mask = [False, True, False, True, ...]  # 30个True
# is_mask_full = [False, ..., True, False, True, ..., False, ...]
#                (answer位置有30个True，其他位置都是False)
```

#### 步骤5: 替换为mask_id

```python
noisy_batch = torch.where(is_mask_full, mask_id, batch)
# 将is_mask_full为True的位置替换为mask_id，其他位置保持不变

# 例如:
# batch = [token1, token2, ..., answer_token1, answer_token2, ..., tokenN]
# noisy_batch = [token1, token2, ..., mask_id, answer_token2, mask_id, ..., tokenN]
#                (answer的30个随机位置被替换为mask_id)
```

#### 步骤6: 计算mask概率

```python
p_mask = torch.zeros(b, l, device=device)
p_mask[:, answer_index] = (x / target_len).unsqueeze(1).repeat(1, target_len)

# 例如: x=30, target_len=50
# p_mask[answer位置] = 30/50 = 0.6
# p_mask[非answer位置] = 0.0
```

### 返回结果

```python
return noisy_batch, p_mask
# noisy_batch: 部分answer被mask的序列
# p_mask: 每个位置的mask概率（用于重要性采样）
```

---

## get_logits详解（模型前向传播）

### 函数签名

```python
def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    """
    Args:
        model: LLaDA模型
        batch: shape (batch_size, seq_len) - 添加噪声后的序列
        prompt_index: shape (seq_len,) - prompt位置的布尔索引
        cfg_scale: CFG scale（当前为0.0，不使用CFG）
        mask_id: mask token的ID
    """
```

### 详细步骤

#### 步骤1: CFG处理（当前cfg_scale=0.0，跳过）

```python
if cfg_scale > 0.:
    # 创建unconditional batch（所有prompt位置替换为mask_id）
    un_batch = batch.clone()
    un_batch[prompt_index] = mask_id
    batch = torch.cat([batch, un_batch])  # 拼接conditional和unconditional
    # 当前cfg_scale=0.0，不执行
```

#### 步骤2: 模型前向传播

```python
logits = model(batch).logits
# shape: (batch_size, seq_len, vocab_size)
# 例如: (1, 500, 50000)  # 序列长度500，词汇表大小50000
```

#### 步骤3: CFG后处理（当前跳过）

```python
if cfg_scale > 0.:
    logits, un_logits = torch.chunk(logits, 2, dim=0)
    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    # 当前cfg_scale=0.0，不执行
```

### 返回结果

```python
return logits  # shape: (batch_size, seq_len, vocab_size)
```

---

## Loss计算详解

### 计算步骤

```python
# src/eval_likelihood.py:95
loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
```

#### 步骤1: 找到mask位置

```python
mask_index = perturbed_seq == mask_id
# shape: (batch_size, seq_len)
# 例如: [[False, ..., True, False, True, ..., False]]
#       只有被mask的位置为True
```

#### 步骤2: 提取logits和targets

```python
# logits: shape (batch_size, seq_len, vocab_size)
# seq: shape (batch_size, seq_len) - 原始序列（未mask）

logits_at_mask = logits[mask_index]  # shape: (num_masked_tokens, vocab_size)
targets_at_mask = seq[mask_index]    # shape: (num_masked_tokens,)
```

#### 步骤3: 计算交叉熵

```python
loss_per_token = F.cross_entropy(
    logits_at_mask, 
    targets_at_mask, 
    reduction='none'
)
# shape: (num_masked_tokens,)
# 每个被mask的token的loss
```

#### 步骤4: 重要性采样（除以mask概率）

```python
p_mask_at_mask = p_mask[mask_index]  # shape: (num_masked_tokens,)
loss_per_token = loss_per_token / p_mask_at_mask
# 重要性采样：除以mask概率，得到无偏估计
```

#### 步骤5: 求和并平均

```python
loss = loss_per_token.sum() / batch_size
# 对所有masked tokens的loss求和，然后除以batch_size
```

### 为什么除以p_mask？

这是**重要性采样（Importance Sampling）**，用于无偏估计：

```
E[loss] = E[loss_per_token / p_mask]
```

当某些token被mask的概率较低时，它们的loss会被放大，从而得到无偏的期望值。

---

## 蒙特卡洛估计

### 最终结果

```python
# 20次采样的loss列表
loss_ = [loss1, loss2, ..., loss20]

# 平均得到log-likelihood
log_likelihood = - sum(loss_) / len(loss_)
# 注意：取负号，因为loss是负对数似然
```

### 数学原理

蒙特卡洛方法估计log-likelihood：

```
log P(answer | prompt) ≈ -1/N * Σ loss_i
```

其中：
- N = mc_num = 20（采样次数）
- loss_i = 第i次采样的loss
- 通过多次采样，得到log-likelihood的无偏估计

---

## InfoScore计算

### 完整流程

```python
# 1. 计算插入前的baseline
score_before = compute_log_likelihood(
    prompt_left=left_before,
    answer=answer,
    prompt_right=right_before,
    mc_num=20,
)

# 2. 对每个(候选ICD, 位置)组合:
for candidate, position in zip(candidate_data_list, position_list):
    # 2.1 构建插入后的prompt
    left_after, _, right_after = build_prompt_parts(
        left_icds + [candidate] if position合适,
        query,
        right_icds + [candidate] if position合适,
    )
    
    # 2.2 计算插入后的分数
    score_after = compute_log_likelihood(
        prompt_left=left_after,
        answer=answer,
        prompt_right=right_after,
        mc_num=20,
    )
    
    # 2.3 计算InfoScore
    infoscore = score_after - score_before
```

### InfoScore含义

```
InfoScore = log P(answer | prompt_with_ICD) - log P(answer | prompt_without_ICD)
          = log [P(answer | prompt_with_ICD) / P(answer | prompt_without_ICD)]
```

- **InfoScore > 0**: 插入ICD后，模型对答案的置信度提高
- **InfoScore < 0**: 插入ICD后，模型对答案的置信度降低
- **InfoScore越大**: ICD对query的帮助越大

---

## 当前运行状态

### 配置参数

- **anchor_sample_num=20**: 处理20个测试样本
- **candidate_num=20**: 每个测试样本有20个候选ICD
- **few_shot_num=4**: 每个序列选择4个ICD
- **beam_size=5**: 保留5个最优序列
- **mc_num=20**: 每次计算log-likelihood时采样20次

### 计算量估算

**每个anchor的计算量:**

- 第1轮: 20候选 × 2位置 × 2次计算（before+after） = 80次log-likelihood计算
- 第2轮: 5序列 × 18候选 × 3位置 × 2次计算 = 540次
- 第3轮: 5序列 × 17候选 × 4位置 × 2次计算 = 680次
- 第4轮: 5序列 × 16候选 × 5位置 × 2次计算 = 800次

**总计每个anchor:** 约2100次log-likelihood计算

**每次log-likelihood计算:**
- 20次蒙特卡洛采样
- 每次采样: 1次模型前向传播

**总计每个anchor:** 约42,000次模型前向传播

**20个anchor总计:** 约840,000次模型前向传播

### 注意事项

⚠️ **mc_num参数问题:**
- 命令行参数: `mc_num=20`
- 但进程在修复前启动，可能实际使用的是默认值128
- 如果使用128，计算量会增加到约5,376,000次前向传播

---

## 总结

Stage1的核心是：
1. **Beam Search**: 逐步选择最优的ICD序列
2. **InfoScore**: 评估每个ICD对query的帮助程度
3. **蒙特卡洛方法**: 通过多次采样估计log-likelihood
4. **重要性采样**: 通过除以mask概率得到无偏估计

整个过程通过大量计算找到对每个query最有帮助的ICD组合。
