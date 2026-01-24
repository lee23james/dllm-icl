# 代码流程分析：为每个test选择候选ICD并生成最优序列

## 整体架构流程

```
配置文件 → 数据集加载 → 采样器生成anchor和candidate_set → 为每个anchor生成最优ICD序列 → 保存结果
```

---

## 文件1: `configs/generate_data.yaml` - 配置文件

**作用**：定义整个数据生成流程的配置参数

**关键配置**：
```yaml
few_shot_num: 4        # 要选择的ICD数量
beam_size: 2           # Beam Search保留的序列数
scorer: "infoscore"    # 评分函数类型
batch_size: 16         # 批处理大小
metric: "no_order"     # 插入策略

sampler:
  candidate_num: 64    # 每个anchor的候选集大小
  anchor_sample_num: 5000  # anchor数量
```

**流程位置**：第0步 - 配置初始化

---

## 文件2: `lever_lm/load_ds_utils.py` - 数据集加载

**作用**：加载和预处理数据集，为每个样本添加必要的字段

### `load_gsm8k_ds()` 函数

**执行流程**：
1. **加载数据**：从parquet文件加载训练集和验证集
   ```python
   ds = load_dataset("parquet", data_files={"train": train_path, "validation": val_path})
   ```

2. **添加索引字段**：为每个split添加全局唯一的`idx`
   ```python
   ds[split] = ds[split].add_column("idx", list(range(len(ds[split]))))
   # train: idx = [0, 1, 2, ..., len(train)-1]
   # validation: idx = [0, 1, 2, ..., len(validation)-1]
   ```

3. **添加isquery字段**：初始化为0（表示ICD，不是query）
   ```python
   ds[split] = ds[split].add_column("isquery", [0]*len(ds[split]))
   ```

4. **格式化数据**：将question和answer拼接成ICL格式
   ```python
   q_a = f"question:{q}\n<answer>\n{a}\n</answer>"
   ```

**输出**：
- `train_ds`: 训练数据集，包含`idx`, `isquery`, `question`, `answer`, `q_a`字段
- `validation_ds`: 验证数据集（当前流程不使用）

**流程位置**：第1步 - 数据准备

---

## 文件3: `lever_lm/candidate_sampler/base_sampler.py` - 采样器基类

**作用**：定义采样器的通用逻辑，包括anchor采样和缓存机制

### 关键方法：

#### `__init__()` - 初始化
- 设置采样参数（`candidate_num`, `anchor_sample_num`等）
- 设置缓存文件路径
- 调用`sample_anchor_set()`生成或加载anchor集合

#### `sample_anchor_set()` - 采样anchor集合
**执行流程**：
1. 检查缓存文件是否存在
2. 如果存在且`overwrite=False`，直接加载
3. 否则，随机采样`anchor_sample_num`个索引
4. 保存到缓存文件

**输出**：
```python
anchor_idx_list = [0, 1, 2, ..., anchor_sample_num-1]  # 例如：[100, 250, 500, ...]
```

#### `__call__()` - 调用采样器
**执行流程**：
1. 构建返回字典，包含`anchor_set`
2. 尝试加载缓存的`candidate_set`
3. 如果缓存不存在，调用`sample()`方法生成
4. 保存`candidate_set`到缓存

**输出格式**：
```python
{
    "anchor_set": [100, 250, 500, ...],  # anchor索引列表
    "candidate_set": {
        100: [10, 20, 30, ..., 1000],     # anchor 100的候选集
        250: [15, 25, 35, ..., 2000],     # anchor 250的候选集
        ...
    }
}
```

**流程位置**：第2步 - Anchor和候选集生成

---

## 文件4: `lever_lm/candidate_sampler/random_sampler.py` - 随机采样器

**作用**：实现具体的采样逻辑（随机采样）

### `sample()` 方法

**执行流程**：
```python
for s_idx in anchor_set:  # 对每个anchor
    # 1. 随机采样candidate_num个候选索引
    random_candidate_set = random.sample(range(0, len(train_ds)), self.candidate_num)
    
    # 2. 确保anchor本身不在候选集中
    while s_idx in random_candidate_set:
        random_candidate_set = random.sample(...)
    
    # 3. 保存到字典
    candidate_set_idx[s_idx] = random_candidate_set
```

**输出**：
```python
{
    100: [10, 20, 30, ..., 1000],  # 64个候选索引（排除100本身）
    250: [15, 25, 35, ..., 2000],  # 64个候选索引（排除250本身）
    ...
}
```

**关键点**：
- 每个anchor都有自己独立的候选集
- 候选集大小 = `candidate_num`（例如64）
- anchor本身不会出现在自己的候选集中

**流程位置**：第2步 - 候选集生成（具体实现）

---

## 文件5: `generate_data_main.py` - 主脚本

**作用**：协调整个数据生成流程

### `main()` 函数执行流程：

#### 步骤1: 加载数据集
```python
ds = load_gsm8k_ds(...)
train_ds = ds["train"]  # 训练集，包含所有可能的ICD
```

#### 步骤2: 使用sampler生成anchor和candidate_set
```python
sampler = RandSampler(...)
sampler_result = sampler(train_ds)
# 返回: {
#     "anchor_set": [100, 250, 500, ...],
#     "candidate_set": {100: [10, 20, ...], 250: [15, 25, ...], ...}
# }
```

#### 步骤3: 为每个anchor生成最优ICD序列
```python
all_results = generate_icd_for_all_anchors(
    train_ds=train_ds,
    sampler_result=sampler_result,
    interface=interface,
    cfg=cfg,
)
```

#### 步骤4: 保存结果
```python
# 保存为JSON文件
{
    anchor_id_1: {
        "id_list": [[icd1, icd2, query], [icd3, icd4, query], ...],
        "score_list": [0.95, 0.93, ...]
    },
    anchor_id_2: {...},
    ...
}
```

**流程位置**：主流程协调

---

### `generate_icd_for_all_anchors()` 函数

**执行流程**：
```python
for anchor_idx in anchor_set:  # 遍历每个anchor
    # 1. 获取anchor数据，标记为query
    anchor_data = train_ds[anchor_idx].copy()
    anchor_data["isquery"] = 1
    
    # 2. 获取该anchor对应的候选集索引
    candidate_indices = candidate_set_dict[anchor_idx]  # 例如：[10, 20, 30, ...]
    
    # 3. 从训练集中提取候选数据
    candidate_data_list = [train_ds[idx] for idx in candidate_indices]
    candidate_set = Dataset.from_list(candidate_data_list)
    
    # 4. 为当前anchor生成最优ICD序列
    result = generate_single_sample_icd(
        interface=interface,
        test_data=anchor_data,      # anchor作为query
        cfg=cfg,
        candidate_set=candidate_set,  # 该anchor的候选ICD集合
        metric=cfg.metric,
    )
```

**关键点**：
- **anchor = test**：anchor样本被当作test（query）处理
- **候选集**：每个anchor都有自己独立的候选ICD集合
- **目标**：从候选集中找到最适配的ICD序列

**流程位置**：第3步 - 为每个anchor生成ICD序列

---

## 文件6: `generate_data.py` - ICD生成核心逻辑

**作用**：为单个test（anchor）从候选集中选择最优的ICD序列

### `generate_single_sample_icd()` 函数

**输入**：
- `test_data`: 测试样本（anchor），`isquery=1`
- `candidate_set`: 候选ICD数据集（64个样本）
- `cfg`: 配置对象（`few_shot_num=4`, `beam_size=2`等）

**执行流程**：

#### 阶段1: 初始化（第105-122行）
```python
# 1. 获取test_data的ID
test_data_id = test_data["idx"]  # 例如：100

# 2. 标记为query
test_data["isquery"] = 1

# 3. 构建ID到数据的映射表
candidateidx2data = {data["idx"]: data for data in candidate_set}
candidateidx2data[test_data_id] = test_data  # 添加query本身

# 4. 初始化序列：只包含query
test_data_id_list = [[test_data_id]]  # [[100]]
```

**关键数据结构**：
- `candidateidx2data`: 所有数据的ID映射（包括候选ICD和query）
- `test_data_id_list`: 当前最优序列列表（Beam Search）

#### 阶段2: 迭代构建ICD序列（第124-245行）

**外层循环**：`for step in range(cfg.few_shot_num)`（例如4次）

**每次迭代的目标**：添加1个最优ICD到序列中

**内层循环**：`for test_data_id_seq in test_data_id_list`（遍历当前所有候选序列）

**步骤2.1: 过滤已使用的候选（第129-137行）**
```python
# 复制候选集
filtered_candidateidx2data = candidateidx2data.copy()

# 移除已经在序列中的元素
for idx in test_data_id_seq:  # 例如：[67, 100]
    if idx in filtered_candidateidx2data:
        filtered_candidateidx2data.pop(idx)  # 移除67和100

# 结果：filtered_candidateidx2data只包含未使用的候选ICD
```

**步骤2.2: 找到query位置（第139-143行）**
```python
query_pos = find_query_position(test_data_id_seq, candidateidx2data)
# 例如：[67, 100] -> query_pos = 1（query在位置1）
```

**步骤2.3: 获取插入位置（第145-146行）**
```python
insert_positions = get_insert_positions(test_data_id_seq, query_pos, metric)
# metric="no_order": [0, 2]  # 只在两侧插入
# metric="order": [0, 1, 2]   # 所有位置都可以插入
```

**步骤2.4: 构建当前序列数据（第148-150行）**
```python
current_seq_data = [candidateidx2data[idx] for idx in test_data_id_seq]
# 例如：[ICD_67的数据, Query_100的数据]
```

**步骤2.5: 构建所有候选-位置组合（第159-169行）**
```python
candidate_position_pairs = []
candidate_data_list = []
position_list = []

for candidate_idx in filtered_idx_list:  # 例如：[10, 20, 30, ...]
    for insert_pos in insert_positions:   # 例如：[0, 2]
        candidate_position_pairs.append((candidate_idx, insert_pos))
        # 例如：(10, 0), (10, 2), (20, 0), (20, 2), ...
        candidate_data_list.append(filtered_candidateidx2data[candidate_idx])
        position_list.append(insert_pos)
```

**示例**：
- 候选ICD数量：60个（64 - 4个已使用）
- 插入位置：2个（[0, 2]）
- 总组合数：60 × 2 = 120个

**步骤2.6: 批量计算分数（第175-196行）**
```python
scores = get_info_score(
    interface,
    choosed_icd_seq_list=current_seq_data,  # 当前序列：[ICD_67, Query_100]
    candidate_data_list=candidate_data_list,  # 120个候选数据
    position_list=position_list,              # 120个插入位置
    batch_size=cfg.batch_size,
    split_token=cfg.task.split_token,
)
# 返回: tensor([0.75, 0.68, 0.92, 0.85, ...])  # shape: (120,)
```

**步骤2.7: 选择top-k（第209-227行）**
```python
# 选择分数最高的beam_size个
topk_scores, indices = scores.topk(cfg.beam_size)  # 例如：top-2

# 构建新序列
for idx, score in zip(indices, topk_scores):
    candidate_idx, insert_pos = candidate_position_pairs[idx]
    # 例如：idx=50 -> (20, 0), score=0.92
    
    new_seq_ids = test_data_id_seq.copy()  # [67, 100]
    new_seq_ids.insert(insert_pos, candidate_idx)  # [20, 67, 100]
    
    new_test_data_id_list.append(new_seq_ids)
    new_test_score_list.append(score)
```

**步骤2.8: Beam Search过滤（第236-242行）**
```python
# 对所有分支进行全局筛选，保留top beam_size个
new_test_score_list, new_test_data_id_list = beam_filter(
    new_test_score_list, new_test_data_id_list, cfg.beam_size
)
# 例如：从所有序列中选择top-2
```

**迭代过程示例**：

```
初始: [[100]]
Step 1: [[20, 100], [67, 100]]  # 添加第1个ICD，保留top-2
Step 2: [[20, 45, 100], [20, 67, 100], [67, 20, 100], [67, 45, 100]]  # 添加第2个ICD
        -> Beam filter -> [[20, 45, 100], [20, 67, 100]]  # 保留top-2
Step 3: ...  # 添加第3个ICD
Step 4: ...  # 添加第4个ICD
最终: [[20, 45, 67, 89, 100], [20, 45, 67, 90, 100]]  # 4个ICD + 1个query
```

**输出**：
```python
{
    test_data_id: {
        "id_list": [
            [20, 45, 67, 89, 100],  # 最优序列1
            [20, 45, 67, 90, 100]   # 最优序列2
        ],
        "score_list": [0.95, 0.93]
    }
}
```

**流程位置**：第4步 - 核心ICD生成逻辑

---

## 文件7: `utils.py` - 评分函数

**作用**：计算InfoScore，评估插入某个ICD后的收益

### `get_info_score()` 函数

**输入**：
- `choosed_icd_seq_list`: 当前序列（插入前）
- `candidate_data_list`: 所有要插入的候选数据
- `position_list`: 所有插入位置

**当前实现**（占位）：
```python
# TODO: 实现真实的InfoScore计算
# 应该计算：
# 1. 插入前的置信度 P(y|x, current_icds)
# 2. 插入后的置信度 P(y|x, current_icds + candidate)
# 3. InfoScore = 插入后 - 插入前

# 临时返回随机分数
scores = torch.randn(num_scores, dtype=torch.float32)
```

**预期逻辑**：
```python
for candidate_data, position in zip(candidate_data_list, position_list):
    # 1. 构建插入后的序列
    new_seq = insert_candidate(current_seq, candidate_data, position)
    
    # 2. 计算插入前后的置信度差异
    score_before = compute_confidence(current_seq, query)
    score_after = compute_confidence(new_seq, query)
    
    # 3. InfoScore = 收益
    info_score = score_after - score_before
```

**流程位置**：第5步 - 评分计算

---

## 完整流程总结

### 数据流：

```
1. 配置文件 (generate_data.yaml)
   ↓
2. 数据集加载 (load_gsm8k_ds)
   - 加载训练集和验证集
   - 添加idx和isquery字段
   - 格式化数据（q_a字段）
   ↓
3. 采样器 (RandSampler)
   - 采样anchor集合（例如5000个）
   - 为每个anchor采样候选集（例如64个候选ICD）
   ↓
4. 主脚本 (generate_data_main.py)
   - 遍历每个anchor
   - 调用generate_single_sample_icd
   ↓
5. ICD生成 (generate_single_sample_icd)
   - 初始化：只有query的序列
   - 迭代few_shot_num次：
     a. 过滤已使用的候选
     b. 找到query位置
     c. 获取插入位置
     d. 构建所有候选-位置组合
     e. 批量计算分数（get_info_score）
     f. 选择top-k（局部筛选）
     g. Beam Search过滤（全局筛选）
   - 返回最优ICD序列
   ↓
6. 保存结果
   - JSON格式保存所有anchor的最优ICD序列
```

### 关键数据结构：

1. **sampler_result**:
   ```python
   {
       "anchor_set": [100, 250, 500, ...],
       "candidate_set": {
           100: [10, 20, 30, ..., 1000],
           250: [15, 25, 35, ..., 2000],
           ...
       }
   }
   ```

2. **candidateidx2data**:
   ```python
   {
       10: {"idx": 10, "question": "...", "answer": "...", "isquery": 0},
       20: {"idx": 20, "question": "...", "answer": "...", "isquery": 0},
       100: {"idx": 100, "question": "...", "answer": "...", "isquery": 1},  # query
       ...
   }
   ```

3. **test_data_id_list** (Beam Search):
   ```python
   [[100]]  # 初始
   [[20, 100], [67, 100]]  # Step 1后
   [[20, 45, 100], [20, 67, 100]]  # Step 2后
   ...
   ```

### 关键参数：

- **anchor_sample_num**: 5000（要处理的test数量）
- **candidate_num**: 64（每个test的候选ICD数量）
- **few_shot_num**: 4（要选择的ICD数量）
- **beam_size**: 2（Beam Search保留的序列数）
- **metric**: "no_order"（插入策略）

### 计算复杂度：

- **候选-位置组合数**: `candidate_num × insert_positions`
  - Step 1: 64 × 2 = 128
  - Step 2: 63 × 2 = 126（排除1个已使用）
  - Step 3: 62 × 2 = 124
  - Step 4: 61 × 2 = 122

- **总计算次数**: 约500次评分调用（每步约125个组合）

---

## 待实现部分

1. **`get_info_score()` 函数**：
   - 需要实现真实的InfoScore计算
   - 计算插入前后的置信度差异

2. **`interface` 初始化**：
   - 在`generate_data_main.py`中需要初始化interface
   - interface用于计算InfoScore

---

## 总结

整个流程的核心思想：
1. **采样阶段**：为每个test（anchor）准备候选ICD池
2. **搜索阶段**：使用Beam Search逐步构建最优ICD序列
3. **评分阶段**：使用InfoScore评估每个候选ICD的收益
4. **选择阶段**：选择分数最高的ICD序列

这样的设计允许：
- 灵活的位置插入（metric参数）
- 高效的批量评分
- 全局最优搜索（Beam Search）

