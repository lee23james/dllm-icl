# 数据导入和打分流程详细说明

## 1. 数据导入方式

### ✅ 从本地导入（当前实现）

**数据格式：** Parquet文件（HuggingFace datasets格式）

**导入函数：** `lever_lm/load_ds_utils.py::load_gsm8k_ds()`

**导入流程：**
```python
# 方式1：只加载单个split（Stage 1使用）
train_ds = load_gsm8k_ds(
    version="local",                    # 使用本地数据
    data_path="/path/to/train.parquet", # 本地parquet文件路径
    split="train"                       # 要加载的split
)

# 方式2：同时加载train和validation（Stage 2使用）
ds = load_gsm8k_ds(
    version="local",
    train_path="/path/to/train.parquet",
    val_path="/path/to/val.parquet"
)
```

**数据处理：**
1. 从parquet文件加载数据
2. 添加 `idx` 字段（全局唯一索引）
3. 添加 `isquery` 字段（初始值为0，标记为test后变为1）
4. 处理数据格式：`question` + `answer` → `q_a` 字段
   - 格式：`"question:{question}\n<answer>\n{answer}\n</answer>"`

**配置文件：** `configs/dataset/gsm8k.yaml`
```yaml
version: "local"  # 使用本地数据
train_path: "/home/share/datasets/train-00000-of-00001.parquet"
val_path: "/home/share/datasets/validation-00000-of-00001.parquet"
```

**注意：**
- ✅ 当前实现**只支持本地导入**（`version="local"`）
- ✅ 数据文件必须是parquet格式
- ✅ 数据文件需要包含 `question` 和 `answer` 字段

---

## 2. 打分函数流程确认

### ✅ 你的描述与代码实现完全一致！

**你描述的流程：**
1. 针对单个test_sample，随机选择一个候选池
2. 针对候选池里的数据集，根据策略依次插入例子
3. 计算插入之后和插入之前的分数
4. 从上到下排序
5. 选出beam_size个例子，选出的不能再选
6. 按照这个步骤从候选池中不断找到最好的例子

**代码实现流程：**

#### 步骤1：随机选择候选池
```python
# generate_data_main.py
sampler = RandSampler(...)
sampler_result = sampler(train_ds)
# 返回：{anchor_idx: [candidate_idx1, candidate_idx2, ...]}
```

#### 步骤2-6：核心算法（`generate_single_sample_icd`）

```python
# 初始化：只包含query的序列
test_data_id_list = [[test_data_id]]

# 循环few_shot_num轮（每轮选择一个ICD）
for step in range(cfg.few_shot_num):
    new_test_data_id_list = []
    new_test_score_list = []
    
    # 对当前每个序列分支进行处理
    for test_data_id_seq in test_data_id_list:
        # 2. 过滤已使用的ICD（选出的不能再选）
        filtered_candidateidx2data = candidateidx2data.copy()
        for idx in test_data_id_seq:
            filtered_candidateidx2data.pop(idx)  # 移除已使用的
        
        # 3. 获取所有可能的插入位置（根据策略）
        insert_positions = get_insert_positions(test_data_id_seq, metric)
        # metric="no_order": 只在两侧插入 [0, len(seq)]
        # metric="order": 所有位置插入 [0, 1, 2, ..., len(seq)]
        
        # 4. 构建所有(候选, 位置)组合
        candidate_position_pairs = []
        candidate_data_list = []
        position_list = []
        for candidate_idx in filtered_idx_list:
            for insert_pos in insert_positions:
                candidate_position_pairs.append((candidate_idx, insert_pos))
                candidate_data_list.append(filtered_candidateidx2data[candidate_idx])
                position_list.append(insert_pos)
        
        # 5. 批量计算分数（插入前 vs 插入后）
        scores = get_info_score(
            interface,
            choosed_icd_seq_list=current_seq_data,  # 插入前的序列
            candidate_data_list=candidate_data_list,  # 所有候选
            position_list=position_list,  # 所有位置
            ...
        )
        # get_info_score内部：
        #   - 计算插入前的baseline分数
        #   - 对每个(候选, 位置)组合：
        #     - 构建插入后的序列
        #     - 计算插入后的分数
        #     - InfoScore = score_after - score_before
        
        # 6. 排序并选出top-k（beam_size）
        topk_scores, indices = scores.topk(cfg.beam_size)
        
        # 7. 构建新序列（选出的不能再选）
        for idx, score in zip(indices, topk_scores):
            candidate_idx, position = candidate_position_pairs[idx]
            new_seq_ids = test_data_id_seq.copy()
            new_seq_ids.insert(real_insert_pos, candidate_idx)
            new_test_data_id_list.append(new_seq_ids)
            new_test_score_list.append(score)
    
    # 8. Beam filter：保留top beam_size个序列
    new_test_score_list, new_test_data_id_list = beam_filter(
        new_test_score_list, new_test_data_id_list, cfg.beam_size
    )
    test_data_id_list = new_test_data_id_list
```

**关键点确认：**

| 你的描述 | 代码实现 | ✅ |
|---------|---------|---|
| 随机选择候选池 | `RandSampler.sample()` | ✅ |
| 根据策略依次插入例子 | `get_insert_positions()` + 循环 | ✅ |
| 计算插入前后分数 | `get_info_score()` 内部实现 | ✅ |
| 从上到下排序 | `scores.topk(beam_size)` | ✅ |
| 选出beam_size个，选出的不能再选 | `filtered_candidateidx2data.pop()` | ✅ |
| 不断找到最好的例子 | 循环 `few_shot_num` 轮 | ✅ |

---

## 3. 详细流程图示

```
开始
  ↓
1. 加载数据集（本地parquet文件）
  ↓
2. 随机选择候选池（RandSampler）
  ├─ anchor_set: [idx1, idx2, ...]
  └─ candidate_set: {idx1: [cand1, cand2, ...], ...}
  ↓
3. 对每个anchor（test_sample）：
  ├─ 初始化：test_data_id_list = [[query_id]]
  │
  ├─ 循环few_shot_num轮：
  │   ├─ 对每个序列分支：
  │   │   ├─ 过滤已使用的ICD（选出的不能再选）
  │   │   ├─ 获取插入位置（根据策略）
  │   │   ├─ 构建所有(候选, 位置)组合
  │   │   ├─ 批量计算InfoScore
  │   │   │   ├─ 计算插入前的baseline分数
  │   │   │   └─ 对每个组合：
  │   │   │       ├─ 构建插入后的序列
  │   │   │       ├─ 计算插入后的分数
  │   │   │       └─ InfoScore = score_after - score_before
  │   │   ├─ 排序并选出top-k（beam_size）
  │   │   └─ 构建新序列
  │   └─ Beam filter：保留top beam_size个序列
  │
  └─ 返回：{anchor_id: {"id_list": [...], "score_list": [...]}}
  ↓
4. 保存结果到JSON文件
```

---

## 4. InfoScore计算细节

**公式：** `InfoScore = score_after - score_before`

**实现位置：** `utils.py::get_info_score()`

**详细步骤：**

1. **定位query位置**
   ```python
   query_idx = None
   for i, item in enumerate(choosed_icd_seq_list):
       if item.get("isquery", 0) == 1:
           query_idx = i
           break
   ```

2. **分割序列**
   ```python
   left_icd_list = choosed_icd_seq_list[:query_idx]  # query前面的ICD
   query = choosed_icd_seq_list[query_idx]          # query本身
   right_icd_list = choosed_icd_seq_list[query_idx+1:]  # query后面的ICD
   ```

3. **计算插入前的baseline分数**
   ```python
   score_before = interface.compute_log_likelihood(
       prompt_left=prompt_left_before,
       answer=answer_ids,
       prompt_right=prompt_right_before,
       mc_num=mc_num,
       batch_size=batch_size,
       cfg_scale=cfg_scale,
   )
   ```

4. **对每个(候选, 位置)组合：**
   ```python
   # 构建插入后的序列
   new_seq = choosed_icd_seq_list.copy()
   insert_pos = len(new_seq) - position  # 反向语义
   new_seq.insert(insert_pos, candidate_data)
   
   # 计算插入后的分数
   score_after = interface.compute_log_likelihood(
       prompt_left=prompt_left_after,
       answer=answer_ids,
       prompt_right=prompt_right_after,
       ...
   )
   
   # 计算InfoScore
   infoscore = score_after - score_before
   ```

---

## 5. 关键配置参数

### 候选池相关
- `sampler.candidate_num`: 候选池大小（每个anchor的候选ICD数量）
- `sampler.anchor_sample_num`: anchor数量（要处理的test_sample数量）

### 筛选策略相关
- `few_shot_num`: 要选择的ICD数量（循环轮数）
- `beam_size`: Beam Search的beam大小（每轮保留的序列数）
- `metric`: 插入策略
  - `"no_order"`: 只在两侧插入（开头和末尾）
  - `"order"`: 所有位置插入

### 评分相关
- `scorer`: 评分函数（目前只支持"infoscore"）
- `batch_size`: 批处理大小（一般设为1）
- `mc_num`: Monte Carlo采样次数（影响评分精度和速度）

---

## 总结

✅ **数据导入：** 从本地parquet文件导入（`version="local"`）

✅ **打分流程：** 完全按照你描述的逻辑实现
- 随机选择候选池 ✅
- 依次插入例子 ✅
- 计算插入前后分数 ✅
- 排序并选出top-k ✅
- 选出的不能再选 ✅
- 不断找到最好的例子 ✅

✅ **代码实现：** 逻辑清晰，与你的描述完全一致
