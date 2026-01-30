# Configs 配置文件详细说明

## 配置文件结构

```
configs/
├── generate_data.yaml          # Stage 1 主配置文件
├── icl_inference.yaml          # Stage 2 主配置文件
├── dataset/
│   └── gsm8k.yaml             # GSM8K数据集配置
├── task/
│   └── gsm8k.yaml             # GSM8K任务配置
├── sampler/
│   └── random.yaml            # 随机采样器配置
├── infer_model/
│   └── llada.yaml             # LLaDA模型配置
└── train/                     # 训练配置（暂未使用）
```

---

## 1. `generate_data.yaml` - Stage 1 主配置文件

**作用：** Stage 1（数据生成）的主配置文件，通过 `defaults` 引入其他配置文件

**使用场景：** 运行 `python generate_data_main.py` 时使用

**关键内容：**

### defaults（引入其他配置）
```yaml
defaults:
  - task/gsm8k          # 引入任务配置
  - dataset/gsm8k      # 引入数据集配置
  - sampler/random     # 引入采样器配置
  - infer_model/llada  # 引入模型配置
```

### 核心算法参数
```yaml
few_shot_num: 4        # 要选择的ICD数量（循环轮数）
beam_size: 2          # Beam Search的beam大小（每轮保留的序列数）
scorer: "infoscore"   # 评分函数（目前只支持 "infoscore"）
batch_size: 1         # 批处理大小（InfoScore计算时使用，一般设为1）
metric: "no_order"    # 插入策略："no_order"（只在两侧）或 "order"（所有位置）
```

### 多GPU配置
```yaml
use_multi_gpu: false   # 是否使用多GPU
gpu_ids: [0]          # GPU ID列表，例如 [0, 1, 2, 3]
sleep_time: 10        # 多卡模式下，各进程启动间隔（秒）
```

### 输出配置
```yaml
output_dir: "./generated_icd_data"  # 输出目录
merge_results: false                # 多卡模式下，是否合并所有rank的结果
```

### 测试配置
```yaml
test_num_samples: 5    # 测试用的test样本数量
test_candidate_num: 10 # 测试用的候选池大小
mc_num: 20            # Monte Carlo采样次数（可以减小以加快速度）
cfg_scale: 0.0        # CFG scale
```

**注意：** 
- `mc_num` 在 `generate_data.yaml` 中定义，但实际使用时需要通过 `get_info_score()` 传递
- 这些参数可以通过shell脚本覆盖：`./run_stage1.sh -s mc_num=128`

---

## 2. `icl_inference.yaml` - Stage 2 主配置文件

**作用：** Stage 2（ICL推理）的主配置文件

**使用场景：** 运行 `python icl_inference.py` 时使用

**关键内容：**

### defaults（引入其他配置）
```yaml
defaults:
  - task/gsm8k          # 引入任务配置
  - dataset/gsm8k      # 引入数据集配置
  - infer_model/llada  # 引入模型配置
```

### Retriever配置
```yaml
retriever:
  nshot: 4    # few-shot示例数量（从训练集中选择多少个ICD）
  seed: 42    # 随机种子（用于可复现性）
```

### Query位置配置
```yaml
query_position: 0  # query在ICD序列中的位置
                   # 0表示最后（ICD在前，query在后）
                   # nshot表示最前（query在前，ICD在后）
```

### 输出配置
```yaml
output_dir: "./icl_inference_results"  # 输出目录
max_samples: null                      # 限制测试样本数量（null表示全部，用于快速测试）
```

**注意：** 
- Stage 2 不使用 `sampler`，而是使用 `retriever`（如 `RandRetriever`）
- `max_samples` 可以设置为数字来限制测试样本数量，例如 `max_samples: 100`

---

## 3. `dataset/gsm8k.yaml` - 数据集配置

**作用：** 定义GSM8K数据集的路径和加载方式

**使用场景：** 被 `generate_data.yaml` 和 `icl_inference.yaml` 引入

**关键内容：**

```yaml
version: "local"  # 数据集版本（目前只支持 "local"）
train_path: "/home/share/datasets/train-00000-of-00001.parquet"      # 训练集路径
val_path: "/home/share/datasets/validation-00000-of-00001.parquet"   # 验证集路径
test_path: "/home/share/datasets/test-00000-of-00001.parquet"        # 测试集路径
```

**说明：**
- `version: "local"` 表示从本地parquet文件加载数据
- `train_path`: Stage 1 使用（用于生成ICD序列）
- `val_path`: 用于 `load_gsm8k_ds()` 的双路径加载模式
- `test_path`: Stage 2 使用（用于ICL推理和评估）

**数据格式要求：**
- 必须是parquet格式
- 必须包含 `question` 和 `answer` 字段
- 数据加载时会自动添加 `idx` 和 `isquery` 字段

**如何修改：**
```yaml
# 修改为你的数据路径
train_path: "/your/path/to/train.parquet"
val_path: "/your/path/to/val.parquet"
test_path: "/your/path/to/test.parquet"
```

---

## 4. `task/gsm8k.yaml` - 任务配置

**作用：** 定义GSM8K任务相关的配置

**使用场景：** 被 `generate_data.yaml` 和 `icl_inference.yaml` 引入

**关键内容：**

```yaml
task_name: "gsm8k"      # 任务名称（用于识别任务类型）
split_token: "\n\n"     # ICD之间的分隔符
```

**说明：**
- `task_name`: 用于任务识别，目前代码中主要用于判断使用哪个数据集加载函数
- `split_token`: 用于分隔不同的ICD示例，在构建prompt时使用

**使用示例：**
```python
# 在代码中访问
cfg.task.task_name      # "gsm8k"
cfg.task.split_token    # "\n\n"
```

**如何修改：**
- 如果要支持其他任务（如mbpp、sudoku），可以创建 `task/mbpp.yaml` 等
- 修改 `split_token` 来改变ICD之间的分隔方式

---

## 5. `sampler/random.yaml` - 采样器配置

**作用：** 定义随机采样器的配置（仅用于Stage 1）

**使用场景：** 被 `generate_data.yaml` 引入

**关键内容：**

```yaml
sampler_name: "random_sampler"  # 采样器名称
candidate_num: 64                # 候选集大小（每个anchor的候选ICD数量）
anchor_sample_num: 5000          # anchor数量（要处理的test_sample数量）
cache_dir: "./cache/gsm8k"       # 缓存目录
overwrite: false                  # 是否覆盖缓存
```

**说明：**
- `sampler_name`: 采样器类型，目前只支持 "random_sampler"
- `candidate_num`: 每个anchor随机采样的候选ICD数量
- `anchor_sample_num`: 从训练集中随机选择多少个样本作为anchor（test_sample）
- `cache_dir`: 缓存anchor和候选集的目录
- `overwrite`: 
  - `false`: 如果缓存存在，直接使用缓存（加快速度）
  - `true`: 强制重新采样，覆盖缓存

**缓存机制：**
- 缓存文件保存在 `cache_dir` 目录下
- 格式：`{dataset_name}-anchor_sample_num:{num}.json`（anchor集合）
- 格式：`{dataset_name}-{sampler_name}-anchor_sample_num:{num}:{candidate_num}.json`（候选集）

**如何修改：**
```yaml
# 修改候选池大小
candidate_num: 100

# 修改anchor数量
anchor_sample_num: 1000

# 修改缓存目录
cache_dir: "./my_cache/gsm8k"

# 强制重新采样
overwrite: true
```

**注意：** 这些参数可以通过shell脚本覆盖：
```bash
./run_stage1.sh -s sampler.candidate_num=32 sampler.anchor_sample_num=100
```

---

## 6. `infer_model/llada.yaml` - 模型配置

**作用：** 定义LLaDA模型的加载和推理配置

**使用场景：** 被 `generate_data.yaml` 和 `icl_inference.yaml` 引入

**关键内容：**

### 模型基础配置
```yaml
mask_id: 126336        # mask token的ID（LLaDA模型专用）
mask_length: 256       # mask token长度
```

### 模型路径配置（必需）
```yaml
model_path: "/path/to/your/llada/model"  # 模型路径（必需，请修改为实际路径）
# 或者使用 model_name（如果模型在 HuggingFace Hub 上）
# model_name: "your-org/llada-model"
```

### 模型加载参数
```yaml
trust_remote_code: true    # 是否信任远程代码（加载自定义模型时需要）
local_files_only: true     # 是否只使用本地文件（不从HuggingFace Hub下载）
torch_dtype: "bfloat16"     # 模型精度：bfloat16, float16, float32
```

### 生成参数（用于Stage 2推理）
```yaml
generation_kwargs:
  steps: 128                    # 采样步数
  gen_length: 128               # 生成长度
  block_length: 128             # 块长度
  temperature: 0.0              # 温度（0.0表示贪婪解码）
  cfg_scale: 0.0                # CFG scale
  remasking: "low_confidence"   # 重掩码策略
```

**说明：**
- `mask_id`: LLaDA模型使用的mask token ID，通常不需要修改
- `mask_length`: mask token的长度，通常不需要修改
- `model_path`: **必需**，必须设置为实际的模型路径
- `torch_dtype`: 
  - `"bfloat16"`: 推荐，平衡精度和速度
  - `"float16"`: 更快但可能精度损失
  - `"float32"`: 最精确但最慢

**如何修改：**
```yaml
# 1. 修改模型路径（必需）
model_path: "/home/user/models/llada-7b"

# 2. 修改模型精度（如果需要）
torch_dtype: "float16"  # 或 "float32"

# 3. 修改生成参数（Stage 2使用）
generation_kwargs:
  steps: 256        # 增加采样步数（更慢但可能更准确）
  gen_length: 512   # 增加生成长度
  temperature: 0.7  # 使用采样（非贪婪）
```

**注意：** 
- `model_path` 可以通过shell脚本覆盖：
  ```bash
  ./run_stage1.sh -s infer_model.model_path="/path/to/model"
  ```
- `generation_kwargs` 主要用于Stage 2，Stage 1主要使用 `compute_log_likelihood`

---

## 配置文件之间的关系

### Stage 1 配置流程

```
generate_data_main.py
    ↓
@hydra.main(config_name="generate_data.yaml")
    ↓
generate_data.yaml
    ├─ defaults:
    │   ├─ task/gsm8k          → task/gsm8k.yaml
    │   ├─ dataset/gsm8k      → dataset/gsm8k.yaml
    │   ├─ sampler/random     → sampler/random.yaml
    │   └─ infer_model/llada  → infer_model/llada.yaml
    │
    └─ 自身参数：
        - few_shot_num
        - beam_size
        - scorer
        - batch_size
        - metric
        - use_multi_gpu
        - gpu_ids
        - output_dir
        - ...
```

### Stage 2 配置流程

```
icl_inference.py
    ↓
@hydra.main(config_name="icl_inference")
    ↓
icl_inference.yaml
    ├─ defaults:
    │   ├─ task/gsm8k          → task/gsm8k.yaml
    │   ├─ dataset/gsm8k      → dataset/gsm8k.yaml
    │   └─ infer_model/llada  → infer_model/llada.yaml
    │
    └─ 自身参数：
        - retriever.nshot
        - retriever.seed
        - query_position
        - output_dir
        - max_samples
```

---

## 配置文件访问方式

### 在代码中访问配置

```python
# 访问主配置文件的参数
cfg.few_shot_num              # 4
cfg.beam_size                 # 2
cfg.use_multi_gpu             # false

# 访问引入的配置文件的参数
cfg.task.task_name            # "gsm8k"
cfg.task.split_token          # "\n\n"
cfg.dataset.train_path        # "/home/share/datasets/..."
cfg.sampler.candidate_num     # 64
cfg.infer_model.model_path    # "/path/to/model"
cfg.infer_model.mask_id       # 126336
cfg.infer_model.generation_kwargs.steps  # 128
```

### 通过命令行覆盖配置

```bash
# 覆盖主配置文件的参数
python generate_data_main.py few_shot_num=3 beam_size=10

# 覆盖引入的配置文件的参数
python generate_data_main.py \
    sampler.candidate_num=32 \
    infer_model.model_path="/path/to/model" \
    task.split_token="\n"
```

---

## 配置文件修改建议

### 首次运行前必须修改

1. **`infer_model/llada.yaml`**
   ```yaml
   model_path: "/path/to/your/llada/model"  # 必需
   ```

2. **`dataset/gsm8k.yaml`**
   ```yaml
   train_path: "/your/path/to/train.parquet"  # 必需
   test_path: "/your/path/to/test.parquet"    # Stage 2必需
   ```

### 根据需求调整

1. **`generate_data.yaml`**
   - `few_shot_num`: 根据需要的ICD数量调整
   - `beam_size`: 根据计算资源调整
   - `mc_num`: 根据精度和速度需求调整

2. **`sampler/random.yaml`**
   - `candidate_num`: 根据候选池大小需求调整
   - `anchor_sample_num`: 根据要处理的test_sample数量调整

3. **`icl_inference.yaml`**
   - `retriever.nshot`: 根据few-shot数量需求调整
   - `max_samples`: 用于快速测试

---

## 配置文件最佳实践

1. **固定参数放在config文件**
   - 模型路径（通常固定）
   - 数据集路径（通常固定）
   - 任务配置（通常固定）

2. **经常调整的参数通过shell脚本控制**
   - test_sample数量
   - 候选池大小
   - ICD数量
   - Beam Search大小

3. **使用环境变量（可选）**
   ```bash
   # 在shell脚本中
   export MODEL_PATH="/path/to/model"
   python generate_data_main.py infer_model.model_path=$MODEL_PATH
   ```

---

## 总结

| 配置文件 | 主要作用 | 使用阶段 | 必需修改 |
|---------|---------|---------|---------|
| `generate_data.yaml` | Stage 1主配置 | Stage 1 | ❌ |
| `icl_inference.yaml` | Stage 2主配置 | Stage 2 | ❌ |
| `dataset/gsm8k.yaml` | 数据集路径 | Stage 1 & 2 | ✅ |
| `task/gsm8k.yaml` | 任务配置 | Stage 1 & 2 | ❌ |
| `sampler/random.yaml` | 采样器配置 | Stage 1 | ❌ |
| `infer_model/llada.yaml` | 模型配置 | Stage 1 & 2 | ✅ |

**必需修改：**
- ✅ `infer_model/llada.yaml` 中的 `model_path`
- ✅ `dataset/gsm8k.yaml` 中的 `train_path` 和 `test_path`

**可选调整：**
- 其他参数可以根据需求在config文件中修改，或通过shell脚本/命令行覆盖
