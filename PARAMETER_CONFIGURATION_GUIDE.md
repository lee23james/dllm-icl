# 参数配置指南

## 参数分类

### 1. Shell脚本中控制的参数（经常需要调整）

这些参数可以通过 `run_stage1.sh` 直接控制，适合频繁调整：

#### Sampler参数
- `sampler.anchor_sample_num`: test_sample数量（默认：5000）
- `sampler.candidate_num`: 候选池大小（默认：64）
- `sampler.overwrite`: 是否覆盖缓存（默认：false）
- `sampler.cache_dir`: 缓存目录（默认："./cache/gsm8k"）

#### 核心算法参数
- `few_shot_num`: 要选择的ICD数量（默认：4）
- `beam_size`: Beam Search大小（默认：2）
- `metric`: 插入策略（"no_order" 或 "order"，默认："no_order"）
- `scorer`: 评分函数（目前只支持 "infoscore"）

#### 性能参数
- `mc_num`: Monte Carlo采样次数（默认：128，影响精度和速度）
- `batch_size`: 批处理大小（默认：1）
- `cfg_scale`: CFG scale（默认：0.0）

#### 多GPU参数
- `use_multi_gpu`: 是否使用多GPU（默认：false）
- `gpu_ids`: GPU ID列表（默认：[0]）
- `sleep_time`: 进程启动间隔（秒，默认：10）
- `merge_results`: 是否合并多卡结果（默认：false）

#### 输出参数
- `output_dir`: 输出目录（默认："./generated_icd_data"）

---

### 2. Config文件中配置的参数（相对固定）

这些参数通常在配置文件中设置，除非需要特殊调整，否则不需要在shell脚本中指定：

#### 模型参数（`configs/infer_model/llada.yaml`）
- `model_path`: **必需** - 模型路径
- `mask_id`: mask token的ID（默认：126336）
- `mask_length`: mask token长度（默认：256）
- `trust_remote_code`: 是否信任远程代码（默认：true）
- `local_files_only`: 是否只使用本地文件（默认：true）
- `torch_dtype`: 模型精度（默认："bfloat16"）

#### 数据集参数（`configs/dataset/gsm8k.yaml`）
- `version`: 数据集版本（默认："local"）
- `train_path`: 训练集路径
- `val_path`: 验证集路径

#### 任务参数（`configs/task/gsm8k.yaml`）
- `task_name`: 任务名称（默认："gsm8k"）
- `split_token`: ICD之间的分隔符（默认："\n\n"）

#### Sampler基础参数（`configs/sampler/random.yaml`）
- `sampler_name`: 采样器名称（默认："random_sampler"）

---

## 使用方式

### 方式1：使用Shell脚本（推荐）

**适合：** 经常需要调整的参数

```bash
# 基本使用
./run_stage1.sh -s

# 覆盖常用参数
./run_stage1.sh -s \
    sampler.anchor_sample_num=100 \
    sampler.candidate_num=32 \
    few_shot_num=3 \
    beam_size=10 \
    mc_num=64

# 覆盖模型路径（如果需要）
./run_stage1.sh -s infer_model.model_path="/path/to/model"
```

### 方式2：直接修改Config文件

**适合：** 相对固定的参数（如模型路径、数据集路径）

**修改文件：**
- `configs/infer_model/llada.yaml` - 模型相关
- `configs/dataset/gsm8k.yaml` - 数据集相关
- `configs/task/gsm8k.yaml` - 任务相关
- `configs/generate_data.yaml` - 默认参数

**示例：**
```yaml
# configs/infer_model/llada.yaml
model_path: "/path/to/your/llada/model"  # 修改这里
torch_dtype: "bfloat16"                    # 或修改这里
```

### 方式3：混合使用

**推荐方式：** 固定参数在config文件中配置，经常调整的参数在shell脚本中控制

```bash
# 1. 先在config文件中设置模型路径等固定参数
# configs/infer_model/llada.yaml
model_path: "/path/to/model"

# 2. 在shell脚本中控制经常调整的参数
./run_stage1.sh -s \
    sampler.anchor_sample_num=100 \
    sampler.candidate_num=32 \
    mc_num=64
```

---

## 参数优先级

**优先级从高到低：**

1. **Shell脚本命令行参数**（最高优先级）
   ```bash
   ./run_stage1.sh -s sampler.candidate_num=50
   ```

2. **Shell脚本预设配置**
   ```bash
   # run_stage1.sh 中的 SMALL_TEST_CONFIG
   ```

3. **Config文件中的参数**
   ```yaml
   # configs/generate_data.yaml
   few_shot_num: 4
   ```

4. **Config文件的默认值**
   ```yaml
   # configs/sampler/random.yaml
   candidate_num: 64
   ```

---

## 常用参数组合

### 快速测试（最快速度）
```bash
./run_stage1.sh -s \
    sampler.anchor_sample_num=10 \
    sampler.candidate_num=10 \
    few_shot_num=2 \
    beam_size=1 \
    mc_num=32
```

### 平衡速度和精度
```bash
./run_stage1.sh -m \
    sampler.candidate_num=32 \
    few_shot_num=4 \
    beam_size=10 \
    mc_num=128
```

### 高精度（较慢）
```bash
./run_stage1.sh -l \
    sampler.candidate_num=64 \
    few_shot_num=4 \
    beam_size=50 \
    mc_num=256 \
    metric=order
```

### 多GPU运行
```bash
./run_stage1.sh -l \
    use_multi_gpu=true \
    gpu_ids=[0,1,2,3] \
    sleep_time=10 \
    merge_results=true
```

---

## 参数说明表

| 参数 | 类型 | 默认值 | 说明 | 推荐范围 |
|------|------|--------|------|---------|
| `sampler.anchor_sample_num` | Shell | 5000 | test_sample数量 | 10-5000 |
| `sampler.candidate_num` | Shell | 64 | 候选池大小 | 10-100 |
| `few_shot_num` | Shell | 4 | ICD数量 | 2-8 |
| `beam_size` | Shell | 2 | Beam Search大小 | 1-50 |
| `metric` | Shell | "no_order" | 插入策略 | "no_order"/"order" |
| `mc_num` | Shell | 128 | Monte Carlo次数 | 32-512 |
| `batch_size` | Shell | 1 | 批处理大小 | 1 |
| `use_multi_gpu` | Shell | false | 是否多GPU | true/false |
| `gpu_ids` | Shell | [0] | GPU列表 | [0,1,2,3] |
| `model_path` | Config | - | 模型路径 | **必需** |
| `mask_id` | Config | 126336 | mask token ID | 固定 |
| `mask_length` | Config | 256 | mask token长度 | 固定 |
| `torch_dtype` | Config | "bfloat16" | 模型精度 | "bfloat16"/"float16" |
| `train_path` | Config | - | 训练集路径 | **必需** |

---

## 常见问题

### Q1: 哪些参数必须在config文件中配置？

**必需参数：**
- `infer_model.model_path` - 模型路径（**必须**）
- `dataset.train_path` - 训练集路径（**必须**）

**推荐在config文件中配置：**
- 模型相关参数（mask_id, mask_length等）
- 数据集路径
- 任务名称

### Q2: 哪些参数应该在shell脚本中控制？

**推荐在shell脚本中控制：**
- test_sample数量（`sampler.anchor_sample_num`）
- 候选池大小（`sampler.candidate_num`）
- ICD数量（`few_shot_num`）
- Beam Search大小（`beam_size`）
- Monte Carlo采样次数（`mc_num`）
- 多GPU相关参数

### Q3: 如何查看所有可用参数？

```bash
# 查看帮助
./run_stage1.sh -h

# 查看配置文件
cat configs/generate_data.yaml
cat configs/infer_model/llada.yaml
cat configs/sampler/random.yaml
```

### Q4: 参数冲突怎么办？

Shell脚本中的参数优先级最高，会覆盖config文件中的设置。

```bash
# 即使config文件中设置了 few_shot_num=4
# 这个命令会使用 few_shot_num=3
./run_stage1.sh -s few_shot_num=3
```

---

## 最佳实践

1. **首次运行前：**
   - ✅ 在 `configs/infer_model/llada.yaml` 中设置模型路径
   - ✅ 在 `configs/dataset/gsm8k.yaml` 中确认数据集路径

2. **日常使用：**
   - ✅ 使用shell脚本控制经常调整的参数
   - ✅ 使用预设配置（-s, -m, -l, -o）
   - ✅ 通过命令行参数覆盖预设配置

3. **大规模运行：**
   - ✅ 先小批量测试（-s）
   - ✅ 逐步增加规模（-m → -l）
   - ✅ 使用多GPU加速（use_multi_gpu=true）
