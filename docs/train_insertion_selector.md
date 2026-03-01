# 训练第二个分类器（InsertionSelector）完整流程

本库已包含第二个分类器的**完整训练链路**：数据准备 → 预计算 embedding → 训练。下面按顺序说明要跑哪些代码、需要哪些文件。

---

## 一、流程概览

```
[可选] Stage1 生成 JSON     →  [可选] 预计算题库 embedding  →  训练 InsertionSelector
     (generate_data)              (precompute_*.py)              (train_insertion.py)
```

- **Step 0（可选）**：若还没有「带插入位置标签」的 JSON，需要先跑 Stage1 的 generate_data，得到 `generated_icd_data/generated_data/xxx.json`。
- **Step 1（可选）**：若**还没有**训练集对应的 embedding 缓存，则运行 `precompute_insertion_embedding.py`，对题库 parquet 用 Qwen 编码，得到 `embedding_cache_path` 的 `.pt`。**若你已有满足格式要求的 embedding 文件，可跳过本步**（见下）。
- **Step 2**：用该 JSON + 该 `.pt` 训练 InsertionSelector，checkpoint 存到 `output.checkpoint_dir`。

### 已有 embedding 时可否跳过 Step 1？

**可以。** 只要你的 embedding 满足以下条件，就**不需要**再跑 `precompute_insertion_embedding.py`：

- 是一个 **单个 `torch.Tensor`**，形状为 **`[N, emb_dim]`**（例如 `[num_parquet_rows, 2560]`）。
- **行号与题库 parquet 一致**：第 `i` 行对应 parquet 第 `i` 行（与训练 JSON 里的 `anchor_id`、`id_list` 中的 id 同一套下标）。
- 训练 config 里 `embedding.embedding_cache_path` 指向该 `.pt` 文件路径。

例如你已有本库其它流程生成的 `generated_icd_data/cache/mmlu/xxx-TextFeatures.pt`（按 parquet 行序的 `[N, D]` tensor），或之前跑过一次 `precompute_insertion_embedding.py` 得到的文件，都可以直接用作 `embedding_cache_path`，只跑 `train_insertion.py` 即可。

---

## 二、Step 0：Stage1 数据（若还没有 JSON）

InsertionSelector 训练用的 JSON 格式为：

```text
{ "anchor_id": { "id_list": [ [id1, id2, ..., anchor_id], ... ], "score_list": [ ... ] } }
```

- `anchor_id`：题库（parquet）中的行号（0-based），表示这条「query」。
- 每个 `id_list` 里是一条序列：若干 ICD 的 id + 一个 anchor（query）id；**anchor 在序列里的 0-based 位置就是插入位置标签**。
- 序列长度 4 或 5（即 3 或 4 个 ICD + 1 个 anchor）。

若你**已有**类似 `generated_icd_data/generated_data/mmlu-mmlu-llada-random_sampler-scorer_infoscore-...json` 的 Stage1 结果，可直接用，跳到 Step 1。

若**没有**，需要先跑本库的 **generate_data** 流程（例如 `generate_data_main.py` + `configs/generate_data.yaml`），生成带 `id_list` / `score_list` 的 JSON。具体命令和配置以仓库内 generate_data 的 README 或脚本为准；生成后路径填到下面 config 的 `data.json_path`。

---

## 三、Step 1：预计算题库 embedding

训练时不再现场跑 Qwen，而是用事先算好的 `[N, emb_dim]` 的 `.pt`（N = 题库 parquet 行数）。

**运行：**

```bash
cd /home/lzh/llada-icl

# 使用默认 config（configs/insertion_selector_train_mmlu.yaml）
python precompute_insertion_embedding.py

# 或用自定义 config / 覆盖路径
python precompute_insertion_embedding.py --config configs/insertion_selector_train_mmlu.yaml
python precompute_insertion_embedding.py --train_path data/MMLU/auxiliary_train/train-00000-of-00001.parquet \
  --embedding_cache_path generated_icd_data/embedding_cache/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt
```

**要求：**

- Config 里 `data.train_path`（或 `--train_path`）指向的 **parquet 必须和 JSON 里的 id 一致**（同一份题库，行号即 id）。
- Config 里 `embedding.embedding_cache_path`（或 `--embedding_cache_path`）为输出的 `.pt` 路径；训练 Step 2 会读这个路径。

**依赖：**  
`lever_lm.load_ds_utils.load_mmlu_ds`、`lever_lm.insertion_selector_data.format_ceval_cmmlu_sample`、`lever_lm.utils.encode_text_qwen`（需 Qwen 模型，如 Qwen3-Embedding-4B）。

---

## 四、Step 2：训练 InsertionSelector

**运行：**

```bash
cd /home/lzh/llada-icl

# 使用默认 config（configs/insertion_selector_train_mmlu.yaml）
python train_insertion.py

# 指定 config
python train_insertion.py --config configs/insertion_selector_train_mmlu.yaml
```

**Config 要点（`configs/insertion_selector_train_mmlu.yaml`）：**

- **data.json_path**：Step 0 得到的 JSON（或已有 Stage1 JSON）。
- **data.train_path**：题库 parquet，须与 Step 1 预计算时用的 parquet 一致。
- **embedding.embedding_cache_path**：Step 1 得到的 `.pt`。
- **embedding.qwen_model_path**：仅 config 占位，训练时不再用 Qwen。
- **model**：emb_dim / n_head / dropout，须与 embedding 维度一致（如 2560）。
- **training**：batch_size、max_epochs、lr、weight_decay、gpu_ids 等。
- **output.checkpoint_dir**：存 checkpoint 的目录（如 `generated_icd_data/checkpoints/insertion_selector_mmlu`）。

训练会按 `val_loss` 存 `save_top_k` 个 checkpoint（如 `min_vl-epoch=8-step=xxx.ckpt`），之后在 `icl_inference` 里用 `insertion_selector.ckpt_path` 或 `insertion_selector.ckpt_dir` + `default_cpk_key` 指向即可。

---

## 五、完整流程小结（MMLU 示例）

1. **准备 JSON**  
   - 已有：把路径写到 `configs/insertion_selector_train_mmlu.yaml` 的 `data.json_path`。  
   - 没有：先跑 generate_data，再填 `data.json_path`。

2. **准备 embedding 缓存（二选一）**  
   - **已有**：确保是 `[N, emb_dim]` 的 tensor、行号与题库 parquet 一致，在 config 里设 `embedding.embedding_cache_path` 指向该 `.pt`，**无需**跑 precompute。  
   - **没有**：运行 `python precompute_insertion_embedding.py`，使 config 中的 `embedding.embedding_cache_path` 指向生成的 `.pt`。

3. **训练**  
   ```bash
   python train_insertion.py
   ```  
   检查 `output.checkpoint_dir` 下是否有 `min_vl-*.ckpt`。

4. **推理**  
   在 `configs/icl_inference.yaml` 或命令行中设置  
   `insertion_selector.enabled=true`、`insertion_selector.ckpt_path`（或 `ckpt_dir` + `default_cpk_key`），然后跑 `icl_inference.py`（见 [icl_inference_runbook.md](icl_inference_runbook.md)）。

---

## 六、涉及的文件

| 文件 | 作用 |
|------|------|
| `precompute_insertion_embedding.py` | Step 1：题库 parquet → 文本 → Qwen 编码 → 存 `.pt` |
| `train_insertion.py` | Step 2：读 JSON + `.pt`，训练 InsertionSelector，写 checkpoint |
| `configs/insertion_selector_train_mmlu.yaml` | Step 1 与 Step 2 的默认 config（json_path、train_path、embedding_cache_path、model、training、output） |
| `lever_lm/insertion_selector_data.py` | 解析 JSON（parse_json_to_samples）、划分 train/val（train_val_split）、Dataset（InsertionSelectorDataset） |
| `lever_lm/models/InsertionSelector.py` | 第二个分类器模型定义 |

训练第二个分类器只需要：**先跑 `precompute_insertion_embedding.py`，再跑 `train_insertion.py`**；若没有 Stage1 JSON，需先完成 generate_data 再执行上述两步。
