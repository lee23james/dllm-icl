# ICL 推理阶段调用流程详解

本文档详细描述 `retriever.type=lever_lm` 时，从 `icl_inference.py` 启动到输出结果的完整调用链。

---

## 一、整体流程概览

```
icl_inference.py main(cfg)
    │
    ├── ① 加载配置与数据
    ├── ② 加载 LLaDA 模型
    ├── ③ 初始化 LeverLM + LeverLMRetriever
    ├── ④ 初始化 LLaDAInterface
    ├── ⑤ 初始化 Metrics
    └── ⑥ 推理循环（每个 test_sample）
            │
            ├── 6.1 Retriever 生成 ICD 索引
            │       └── LeverLMRetriever.retrieve()
            │               └── GPT2LeverLM.generation()
            │
            ├── 6.2 构建 Prompt
            │       └── LLaDAInterface.build_prompt()
            │
            ├── 6.3 Tokenize + LLaDA 生成
            │       └── LLaDAInterface.tokenize_prompt() + generate()
            │               └── src.generate.generate()
            │
            └── 6.4 解码、提取答案、评估
                    └── decode_output() + extract_answer() + metrics.evaluate_single()
```

---

## 二、初始化阶段

### 2.1 配置加载

| 文件 | 调用 |
|------|------|
| `configs/icl_inference.yaml` | `@hydra.main(config_name="icl_inference")` |
| `configs/task/mmlu.yaml` | defaults 合并 |
| `configs/dataset/mmlu.yaml` | defaults 合并 |
| `configs/infer_model/llada.yaml` | defaults 合并 |
| `configs/train/base_experiment.yaml` | defaults 合并（LeverLM 结构） |

### 2.2 数据加载

```
icl_inference.py 第 44-62 行
    │
    └── utils.load_ds(cfg, split="train" / "validation")
            └── lever_lm.load_ds_utils.load_mmlu_ds(...)
                    └── 返回 train_ds, test_ds
```

### 2.3 LLaDA 模型加载

```
icl_inference.py 第 66-112 行
    │
    ├── AutoModel.from_pretrained(model_path)  → model
    └── AutoTokenizer.from_pretrained(model_path)  → tokenizer
```

### 2.4 LeverLM 初始化（retriever.type=lever_lm）

```
icl_inference.py 第 116-152 行
    │
    ├── utils.get_lever_lm_path(ckpt_dir, default_cpk_key)
    │       文件: utils.py 第 508-534 行
    │       作用: 从 ckpt_dir 解析 checkpoint 路径（min_vl / min_tl / last）
    │
    ├── utils.init_lever_lm(cfg, ckpt_path, train_ds, embedding_path, device)
    │       文件: utils.py 第 537-586 行
    │       作用:
    │         1. _load_embedding_dict(embedding_path) → embedding_dict
    │         2. 构建 emb_lookup [index_ds_size+3, input_dim]
    │         3. hydra.utils.instantiate(cfg.train.lever_lm, ...) → GPT2LeverLM
    │         4. LeverLM(lever_lm, ...).load_state_dict(ckpt)
    │         5. 返回 model.lever_lm（即 GPT2LeverLM 实例）
    │
    └── LeverLMRetriever(index_ds, lever_lm, qwen_model_path, query_text_extractor, nshot, ...)
            文件: open_mmicl/retriever/lever_lm_retriever.py 第 24-57 行
```

### 2.5 LLaDAInterface 初始化

```
icl_inference.py 第 186-196 行
    │
    └── LLaDAInterface(model, tokenizer, task, mask_id, mask_length, ...)
            文件: open_mmicl/interface/llada_interface.py 第 19-66 行
            内部: 初始化 PromptTemplate (pt)
```

### 2.6 Metrics 初始化

```
icl_inference.py 第 209-215 行
    │
    └── MMLUMetrics() 或 GSM8KMetrics()
            文件: open_mmicl/metrics/mmlu_metrics.py
```

---

## 三、推理循环（单样本详细调用链）

对每个 `test_sample in test_ds`：

### 3.1 检索 ICD 索引

```
icl_inference.py 第 221-225 行
    │
    retriever.retrieve(test_sample, exclude_indices=[...])
    │
    ▼
open_mmicl/retriever/lever_lm_retriever.py
    │
    ├── 第 99 行: text = self._extract_text(test_sample)
    │       │
    │       └── 第 61 行: return self.query_text_extractor(sample)
    │               │
    │               └── icl_inference 传入的 lambda:
    │                       format_text_for_embedding(s, dataset_type, text_field_name)
    │                       文件: lever_lm/utils.py (与 TextSimQwenMMRSampler 格式一致)
    │                       输出: "question\nA. choice_a\nB. choice_b\nC. choice_c\nD. choice_d"
    │
    ├── 第 100 行: query_emb = self._encode_query(text)
    │       │
    │       └── 第 65-73 行: encode_text_qwen(text_list=[text], ...)
    │               文件: lever_lm/utils.py 第 134-212 行
    │               作用: SentenceTransformer(Qwen3-Embedding-4B).encode(text_list)
    │               输出: query_embedding [1, 2560]
    │
    ├── 第 101-106 行: pred_seqs = self.lever_lm.generation(...)
    │       │
    │       └── 见下方 3.1.1 GPT2LeverLM.generation
    │
    ├── 第 107 行: pred_ids = pred_seqs[0][2 : 2 + self.nshot]
    │       提取 ICD 索引，跳过 [BOS, QUERY_placeholder]
    │
    └── 第 108 行: return self._postprocess_ids(pred_ids)
            第 75-80 行: 截断、过滤、可选 reverse_seq
```

#### 3.1.1 GPT2LeverLM.generation 详细流程

```
lever_lm/models/gpt2_lever_lm.py 第 133-182 行
    │
    ├── 第 158-162 行: 设置 token id
    │       bos_token_id = index_ds_size
    │       eos_token_id = index_ds_size + 1
    │       query_placeholder_id = index_ds_size + 2
    │
    ├── 第 165-169 行: 初始序列
    │       icd_seq_idx = [[bos_token_id, query_placeholder_id]]  # [BOS, QUERY]
    │
    └── 第 171-180 行: 自回归生成 shot_num 步
            for _ in range(shot_num):
                out = self.forward(query_embedding, icd_seq_idx)   # 第 79-131 行
                logits = out["logits"][:, -1, :]
                logits[:, index_ds_size:] = -inf   # mask 特殊 token
                for b, j: logits[b, icd_seq_idx[b,j]] = -inf   # mask 已用 id
                next_token = logits.argmax(dim=-1)
                icd_seq_idx = cat([icd_seq_idx, next_token], dim=1)
    │
    └── 第 182 行: return icd_seq_idx.tolist()
            格式: [[BOS, QUERY_ph, id_1, id_2, id_3, id_4]]
```

#### 3.1.2 GPT2LeverLM.forward（generation 内部调用）

```
lever_lm/models/gpt2_lever_lm.py 第 79-131 行
    │
    ├── 第 95 行: inputs_embeds = lm_model.transformer.wte(icd_seq_idx)
    │       从 token id 得到基础 embedding
    │
    ├── 第 99-107 行: query 位置注入
    │       query_feat = input_adapter(query_embedding)
    │       inputs_embeds[:, 1] += query_feat   # 位置 1 为 QUERY
    │
    ├── 第 111-121 行: ICD 位置注入（embedding_lookup）
    │       icd_embeds = embedding_lookup[icd_seq_idx]
    │       inputs_embeds[:, 2:] += icd_feats[:, 2:] * content_mask
    │
    └── 第 126-130 行: output = lm_model(inputs_embeds=inputs_embeds, ...)
            返回 logits
```

---

### 3.2 获取 ICD 样本并构建 Prompt

```
icl_inference.py 第 227-236 行
    │
    ├── 第 228 行: icd_samples = [train_ds[i] for i in icd_indices]
    │
    └── 第 232-236 行: prompt = interface.build_prompt(icd_samples, test_sample, query_position)
            │
            ▼
open_mmicl/interface/llada_interface.py 第 142-187 行
    │
    ├── 第 164-168 行: 遍历 ice_samples，调用 _format_icd(ice)
    │       │
    │       └── 第 68-93 行: _format_icd(ice)
    │               pt.generate_ice_item(ice)  # PromptTemplate
    │               输出: 单个 ICD 的 prompt 文本（含 question + answer）
    │
    ├── 第 171 行: query_text = self._format_query(test_sample)
    │       │
    │       └── 第 95-140 行: _format_query(test_sample)
    │               pt.generate_query_item(test_sample, use_mask=True)
    │               输出: query 文本（answer 部分用 mask_token 替换）
    │
    ├── 第 174-182 行: 按 query_position 插入 query
    │       insertion_idx = icd_count - normalized_position
    │       prompt_parts.insert(insertion_idx, query_text)
    │
    └── 第 185 行: return split_token.join(prompt_parts)
            输出: 完整 prompt 字符串（ICD1 \n\n ICD2 \n\n ... \n\n query）
```

---

### 3.3 Tokenize 与 LLaDA 生成

```
icl_inference.py 第 239-244 行
    │
    ├── 第 239 行: prompt_tensor = interface.tokenize_prompt(prompt)
    │       │
    │       └── open_mmicl/interface/base_interface.py 第 97-112 行
    │               encoded = tokenizer(prompt, return_tensors="pt")
    │               return encoded["input_ids"].to(device)
    │               输出: prompt_tensor [1, L]
    │
    └── 第 244 行: output_tensor = interface.generate(prompt_tensor, **gen_kwargs)
            │
            ▼
open_mmicl/interface/llada_interface.py 第 190-237 行
    │
    ├── 第 213-217 行: 定位 mask 位置
    │       mask_positions = (prompt == mask_id).nonzero()
    │       gen_start = first_mask_pos
    │
    └── 第 222-234 行: from src.generate import generate
            output = generate(
                model, prompt, gen_start,
                steps=..., gen_length=..., block_length=...,
                temperature=..., cfg_scale=..., remasking=...,
                mask_id=...,
            )
            文件: src/generate.py
            作用: LLaDA diffusion 迭代生成，填充 mask 区域
            输出: output_tensor [1, L']（完整序列）
```

---

### 3.4 解码、提取答案、评估

```
icl_inference.py 第 247-256 行
    │
    ├── 第 247 行: generated_text = interface.decode_output(output_tensor)
    │       │
    │       └── base_interface.py 第 127-139 行
    │               tokenizer.decode(output[0], skip_special_tokens=True)
    │
    ├── 第 250 行: predicted_answer = interface.extract_answer(generated_text)
    │       │
    │       └── base_interface.py 第 172-207 行
    │               正则匹配 "The answer is X" / "#### X" / <answer>...</answer>
    │
    ├── 第 253 行: ground_truth = test_sample.get("answer", "")
    │
    └── 第 256 行: eval_result = metrics.evaluate_single(generated_text, ground_truth)
            │
            └── open_mmicl/metrics/mmlu_metrics.py
                    extract_choice(generated_text) → 提取预测选项 A/B/C/D
                    is_correct = (pred_choice == ground_truth)
```

---

## 四、后处理与保存

```
icl_inference.py 第 277-298 行
    │
    ├── 第 282 行: batch_metrics = metrics.evaluate_batch(all_generated, all_ground_truths)
    │       计算 accuracy, correct_count, total_count
    │
    └── 第 291-295 行: json.dump({ "metrics": batch_metrics, "results": results }, output_path)
```

---

## 五、调用关系汇总表

| 阶段 | 入口 | 被调用模块/函数 | 文件 |
|------|------|-----------------|------|
| 配置 | Hydra | configs/icl_inference.yaml, task, dataset, infer_model, train | configs/ |
| 数据 | load_ds | load_mmlu_ds | lever_lm/load_ds_utils.py |
| 模型 | AutoModel | LLaDA 模型 | infer_model/llada.yaml |
| LeverLM | init_lever_lm | get_lever_lm_path, _load_embedding_dict, LeverLM | utils.py, train.py |
| Retriever | retrieve | _extract_text, _encode_query, generation, _postprocess_ids | lever_lm_retriever.py |
| Query 文本 | query_text_extractor | format_text_for_embedding | lever_lm/utils.py |
| Qwen 编码 | _encode_query | encode_text_qwen | lever_lm/utils.py |
| ICD 生成 | lever_lm.generation | GPT2LeverLM.forward, generation | lever_lm/models/gpt2_lever_lm.py |
| Prompt | build_prompt | _format_icd, _format_query, PromptTemplate | llada_interface.py |
| Tokenize | tokenize_prompt | tokenizer(prompt) | base_interface.py |
| LLaDA 生成 | interface.generate | src.generate.generate | src/generate.py |
| 解码 | decode_output | tokenizer.decode | base_interface.py |
| 答案提取 | extract_answer | 正则匹配 | base_interface.py |
| 评估 | evaluate_single | extract_choice, is_correct | mmlu_metrics.py |

---

## 六、数据流示意

```
test_sample (dict)
    │
    ├─[Retriever]─► format_text_for_embedding ─► "Q\nA. a\nB. b\nC. c\nD. d"
    │                      │
    │                      ▼
    │               encode_text_qwen ─► query_embedding [1, 2560]
    │                      │
    │                      ▼
    │               GPT2LeverLM.generation ─► [BOS, QUERY, id1, id2, id3, id4]
    │                      │
    │                      ▼
    │               pred_ids = [id1, id2, id3, id4]
    │
    ├─[Interface]─► icd_samples = [train_ds[i] for i in pred_ids]
    │                      │
    │                      ▼
    │               build_prompt(icd_samples, test_sample) ─► prompt 字符串
    │                      │
    │                      ▼
    │               tokenize_prompt ─► prompt_tensor [1, L]
    │                      │
    │                      ▼
    │               src.generate.generate ─► output_tensor [1, L']
    │                      │
    │                      ▼
    │               decode_output ─► generated_text
    │                      │
    └─[Metrics]──── extract_answer ─► predicted_answer
                           │
                           ▼
                    evaluate_single ─► is_correct
```

---

## 七、关键 Token 约定（LeverLM）

| Token | ID | 含义 |
|-------|-----|------|
| 0 ~ index_ds_size-1 | 内容 | ICD 索引（对应 train_ds 行号） |
| index_ds_size | BOS | 序列开始 |
| index_ds_size + 1 | EOS | 序列结束 |
| index_ds_size + 2 | QUERY_placeholder | Query 占位符 |

生成时以 `[BOS, QUERY_placeholder]` 为起点，自回归追加 `shot_num` 个 ICD id。
