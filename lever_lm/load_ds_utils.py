import os 
import datasets
from datasets import DatasetDict, load_dataset
#这整段代码都进行数据导入
#这里我需要加载hf数据集,同时还需要建立本地加载gsm8k数据集
#目前我打算弄一下gsm8k,看\看能不能进行提点

def expand_choices_features(example):
    """
    展开choices数组字段为独立的字段
    
    将choices数组（如[choice0, choice1, choice2, choice3]）展开为：
    - choice_a: choices[0]
    - choice_b: choices[1] 
    - choice_c: choices[2]
    - choice_d: choices[3]
    
    Args:
        example: 数据样本字典，包含choices字段
        
    Returns:
        展开后的数据样本字典
    """
    if "choices" in example and isinstance(example["choices"], list):
        choices = example["choices"]
        expanded = example.copy()
        # 展开choices数组为独立字段
        if len(choices) >= 1:
            expanded["choice_a"] = choices[0]
        if len(choices) >= 2:
            expanded["choice_b"] = choices[1]
        if len(choices) >= 3:
            expanded["choice_c"] = choices[2]
        if len(choices) >= 4:
            expanded["choice_d"] = choices[3]
        # 保留原始choices字段以保持兼容性
        return expanded
    return example

def load_hf_ds(
    name,
    split=None,
    expand_choices=False
):
    """
    加载HuggingFace数据集
    
    Args:
        name: 数据集名称
        split: 数据集分割
        expand_choices: 是否展开choices数组字段（用于MMLU等任务）
    
    Returns:
        加载的数据集
    """
    ds=load_dataset(name,split=split)
    if isinstance(ds,DatasetDict):
        for split_name in ds.keys():
            ds[split_name] = ds[split_name].add_column("idx", list(range(len(ds[split_name]))))
            ds[split_name] = ds[split_name].add_column("isquery", [0] * len(ds[split_name]))
            if expand_choices:
                # 展开choices字段
                ds[split_name] = ds[split_name].map(expand_choices_features)
    else:
        ds=ds.add_column("idx",list(range(len(ds))))
        ds=ds.add_column("isquery", [0] * len(ds))
        if expand_choices:
            # 展开choices字段
            ds = ds.map(expand_choices_features)
    return ds

#加载本地的gsm8k数据集,
#这里主要利用的是main的文件夹
#在这里就可以处理成一个数据方便之后进行处理
#我这里最好加一个是否是query的问题,方便之后可以进行筛选
#从hub上下载的还没有适配
def load_gsm8k_ds(
    version,
    train_path=None,
    val_path=None,
    data_path=None,
    split=None,
):
    """
    加载GSM8K数据集
    
    支持两种调用方式：
    1. 同时加载train和validation：load_gsm8k_ds(version, train_path, val_path) -> DatasetDict
    2. 只加载单个split：load_gsm8k_ds(version, data_path=..., split="train") -> Dataset
    
    Args:
        version: 数据集版本（目前只支持"local"）
        train_path: 训练集路径（方式1）
        val_path: 验证集路径（方式1）
        data_path: 数据文件路径（方式2）
        split: 要加载的split名称，如"train"或"validation"（方式2）
    
    Returns:
        DatasetDict（方式1）或 Dataset（方式2）
    """
#注意一定要加载main的数据集文件夹
#最后我需要的是提取出来对某个问题的信心程度
#根据原来论文的结果,我应该对tarin去找结果,但是可以再val验证效果,所以还是要把val也加进来
    if version == "local":
        # 方式2：只加载单个split
        if data_path is not None and split is not None:
            data_files = {split: data_path}
            ds = load_dataset("parquet", data_files=data_files)
            # 只处理指定的split
            ds[split] = ds[split].add_column("idx", list(range(len(ds[split]))))
            ds[split] = ds[split].add_column("isquery", [0] * len(ds[split]))
            
            def process_gsm8k_data(batch):
                questions = [i for i in batch["question"]]
                answers = [q for q in batch["answer"]]
                #拼接成ICL的形式(之后再进行answer的消除)
                #这里已经拼接成独立的字符串,然后每个QA初始化为isquery=0,判断是test之后再变成1
                q_a = [f"question:{q}\n<answer>\n{a}\n</answer>" for q, a in zip(questions, answers)]
                batch["q_a"] = q_a
                return batch
            
            #应用数据处理函数
            ds[split] = ds[split].map(process_gsm8k_data, batched=True, num_proc=12)
            return ds[split]  # 返回单个Dataset
        
        # 方式1：同时加载train和validation
        elif train_path is not None and val_path is not None:
            data_files = {
                "train": train_path,
                "validation": val_path
        }
            ds = load_dataset("parquet", data_files=data_files)

        #添加全局唯一的idx适配杠杠模型,为所有数据集都添上idx的标识
        #这里加上任务的唯一标识
            for split_name in ds.keys():
                ds[split_name] = ds[split_name].add_column("idx", list(range(len(ds[split_name]))))
            #添加isquery字段，初始值为0（每个ICL样本都是0）
                ds[split_name] = ds[split_name].add_column("isquery", [0] * len(ds[split_name]))

            def process_gsm8k_data(batch):
                questions = [i for i in batch["question"]]
                answers = [q for q in batch["answer"]]
                #拼接成ICL的形式(之后再进行answer的消除)
                #这里已经拼接成独立的字符串,然后每个QA初始化为isquery=0,判断是test之后再变成1
                q_a = [f"question:{q}\n<answer>\n{a}\n</answer>" for q, a in zip(questions, answers)]
                batch["q_a"] = q_a
                return batch

            #应用数据处理函数
            ds = ds.map(process_gsm8k_data, batched=True, num_proc=12)
            return ds  # 返回DatasetDict
        else:
            raise ValueError(
                "load_gsm8k_ds requires either (train_path, val_path) or (data_path, split) parameters"
            )
    else:
        raise ValueError(f"Invalid version: {version}")
#其他数据集之后再去想,这里先进行gsm8k的数据处理

#我这里要进行处理的是mmlu这个数据集,来准备进行mmlu数据格式的导入


def load_mmlu_ds(
    version: str,
    train_path: str | None = None,
    val_path: str | None = None,
    data_path: str | None = None,
    split: str | None = None,
):
    """
    加载 MMLU 数据集（本地版），在原始数据基础上**仅添加 idx 和 isquery 字段**。
    
    注意：
    - 不进行 prompt 拼接（不生成 q_a）
    - 不修改原有字段（如 question / answer / choices / choice_a~d）
    
    支持两种调用方式（与 load_gsm8k_ds 保持一致）：
    1. 同时加载 train 和 validation：
         load_mmlu_ds(version, train_path=..., val_path=...) -> DatasetDict
    2. 只加载单个 split：
         load_mmlu_ds(version, data_path=..., split="train") -> Dataset
    """
    if version != "local":
        raise ValueError(f"Invalid version for MMLU: {version}")

    def _cast_answer_to_string(ds_split: datasets.Dataset) -> datasets.Dataset:
        """Force `answer` column to `string`.

        Some parquet/arrow schemas keep `answer` typed as int in validation/test.
        Even if we map values to "A"-"D", Datasets may coerce them back to int.
        This helper makes the conversion stick by explicitly casting features.
        """
        if "answer" not in ds_split.column_names:
            return ds_split

        # Re-apply 0-3 -> A-D mapping defensively (some splits keep `answer` as int
        # despite the earlier map due to Arrow schema coercion).
        idx2label = {0: "A", 1: "B", 2: "C", 3: "D"}

        def _ensure_label(ex):
            a = ex.get("answer")
            if a is None:
                return {"answer": ""}
            if isinstance(a, str) and a in {"A", "B", "C", "D"}:
                return {"answer": a}
            try:
                ai = int(a)
                return {"answer": idx2label.get(ai, str(a))}
            except Exception:
                return {"answer": str(a)}

        ds_split = ds_split.map(_ensure_label)

        # Make sure all values are representable as strings before casting.
        # Preserve already-mapped labels ("A"-"D") if present.
        # (after _ensure_label, everything should already be a string label)

        features = ds_split.features.copy()
        features["answer"] = datasets.Value("string")
        return ds_split.cast(features)

    def _process_mmlu_example(example):
        """
        将 MMLU 样本中的嵌套字段展开，并进行标准化：
        1. 如果存在 'train' 子字典，则将其中的键值提升到顶层，并删除 'train'
        2. 将 choices 数组展开为 choice_a~choice_d
        3. 将 answer 从数字索引(0~3)映射为选项字母(A~D)
        """
        # avoid in-place mutation issues with Arrow/datasets typing by building a new dict
        out = dict(example)

        # 1) 展开嵌套的 'train' 字段（HF 读取 parquet 时常见结构）
        if "train" in out and isinstance(out["train"], dict):
            inner = out["train"]
            for k, v in inner.items():
                out.setdefault(k, v)
            out.pop("train", None)

        # 2) 处理 choices -> choice_a~d
        if "choices" in out and isinstance(out["choices"], list):
            choices = out["choices"]
            if len(choices) >= 1:
                out["choice_a"] = choices[0]
            if len(choices) >= 2:
                out["choice_b"] = choices[1]
            if len(choices) >= 3:
                out["choice_c"] = choices[2]
            if len(choices) >= 4:
                out["choice_d"] = choices[3]
            out.pop("choices", None)

        # 3) 处理 answer: 0/1/2/3 -> A/B/C/D
        if "answer" in out:
            idx2label = {0: "A", 1: "B", 2: "C", 3: "D"}
            ans = out["answer"]
            try:
                ans_idx = int(ans)
                out["answer"] = idx2label.get(ans_idx, str(ans))
            except Exception:
                out["answer"] = str(ans)

        return out

    # 方式2：只加载单个 split
    if data_path is not None and split is not None:
        data_files = {split: data_path}
        ds = load_dataset("parquet", data_files=data_files)

        # 只处理指定的 split，添加 idx 和 isquery 字段
        ds_split = ds[split]
        ds_split = ds_split.add_column("idx", list(range(len(ds_split))))
        ds_split = ds_split.add_column("isquery", [0] * len(ds_split))

        # 展开 choices 并映射 answer
        ds_split = ds_split.map(_process_mmlu_example)
        ds_split = _cast_answer_to_string(ds_split)
        return ds_split  # 返回单个 Dataset

    # 方式1：同时加载 train 和 validation
    if train_path is not None and val_path is not None:
        data_files = {
            "train": train_path,
            "validation": val_path,
        }
        ds = load_dataset("parquet", data_files=data_files)

        # 为每个 split 添加全局唯一的 idx 和 isquery 字段，并处理 choices/answer
        for split_name in ds.keys():
            ds_split = ds[split_name]
            ds_split = ds_split.add_column("idx", list(range(len(ds_split))))
            ds_split = ds_split.add_column("isquery", [0] * len(ds_split))
            ds_split = ds_split.map(_process_mmlu_example)
            ds_split = _cast_answer_to_string(ds_split)
            ds[split_name] = ds_split

        return ds  # 返回 DatasetDict

    raise ValueError(
        "load_mmlu_ds requires either (train_path, val_path) or (data_path, split) parameters"
    )


def load_subj_ds(
    data_path: str,
    split: str = "train",
    label_mapping: dict | None = None,
):
    """
    加载 Subj 数据集（JSONL 格式）。

    每行格式：{"sentence": "...", "label": "0"|"1"}。
    通过 label_mapping 将 label 数字转为含义文本（如 subjective/objective），
    供 prompt 与蒙特卡洛 InfoScore 使用。

    Args:
        data_path: JSONL 文件路径（如 data/subj/train.jsonl）
        split: split 名称（仅用于 load_dataset 的 key，实际单文件）
        label_mapping: {"0": "subjective", "1": "objective"}，必须提供

    Returns:
        Dataset，含 idx, isquery, sentence, label, answer
    """
    if label_mapping is None:
        label_mapping = {"0": "subjective", "1": "objective"}
    label_mapping = {str(k): str(v) for k, v in label_mapping.items()}

    data_files = {split: data_path}
    ds = load_dataset("json", data_files=data_files)
    ds_split = ds[split]

    ds_split = ds_split.add_column("idx", list(range(len(ds_split))))
    ds_split = ds_split.add_column("isquery", [0] * len(ds_split))

    def _add_answer(example):
        lab = str(example.get("label", ""))
        return {"answer": label_mapping.get(lab, lab)}

    ds_split = ds_split.map(_add_answer, remove_columns=[])
    return ds_split