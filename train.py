import json
import os
import pickle
from typing import Dict, List, Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import torch

from lever_lm.utils import data_split
from utils import load_ds, fix_icd_json_query_last


def _load_embedding_dict(path: str) -> Dict[int, Union[torch.Tensor, List[float]]]:
    """
    Load precomputed embedding: {query_id(int): embedding(Tensor or list[float])}.

    Supported formats:
    - Dict .pt/.pkl: {query_id: vector}（含 utils.sampler_cache_to_embedding_dict 产出的文件）
    - Sampler cache .pt: 单个 tensor shape (N, D)，会按行转成 {i: tensor[i]}，无需事先跑 sampler_cache_to_embedding_dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"embedding_path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Unsupported embedding file extension: {ext} (path={path})")

    # Sampler 的 cache 是 (N, D) tensor：按行转成 dict
    if isinstance(obj, torch.Tensor):
        if obj.dim() != 2:
            raise TypeError(f"Embedding tensor must be 2D (N, D), got shape {obj.shape}")
        return {i: obj[i] for i in range(obj.shape[0])}

    if not isinstance(obj, dict):
        raise TypeError(f"Embedding file must be dict or 2D tensor, got: {type(obj)}")

    out: Dict[int, Union[torch.Tensor, List[float]]] = {}
    for k, v in obj.items():
        out[int(k)] = v
    return out


def make_collate_fn(index_ds_size: int):
    """
    Build full GPT2 token sequence from dataset output.

    Dataset returns:
      - query_embedding: [D]
      - icd_seq_idx: [K]  (ICD ids only; query id already removed)

    We construct:
      input_ids = [BOS, QUERY_PLACEHOLDER] + icd_ids + [EOS]
      attention_mask: 1 for real tokens, 0 for padding
      labels: same as input_ids but pad positions set to -100

    若 few_shot 固定，每条序列长度相同，max_len=L，padding 不会生效；保留逻辑以兼容变长数据。
    """

    bos_id = index_ds_size
    eos_id = index_ds_size + 1
    query_placeholder_id = index_ds_size + 2

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        query_embeds = torch.stack([b["query_embedding"] for b in batch], dim=0)

        seqs: List[torch.Tensor] = []
        for b in batch:
            icd_ids = b["icd_seq_idx"].to(torch.long)
            full = torch.cat(
                [
                    torch.tensor([bos_id, query_placeholder_id], dtype=torch.long),
                    icd_ids,
                    torch.tensor([eos_id], dtype=torch.long),
                ],
                dim=0,
            )
            seqs.append(full)

        max_len = max(s.numel() for s in seqs)
        input_ids = torch.full((len(seqs), max_len), eos_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)

        for i, s in enumerate(seqs):
            L = s.numel()
            input_ids[i, :L] = s
            attention_mask[i, :L] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "query_embedding": query_embeds,
            "icd_seq_idx": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


class LeverLM(pl.LightningModule):
    def __init__(self, lever_lm, lr: float, weight_decay: float = 1e-2, warm_steps: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["lever_lm"])
        self.lever_lm = lever_lm

    def training_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log("train_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log("val_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.lever_lm.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.hparams.warm_steps, float):
            warm_steps = int(self.hparams.warm_steps * step_batches)
        elif isinstance(self.hparams.warm_steps, int):
            warm_steps = self.hparams.warm_steps
        else:
            raise ValueError(f"warm_steps must be int or float, got {type(self.hparams.warm_steps)}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ICDSeqDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # 若配置了 fix_query_last_json，先对 stage1 的「query 在中间」的 JSON 做置尾，再加载
        if cfg.get("fix_query_last_json"):
            fixed_path = fix_icd_json_query_last(cfg.fix_query_last_json)
            data_files_path = fixed_path
        else:
            data_files_path = os.path.join(cfg.result_dir, "generated_data", cfg.data_files)

        with open(data_files_path, "r") as f:
            generated_data = json.load(f)

        self.train_data, self.val_data = data_split(generated_data, cfg.train_ratio)
        self.embedding_dict = _load_embedding_dict(cfg.embedding_path)

        # dataset length for special token ids and model vocab
        # We still load the HF dataset only to get its size (len), no CLIP / raw fields needed.
        self.index_ds_size = len(load_ds(cfg, "train"))

        self.ds_factory = hydra.utils.instantiate(cfg.train.lever_lm_ds, _partial_=True)
        self.collate_fn = make_collate_fn(self.index_ds_size)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.trainset = self.ds_factory(data=self.train_data, embedding_dict=self.embedding_dict)
            self.valset = self.ds_factory(data=self.val_data, embedding_dict=self.embedding_dict)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    try:
        from pytorch_lightning.loggers.wandb import WandbLogger
        logger = WandbLogger(**cfg.wandb_args)
    except (ImportError, ModuleNotFoundError) as e:
        save_dir = cfg.wandb_args.get("save_dir", cfg.result_dir + "/wandb_logs")
        logger = TensorBoardLogger(save_dir=save_dir, name=cfg.wandb_args.get("name", "debug"))
        print(f"[train] wandb 不可用 ({e})，使用 TensorBoard，日志保存在 {save_dir}")
    tl_model_cpk_callback = ModelCheckpoint(
        filename="min_tl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="train_loss",
        save_last=False,
        save_top_k=1,
        mode="min",
        dirpath=cfg.dirpath,
    )
    vl_model_cpk_callback = ModelCheckpoint(
        filename="min_vl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=5,  # 保留 val_loss 最优的 5 个，便于使用 epoch 0/1 等早期权重
        mode="min",
        dirpath=cfg.dirpath,
    )
    early_stop_cfg = cfg.get("early_stopping")
    callbacks = [
        LearningRateMonitor(),
        tl_model_cpk_callback,
        vl_model_cpk_callback,
    ]
    if early_stop_cfg:
        early_stop = EarlyStopping(**OmegaConf.to_container(early_stop_cfg, resolve=True))
        callbacks.append(early_stop)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer_args,
    )

    data_module = ICDSeqDataModule(cfg)
    # 构建 embedding_lookup [vocab_size, input_dim]，供 ICD 位置叠加 text embedding
    input_dim = cfg.train.lever_lm.get("input_dim", 2560)
    emb_lookup = torch.zeros(data_module.index_ds_size + 3, input_dim, dtype=torch.float32)
    missing_count = 0
    for i in range(data_module.index_ds_size):
        if i in data_module.embedding_dict:
            v = data_module.embedding_dict[i]
            v = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
            emb_lookup[i] = v.squeeze().float()
        else:
            missing_count += 1
    # 诊断：若候选 ICD 样本缺少 embedding，会导致 Zero-Embedding，模型无法学到语义，验证集崩溃
    if missing_count > 0:
        raise RuntimeError(
            f"[致命] embedding_dict 缺少 {missing_count}/{data_module.index_ds_size} 个索引的向量！\n"
            f"emb_lookup 中索引 0..{data_module.index_ds_size - 1} 必须全部有值（对应所有候选 ICD 样本的 Qwen 语义特征）。\n"
            f"当前 embedding_path 可能只包含 query 向量。请确保使用 sampler 的完整 TextFeatures.pt：\n"
            f"  例如 embedding_path=generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt\n"
            f"  或先运行 utils.sampler_cache_to_embedding_dict 将 (N,D) tensor 转为 dict 后使用。"
        )
    print(f"[emb_lookup] 已加载 {data_module.index_ds_size} 条 Qwen embedding，覆盖所有候选 ICD 索引 0..{data_module.index_ds_size - 1}")
    # 实例化模型，传入 index_ds_size 和 embedding_lookup
    lever_lm = hydra.utils.instantiate(
        cfg.train.lever_lm,
        index_ds_size=data_module.index_ds_size,
        embedding_lookup=emb_lookup,
    )
    model = LeverLM(lever_lm, cfg.lr, cfg.weight_decay, cfg.warm_steps)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()