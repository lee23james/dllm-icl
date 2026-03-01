import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

# 从 lever_lm 导入模型与数据
from lever_lm.models import InsertionSelector
from lever_lm.insertion_selector_data import (
    parse_json_to_samples,
    train_val_split,
    InsertionSelectorDataset,
)


# ==========================================
# 1. 闪电模型模块 (The Engine Wrapper)
# ==========================================
class LeverLM_Trainer(pl.LightningModule):
    def __init__(self, emb_dim=2560, n_head=16, lr=1e-4, warm_steps=0.1, weight_decay=1e-2, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = InsertionSelector(emb_dim=emb_dim, n_head=n_head, dropout=dropout)

    def training_step(self, batch, batch_idx):
        output = self.model(
            query_embed=batch["query_embed"],
            icd_embeds=batch["icd_embeds"],
            labels=batch["labels"],
        )
        loss = output["loss"]
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        train_acc = (output["logits"].argmax(dim=-1) == batch["labels"]).float().mean()
        self.log("train_acc", train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            query_embed=batch["query_embed"],
            icd_embeds=batch["icd_embeds"],
            labels=batch["labels"],
        )
        val_loss = output["loss"]
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        acc = (output["logits"].argmax(dim=-1) == batch["labels"]).float().mean()
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        # 各位置平均概率，便于检查是否总预测 0（看 val_prob_0 是否接近 1）
        probs = torch.softmax(output["logits"], dim=-1)
        num_slots = probs.size(-1)
        for k in range(num_slots):
            self.log(f"val_prob_{k}", probs[:, k].mean(), sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        weight_decay = float(self.hparams.weight_decay)
        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


# ==========================================
# 2. 数据管道模块 (The Fuel Pipe)
# ==========================================
class OfflineEmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError(
                "train_dataset 长度为 0，无法创建 DataLoader。"
                "请检查数据与 config（json_path、val_ratio 等）或减少 GPU 数量使样本数 >= GPU 数。"
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_insertion_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_insertion_batch,
        )


def collate_insertion_batch(batch_list):
    """Collate list of dicts with query_embed, icd_embeds, labels into batched tensors."""
    import torch
    query_embed = torch.stack([b["query_embed"] for b in batch_list])
    icd_embeds = torch.stack([b["icd_embeds"] for b in batch_list])
    labels = torch.stack([b["labels"] for b in batch_list])
    return {"query_embed": query_embed, "icd_embeds": icd_embeds, "labels": labels}


# ==========================================
# 3. 启动中枢 (The Starter)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "insertion_selector_train_mmlu.yaml"),
        help="Path to insertion_selector_train_mmlu.yaml",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    emb_cfg = cfg.get("embedding", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    out_cfg = cfg.get("output", {})

    json_path = project_root / data_cfg["json_path"]
    embedding_cache_path = project_root / emb_cfg["embedding_cache_path"]
    val_ratio = data_cfg.get("val_ratio", 0.2)
    val_split_seed = data_cfg.get("val_split_seed", 42)

    samples = parse_json_to_samples(json_path)
    train_samples, val_samples = train_val_split(samples, val_ratio=val_ratio, seed=val_split_seed, by_anchor=True)

    # 多卡 DDP 时每张卡至少要有 1 条样本，否则会报 num_samples=0
    gpu_ids = train_cfg.get("gpu_ids", None)
    num_devices = len(gpu_ids) if (gpu_ids and isinstance(gpu_ids, list)) else 1
    if len(train_samples) == 0:
        raise ValueError(
            "训练集为空 (train_samples=0)。请检查：1) json_path 是否存在且格式正确；"
            f"2) JSON 内是否有有效样本。当前 json_path={json_path}"
        )
    if num_devices > 1 and len(train_samples) < num_devices:
        raise ValueError(
            f"训练样本数 ({len(train_samples)}) 少于 GPU 数 ({num_devices})，DDP 下会导致部分 rank 无数据。"
            "请减少 config 中 training.gpu_ids 为单卡（如 [1]）或增加数据量。"
        )

    train_dataset = InsertionSelectorDataset(train_samples, embedding_cache_path)
    val_dataset = InsertionSelectorDataset(val_samples, embedding_cache_path)

    batch_size = train_cfg.get("batch_size", 32)
    data_module = OfflineEmbeddingDataModule(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
    )

    emb_dim = model_cfg.get("emb_dim", 2560)
    n_head = model_cfg.get("n_head", 16)
    dropout = float(model_cfg.get("dropout", 0.1))
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-2)
    model = LeverLM_Trainer(
        emb_dim=emb_dim,
        n_head=n_head,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_dir = project_root / out_cfg.get("checkpoint_dir", "checkpoints/insertion_selector")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_top_k = out_cfg.get("save_top_k", 2)
    max_epochs = train_cfg.get("max_epochs", 10)

    # 多 GPU：config 中 training.gpu_ids 为列表如 [1,2,3] 则用这些卡并行
    if gpu_ids is not None and isinstance(gpu_ids, list) and len(gpu_ids) > 0:
        devices = gpu_ids
        strategy = "ddp" if len(devices) > 1 else "auto"
    else:
        devices = "auto"
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        logger=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor="val_loss",
                mode="min",
                save_top_k=save_top_k,
            ),
            # pl.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     mode="min",
            #     patience=5,
            #     verbose=True,
            # ),
        ],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
