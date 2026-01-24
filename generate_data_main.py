"""
阶段1：数据生成主脚本
使用 sampler 生成 anchor 和 candidate_set，然后对每个 anchor 找到最优 ICD 序列
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List
from time import sleep

import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
from datasets import Dataset

# 添加项目路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from generate_data import generate_single_sample_icd
from lever_lm.load_ds_utils import load_gsm8k_ds
from lever_lm.candidate_sampler.random_sampler import RandSampler


def generate_icd_for_all_anchors(
    train_ds: Dataset,
    sampler_result: Dict,
    interface,  # 用于计算InfoScore的接口（用户自己实现）
    cfg: DictConfig,
) -> Dict:
    """
    为所有anchor样本生成最优ICD序列
    
    Args:
        train_ds: 训练数据集
        sampler_result: sampler返回的结果，包含anchor_set和candidate_set
        interface: 用于计算InfoScore的接口
        cfg: 配置对象
    
    Returns:
        包含所有anchor的ICD序列结果
    """
    anchor_set = sampler_result['anchor_set']
    candidate_set_dict = sampler_result['candidate_set']
    
    all_results = {}
    
    logger.info(f"Processing {len(anchor_set)} anchor samples...")
    
    for anchor_idx in tqdm(anchor_set, desc="Generating ICD for anchors", ncols=100):
        try:
            # 获取anchor数据
            anchor_data = train_ds[anchor_idx].copy()
            anchor_data["isquery"] = 1  # 标记为query,每一个都是query=1,方便之后进行配平
            
            # 获取该anchor对应的候选集索引
            candidate_indices = candidate_set_dict.get(anchor_idx, [])
            
            if len(candidate_indices) == 0:
                logger.warning(f"No candidates for anchor {anchor_idx}, skipping...")
                continue
            
            # 从训练集中提取候选数据
            candidate_data_list = [train_ds[idx] for idx in candidate_indices]
            
            # 转换为Dataset
            from datasets import Dataset as HFDataset
            candidate_set = HFDataset.from_list(candidate_data_list)
            
            # 为当前anchor生成最优ICD序列
            result = generate_single_sample_icd(
                interface=interface,
                test_data=anchor_data,
                cfg=cfg,
                candidate_set=candidate_set,
                metric=cfg.get("metric", "no_order"),
            )
            
            # 合并结果
            all_results.update(result)
            
        except Exception as e:
            logger.error(f"Error processing anchor {anchor_idx}: {e}")
            continue
    
    return all_results


def init_interface(cfg: DictConfig, device: str):
    """
    初始化interface（用于计算InfoScore）
    
    Args:
        cfg: 配置对象
        device: 设备字符串（如 "cuda:0"）
    
    Returns:
        interface对象
    """
    # TODO: 用户自己实现interface的初始化
    # 示例：
    # from open_mmicl.interface import LLaDAInterface
    # model = ...  # 加载模型
    # tokenizer = ...  # 加载tokenizer
    # interface = LLaDAInterface(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task=cfg.task.task_name,
    #     mask_id=cfg.get("mask_id", 126336),
    #     mask_length=cfg.get("mask_length", 256),
    # )
    # return interface
    raise NotImplementedError("Please implement init_interface function")


def gen_data(
    rank: int,
    cfg: DictConfig,
    sample_data: Dataset,
    train_ds: Dataset,
    candidate_set_idx: List[List[int]],
    save_path: str,
):
    """
    多卡数据分片处理函数
    
    Args:
        rank: 当前进程的rank
        cfg: 配置对象
        sample_data: anchor数据组成的Dataset
        train_ds: 训练数据集
        candidate_set_idx: 每个anchor对应的候选集索引列表（二维列表）
        save_path: 保存路径
    """
    world_size = len(cfg.gpu_ids)
    process_device = f"cuda:{cfg.gpu_ids[rank]}"

    subset_size = len(sample_data) // world_size
    subset_start = rank * subset_size
    subset_end = (
        subset_start + subset_size if rank != world_size - 1 else len(sample_data)
    )
    subset = sample_data.select(range(subset_start, subset_end))
    sub_cand_set_idx = candidate_set_idx[subset_start:subset_end]

    # load several models will cost large memory at the same time.
    # use sleep to load one by one.
    #这里的padding逻辑出现了严重问题,如果需要支持padding ,需要从头去修改llada的架构逻辑,不能只是单纯的从文件夹里面抄模型
    #===========================================
    sleep(cfg.sleep_time * rank)
    interface = init_interface(cfg, device=process_device)
    if cfg.scorer == "infoscore":
        interface.tokenizer.padding_side = "right"
    elif cfg.scorer == "cider":
        interface.tokenizer.padding_side = "left"
    #===========================================

    final_res = {}
    sub_res_basename = (
        os.path.basename(save_path).split(".")[0]
        + f"_rank:{rank}_({subset_start}, {subset_end}).json"
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)
    if os.path.exists(save_path):
        final_res.update(json.load(open(save_path)))
        logger.info(
            f"Rank: {rank} reloading data from {save_path}, begin from {len(final_res)}"
        )
    if len(final_res) == subset_size:
        logger.info(f"Rank: {rank} task is Done.")
        return

    subset = subset.select(range(len(final_res), len(subset)))
    for i, test_data in enumerate(
        tqdm(
            subset,
            disable=(rank != world_size - 1),
            total=subset_size,
            initial=len(final_res),
            ncols=100,
        ),
    ):
        # 获取当前anchor对应的候选集索引列表
        # subset已经是从len(final_res)开始的，所以需要加上len(final_res)来索引sub_cand_set_idx
        candidate_indices = sub_cand_set_idx[len(final_res) + i]
        # 从训练集中提取候选数据
        candidate_data_list = [train_ds[idx] for idx in candidate_indices]
        # 转换为Dataset
        from datasets import Dataset as HFDataset
        candidate_set = HFDataset.from_list(candidate_data_list)
        
        res = generate_single_sample_icd(
            interface=interface,
            test_data=test_data,
            cfg=cfg,
            candidate_set=candidate_set,
            metric=cfg.get("metric", "no_order"),
        )
        final_res.update(res)
        with open(save_path, "w") as f:
            json.dump(final_res, f)
    return


@hydra.main(version_base=None, config_path="configs", config_name="generate_data")
def main(cfg: DictConfig):
    """
    主函数：生成ICD序列数据
    """
    logger.info("="*80)
    logger.info("Stage 1: Data Generation")
    logger.info("="*80)
    
    # 1. 加载数据集
    #加载数据
    logger.info("Loading datasets...")
    if cfg.task.task_name == "gsm8k":
        train_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.train_path,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task.task_name}")
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    
    # 2. 使用sampler生成anchor和candidate_set
    logger.info("Sampling anchor set and candidate sets...")
    sampler = RandSampler(
        candidate_num=cfg.sampler.candidate_num,
        sampler_name=cfg.sampler.sampler_name,
        anchor_sample_num=cfg.sampler.anchor_sample_num,
        index_ds_len=len(train_ds),
        dataset_name=cfg.task.task_name,
        cache_dir=cfg.sampler.cache_dir,
        overwrite=cfg.sampler.overwrite,
    )
    
    sampler_result = sampler(train_ds)
    logger.info(f"Anchor set size: {len(sampler_result['anchor_set'])}")
    logger.info(f"Candidate sets generated for {len(sampler_result['candidate_set'])} anchors")
    
    # 3. 检查是否使用多卡模式
    use_multi_gpu = cfg.get("use_multi_gpu", False)
    gpu_ids = cfg.get("gpu_ids", [0])
    sleep_time = cfg.get("sleep_time", 10)
    
    if use_multi_gpu and len(gpu_ids) > 1:
        # 多卡模式：使用gen_data进行数据分片
        logger.info(f"Using multi-GPU mode with {len(gpu_ids)} GPUs: {gpu_ids}")
        
        # 准备sample_data（anchor数据组成的Dataset）
        anchor_set = sampler_result['anchor_set']
        candidate_set_dict = sampler_result['candidate_set']
        
        anchor_data_list = []
        candidate_set_idx = []
        
        for anchor_idx in anchor_set:
            # 获取anchor数据
            anchor_data = train_ds[anchor_idx].copy()
            anchor_data["isquery"] = 1  # 标记为query
            anchor_data_list.append(anchor_data)
            
            # 获取该anchor对应的候选集索引
            candidate_indices = candidate_set_dict.get(anchor_idx, [])
            candidate_set_idx.append(candidate_indices)
        
        # 转换为Dataset
        from datasets import Dataset as HFDataset
        sample_data = HFDataset.from_list(anchor_data_list)
        
        # 准备保存路径
        output_dir = Path(cfg.get("output_dir", "./generated_icd_data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{cfg.task.task_name}_icd_results.json")
        
        # 更新cfg以包含gpu_ids和sleep_time
        cfg.gpu_ids = gpu_ids
        cfg.sleep_time = sleep_time
        
        # 使用torch.multiprocessing启动多进程
        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(len(gpu_ids)):
            p = mp.Process(
                target=gen_data,
                args=(rank, cfg, sample_data, train_ds, candidate_set_idx, output_path)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        logger.info("All processes completed. Results saved to separate files per rank.")
        logger.info(f"Check files matching pattern: {cfg.task.task_name}_icd_results_rank:*.json")
        
        # 可选：合并所有rank的结果
        if cfg.get("merge_results", False):
            logger.info("Merging results from all ranks...")
            all_results = {}
            for rank in range(len(gpu_ids)):
                sub_res_basename = (
                    os.path.basename(output_path).split(".")[0]
                    + f"_rank:{rank}_*.json"
                )
                import glob
                pattern = output_path.replace(os.path.basename(output_path), sub_res_basename)
                files = glob.glob(pattern)
                for f in files:
                    with open(f, 'r') as file:
                        all_results.update(json.load(file))
            
            merged_path = output_dir / f"{cfg.task.task_name}_icd_results_merged.json"
            with open(merged_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Merged results saved to: {merged_path}")
            return all_results
        
        return None
    else:
        # 单卡模式：使用原有的generate_icd_for_all_anchors
        logger.info("Using single-GPU mode")
        
        # 初始化interface（用于计算InfoScore）
        # TODO: 用户自己实现interface的初始化
        interface = None
        if interface is None:
            logger.warning("Interface not initialized. InfoScore calculation will be skipped.")
            logger.warning("Please implement interface initialization in generate_data_main.py")
            # 临时：返回空结果
            logger.info("Skipping ICD generation (interface not implemented)")
            return
        
        # 为每个anchor生成最优ICD序列
        logger.info("Generating optimal ICD sequences for each anchor...")
        all_results = generate_icd_for_all_anchors(
            train_ds=train_ds,
            sampler_result=sampler_result,
            interface=interface,
            cfg=cfg,
        )
        
        # 保存结果
        output_dir = Path(cfg.get("output_dir", "./generated_icd_data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{cfg.task.task_name}_icd_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total anchor samples processed: {len(all_results)}")
        
        # 打印统计信息
        total_sequences = sum(len(r["id_list"]) for r in all_results.values())
        logger.info(f"Total ICD sequences generated: {total_sequences}")
        
        return all_results


if __name__ == "__main__":
    main()





