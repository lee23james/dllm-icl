# -*- coding: utf-8 -*-
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
from omegaconf import DictConfig, OmegaConf
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
from utils import load_ds

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
        cfg: 配置对象，需要包含 infer_model.model_path 等配置
        device: 设备字符串（如 "cuda:0"）
    
    Returns:
        interface对象
    """
    import torch
    from transformers import AutoTokenizer
    from open_mmicl.interface import LLaDAInterface
    # 使用本地实现的 LLaDAModelLM（带 post_init / all_tied_weights_keys 兼容逻辑）
    # 权重仍然从 infer_model.model_path 指定的目录加载（见 configs/infer_model/llada.yaml）
    from model.model_llada import LLaDAModelLM
    
    # 获取模型路径（优先从 infer_model.model_path，其次从 infer_model.model_name）
    model_path = cfg.infer_model.get("model_path", None)
    if model_path is None:
        model_path = cfg.infer_model.get("model_name", None)
    
    if model_path is None:
        raise ValueError(
            "Model path not found in config. Please set infer_model.model_path or infer_model.model_name"
        )
    
    logger.info(f"Loading LLaDA model from {model_path} on {device}...")
    logger.info("Using local LLaDAModelLM (model/model_llada.py) with weights from model_path")
    
    # 设置设备
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 准备模型参数
    model_kwargs = {
        "trust_remote_code": cfg.infer_model.get("trust_remote_code", True),
        "torch_dtype": getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
        "local_files_only": cfg.infer_model.get("local_files_only", True),
    }
    
    # 使用本地的 LLaDAModelLM.from_pretrained，而不是 AutoModel.from_pretrained。
    # 这样结构代码走本地实现，权重仍然来自模型目录，兼容 transformers==5.1.0。
    model = LLaDAModelLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.to(torch_device)
    model.eval()  # 设置为评估模式
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_kwargs["trust_remote_code"],
        local_files_only=model_kwargs["local_files_only"],
    )
    
    logger.info(f"Model loaded successfully on {device}")
    
    # 获取prompt模板（从task配置中读取）
    prompt_template = cfg.task.get("template", None)
    
    # 获取column_token_map（从task配置中读取）
    column_token_map = cfg.task.get("column_token_map", None)
    if column_token_map is not None:
        # 将DictConfig转换为普通字典
        column_token_map = dict(column_token_map)
    
    # 获取mask_column_token_map（从task配置中读取）
    mask_column_token_map = cfg.task.get("mask_column_token_map", None)
    if mask_column_token_map is not None:
        # 将DictConfig转换为普通字典（如果是字典）或保持字符串格式
        if isinstance(mask_column_token_map, dict):
            mask_column_token_map = dict(mask_column_token_map)
        # 如果是字符串，保持原样（PromptTemplate会处理）
    
    # 创建interface（generate_data阶段使用is_scoring_mode=True，query使用完整answer）
    interface = LLaDAInterface(
        model=model,
        tokenizer=tokenizer,
        task=cfg.task.task_name,
        mask_id=cfg.infer_model.get("mask_id", 126336),
        mask_length=cfg.infer_model.get("mask_length", 256),
        split_token=cfg.task.get("split_token", "\n\n"),
        prompt_template=prompt_template,
        is_scoring_mode=True,  # generate_data阶段：打分模式，query使用完整answer
        column_token_map=column_token_map,
        mask_column_token_map=mask_column_token_map,
    )
    
    return interface


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
    # 注意：单batch模式下不需要padding，所以不需要设置padding_side
    # 这里的padding逻辑出现了严重问题,如果需要支持padding ,需要从头去修改llada的架构逻辑,不能只是单纯的从文件夹里面抄模型
    #===========================================
    sleep(cfg.sleep_time * rank)
    interface = init_interface(cfg, device=process_device)
    # 单batch模式：不需要设置padding_side，因为不需要padding
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
    # 固定 resume 起点，避免在循环过程中 len(final_res) 变化导致索引越界
    start_idx = len(final_res)
    if start_idx >= subset_size:
        logger.info(f"Rank: {rank} task is Done.")
        return

    subset = subset.select(range(start_idx, len(subset)))
    sub_cand_set_idx = sub_cand_set_idx[start_idx:]
    for i, test_data in enumerate(
        tqdm(
            subset,
            disable=(rank != world_size - 1),
            total=subset_size,
            initial=start_idx,
            ncols=100,
        ),
    ):
        # 确保test_data有isquery标记
        if "isquery" not in test_data or test_data.get("isquery", 0) != 1:
            test_data = test_data.copy()
            test_data["isquery"] = 1
        
        # 获取当前anchor对应的候选集索引列表
        # subset/sub_cand_set_idx 已经基于 start_idx 对齐切片，因此这里直接用 i 索引即可
        candidate_indices = sub_cand_set_idx[i]
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


def _main_impl(cfg: DictConfig):
    """
    主函数实现：生成ICD序列数据
    """
    logger.info("="*80)
    logger.info("Stage 1: Data Generation")
    logger.info("="*80)
    
    # 创建必要的目录
    result_dir = cfg.get("result_dir", cfg.get("output_dir", "./generated_icd_data"))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    cache_dir = cfg.sampler.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    save_dir = os.path.join(result_dir, "generated_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sub_proc_save_dir = os.path.join(save_dir, "sub_proc_data")
    if not os.path.exists(sub_proc_save_dir):
        os.makedirs(sub_proc_save_dir)
    
    # 生成保存文件名（包含更多参数信息）
    dataset_name = cfg.dataset.get("name", cfg.task.task_name)
    model_name = cfg.infer_model.get("name", "llada")
    construct_order = cfg.get("metric", "no_order")
    sample_num = cfg.sampler.get(
        "anchor_sample_num", cfg.sampler.get("anchor_sample_num", 5000)
    )
    # 记录 mc_num，方便区分不同蒙特卡洛采样次数生成的结果文件
    mc_num = cfg.get("mc_num", 128)
    
    # 额外记录采样阶段的重要超参数（例如 coarse_k 与 mmr_lambda），方便后续加载与区分
    coarse_k = cfg.sampler.get("coarse_k", None)
    mmr_lambda = cfg.sampler.get("mmr_lambda", None)
    other_info = ""
    if coarse_k is not None and mmr_lambda is not None:
        other_info = f"-coarse_k:{coarse_k}-lambda:{mmr_lambda}"

    save_file_name = (
        f"{cfg.task.task_name}-{dataset_name}-"
        f"{model_name}-{cfg.sampler.sampler_name}-scorer:{cfg.scorer}-construct_order:{construct_order}-"
        f"beam_size:{cfg.beam_size}-few_shot:{cfg.few_shot_num}-"
        f"candidate_num:{cfg.sampler.candidate_num}-sample_num:{sample_num}-mc_num:{mc_num}"
        f"{other_info}.json"
    )
    
    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)
    
    # 1. 加载数据集
    logger.info("Loading datasets...")
    train_ds = load_ds(cfg, "train")
    logger.info(f"Train dataset size: {len(train_ds)}")
    
    # 2. 使用 sampler 生成 anchor 和 candidate_set（由 generate_data.yaml defaults 指定，如 sampler/text_sim_qwen_mmr）
    logger.info("Sampling anchor set and candidate sets...")
    try:
        sampler = hydra.utils.instantiate(
            cfg.sampler,
            index_ds_len=len(train_ds),
            dataset_name=cfg.task.task_name,
        )
        if not callable(sampler):
            raise ValueError("hydra instantiate returned non-callable object")
        logger.info(f"Sampler instantiated via Hydra: {type(sampler).__name__}")
    except Exception as e:
        logger.warning(
            f"Failed to instantiate sampler with hydra, falling back to RandSampler: {e}"
        )
        # 兜底时若 cfg.sampler 不可用（如配置未加载），只用根配置/默认值并打警告
        sampler_cfg = getattr(cfg, "sampler", None)
        if sampler_cfg is None:
            logger.warning(
                "cfg.sampler is missing (config may be incomplete), using root/default values for RandSampler"
            )
        try:
            candidate_num = sampler_cfg.get("candidate_num", cfg.get("candidate_num", 64)) if sampler_cfg is not None else cfg.get("candidate_num", 64)
            sampler_name = sampler_cfg.get("sampler_name", "random_sampler") if sampler_cfg is not None else "random_sampler"
            anchor_sample_num = sampler_cfg.get("anchor_sample_num", cfg.get("anchor_sample_num", 100)) if sampler_cfg is not None else cfg.get("anchor_sample_num", 100)
            cache_dir = sampler_cfg.get("cache_dir", "./cache") if sampler_cfg is not None else "./cache"
            overwrite = sampler_cfg.get("overwrite", False) if sampler_cfg is not None else False
        except Exception as cfg_err:
            logger.warning(
                f"Could not read sampler params from config, using defaults: {cfg_err}"
            )
            candidate_num = 64
            sampler_name = "random_sampler"
            anchor_sample_num = 100
            cache_dir = "./cache"
            overwrite = False
        sampler = RandSampler(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=len(train_ds),
            dataset_name=cfg.task.task_name,
            cache_dir=cache_dir,
            overwrite=overwrite,
        )
        logger.info("Sampler fallback instantiated: RandSampler")
    
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
        
        # 使用select方法直接从train_ds中选择anchor数据，并设置isquery标记
        anchor_data = train_ds.select(sampler_result["anchor_set"])
        # 为所有anchor数据添加isquery标记
        def add_isquery(example):
            example["isquery"] = 1
            return example
        anchor_data = anchor_data.map(add_isquery)
        
        candidate_set_idx = [
            candidate_set_dict[k] for k in sampler_result["anchor_set"]
        ]
        
        # 更新cfg以包含gpu_ids和sleep_time
        cfg.gpu_ids = gpu_ids
        cfg.sleep_time = sleep_time
        
        # 使用torch.multiprocessing启动多进程
        # 注意：参考代码使用spawn函数，这里使用mp.Process
        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(len(gpu_ids)):
            p = mp.Process(
                target=gen_data,
                args=(rank, cfg, anchor_data, train_ds, candidate_set_idx, sub_save_path)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        # 如果有子进程异常退出，直接报错，避免 merge 出不完整结果
        failed = [p for p in processes if p.exitcode not in (0, None)]
        if failed:
            msg = ", ".join([f"pid={p.pid} exitcode={p.exitcode}" for p in failed])
            raise RuntimeError(f"One or more worker processes failed: {msg}")
        
        logger.info("All processes completed. Merging results from all ranks...")
        
        # 合并所有rank的结果（参考代码的逻辑）
        world_size = len(gpu_ids)
        subset_size = len(anchor_data) // world_size
        total_data = {}
        for rank in range(world_size):
            subset_start = rank * subset_size
            subset_end = (
                subset_start + subset_size if rank != world_size - 1 else len(anchor_data)
            )
            sub_res_basename = (
                os.path.basename(save_path).split(".")[0]
                + f"_rank:{rank}_({subset_start}, {subset_end}).json"
            )
            rank_save_path = sub_save_path.replace(
                os.path.basename(sub_save_path), sub_res_basename
            )
            if os.path.exists(rank_save_path):
                with open(rank_save_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Load data from {rank_save_path}, data length: {len(data)}")
                total_data.update(data)
        
        # 保存最终结果
        with open(save_path, "w") as f:
            json.dump(total_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Save final data to {save_path}")
        
        return total_data
    else:
        # 单卡模式：使用原有的generate_icd_for_all_anchors
        logger.info("Using single-GPU mode")
        
        # 初始化interface（用于计算InfoScore）
        try:
            device = f"cuda:{cfg.gpu_ids[0]}" if cfg.get("gpu_ids") and len(cfg.gpu_ids) > 0 else "cuda:0"
            interface = init_interface(cfg, device=device)
            # 单batch模式：不需要设置padding_side，因为不需要padding
            logger.info(f"Interface initialized on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize interface: {e}")
            logger.error("Please check your model_path configuration in configs/infer_model/llada.yaml")
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
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {save_path}")
        logger.info(f"Total anchor samples processed: {len(all_results)}")
        
        # 打印统计信息
        total_sequences = sum(len(r["id_list"]) for r in all_results.values())
        logger.info(f"Total ICD sequences generated: {total_sequences}")
        
        return all_results


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data.yaml"
)
def hydra_loguru_init(cfg: DictConfig) -> None:
    """Hydra入口函数，初始化loguru日志"""
    hydra_path = hydra.core.hydra_config.HydraConfig.get().run.dir
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name
    logger.remove()
    logger.add(sys.stderr, level=hydra.core.hydra_config.HydraConfig.get().verbose)
    logger.add(os.path.join(hydra_path, f"{job_name}.log"))
    
    # 调用实际的主函数
    _main_impl(cfg)


if __name__ == "__main__":
    hydra_loguru_init()