# -*- coding: utf-8 -*-
from operator import truediv
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import accelerate
from tqdm import tqdm
import torch
from pathlib import Path
from collections import defaultdict
import textwrap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

import os
# 导入可视化函数

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False



def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel max采样方法，用于分类分布采样
    根据arXiv:2409.02908，对于MDM，低精度Gumbel Max提高困惑度分数但降低生成质量
    因此使用float64
    
    Args:
        logits: 模型输出的logits
        temperature: 采样温度
        
    Returns:
        添加Gumbel噪声后的logits
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    在反向过程中，区间[0,1]被均匀离散化为steps个区间
    由于LLaDA采用线性噪声调度（如Eq.(8)定义），
    每步预期的token转移数量应该是一致的
    
    此函数预计算每步需要转移的token数量
    
    Args:
        mask_index: 掩码索引张量
        steps: 采样步数
        
    Returns:
        每步转移token数量的张量
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens

#计算IEAR(AR 版本和NAR 版本)
def calculate_iear_metrics(
    attentions: tuple[torch.Tensor],
    current_sequence_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    # mask_id: int,
    gen_start:int, #限制指标计算区域
    gen_length: int,  # 新增参数
    split_label: str = "\n\n"  # 使用\n\n作为ICE分隔符
) -> Optional[Dict]:
    """
    计算改进版的 IEAR 指标。
    
    同时关注：
    1. Individual-level (点对点): 每个Current ICE token从Previous/Current获得的平均注意力
    2. Collective-level (整体): Previous/Current ICE作为整体对Current ICE的总体影响力
    
    Args:
        attentions: 模型所有层的注意力权重元组，形状为 [B, H, S, S]
        current_sequence_ids: 当前批次的token ID张量，形状为 [B, S]
        tokenizer: 用于解码的分词器
        mask_id: mask token的ID，用于识别生成部分
        split_label: 用于分隔ICE的字符串，默认为"\n\n"
    
    Returns:
        包含多层次IEAR指标的字典，或在不适用时返回None
    """
    if attentions is None or len(attentions) == 0:
        return None

    # 当前序列（去掉 batch 维）
    seq_ids = current_sequence_ids[0]  # 长度 S
    S = len(seq_ids)

    # 1. 找到anchor token的位置（需要排除的特殊token）
    decoded = [tokenizer.decode([tid]) for tid in seq_ids]
    anchor_chars = {'.', '\n'}
    anchor_idx = {i for i, s in enumerate(decoded) if (s.strip() == '.' or s == '\n')}

    # 1. 找到ICE的边界（使用\n\n作为分隔符）
    # 获取split_label的token IDs
    split_token_ids = tokenizer.encode(split_label, add_special_tokens=False)
    split_positions = []
    gen_end = gen_start + gen_length
    
    # 这里可以进行改进
    # 找到所有分隔符的位置
    # split_positions = []
    for i in range(S - len(split_token_ids) + 1):
        # 检查是否匹配分隔符序列
        if all(seq_ids[i + j] == split_token_ids[j] for j in range(len(split_token_ids))):
            # 排除生成区域内的分隔符
            if not (gen_start <= i < gen_end):
                split_positions.append(i + len(split_token_ids) - 1)  # 记录分隔符的最后一个位置

            # else:
            #     # 🔍 调试2：打印被排除的分隔符
            #     print(f"[IEAR调试] 排除生成区域内的分隔符: position {i}")

    # 🔍 调试3：打印找到的分隔符位置
    # print(f"[IEAR调试] 找到 {len(split_positions)} 个有效分隔符: {split_positions}")

    # 如果没有分隔符，无法计算IEAR
    if len(split_positions) < 2:
        # print(f"[IEAR调试] ❌ 分隔符不足2个，返回None")
        return None

    # 3. 定义ICE边界
    # ICE边界: [0, split_pos[0]], [split_pos[0]+1, split_pos[1]], ..., [split_pos[-2]+1, split_pos[-1]]
    # 最后一个分隔符之后到generation_start之前的是最后的query（需要排除）
    ice_boundaries = []

    # 第一个ICE: 从序列开始到第一个分隔符
    if split_positions[0] > 2:  # 确保ICE有足够的token
        ice_boundaries.append((0, split_positions[0]))

    # 中间的ICE: 每两个分隔符之间
    for i in range(len(split_positions) - 1):
        start = split_positions[i] + 1
        end = split_positions[i + 1]
        # 确保ICE不与生成区域重叠
        if end < gen_start or start >= gen_end:
            ice_boundaries.append((start, end))

        # else:
        #     # 🔍 调试4：打印被排除的ICE
        #     print(f"[IEAR调试] 排除与生成区域重叠的ICE: [{start}, {end}]")

    # 🔧 新增：最后一个ICE（从最后一个分隔符到序列末尾）
    if len(split_positions) > 0:
        last_start = split_positions[-1] + 1
        last_end = S - 1  # 序列末尾

        # 判断是否与生成区域重叠
        if last_end < gen_start or last_start >= gen_end:
            # 完全不重叠，添加整个ICE
            ice_boundaries.append((last_start, last_end))
            # print(f"[IEAR调试] 添加最后一个ICE: [{last_start}, {last_end}]")
        elif last_start < gen_start <= last_end:
            # 被生成区域分割，只保留前半段
            ice_boundaries.append((last_start, gen_start - 1))
            # print(f"[IEAR调试] 添加最后一个ICE的前半段: [{last_start}, {gen_start - 1}]")
        elif last_start < gen_end <= last_end:
            # 被生成区域分割，只保留后半段
            ice_boundaries.append((gen_end, last_end))
            # print(f"[IEAR调试] 添加最后一个ICE的后半段: [{gen_end}, {last_end}]")
        # else:
        #     print(f"[IEAR调试] 最后一个ICE完全在生成区域内，跳过: [{last_start}, {last_end}]")

    # 🔍 调试5：打印最终的ICE边界
    # print(f"[IEAR调试] 识别到 {len(ice_boundaries)} 个有效ICE:")
    # for idx, (start, end) in enumerate(ice_boundaries):
    #     ice_text = tokenizer.decode(seq_ids[start:end+1])[:1000]  # 只显示前50个字符
    #     print(f"  ICE {idx}: [{start:4d}, {end:4d}] 长度={end-start+1:3d} | 内容: {ice_text}...")
    

    
    if len(ice_boundaries)<2:
        # print(f"[IEAR调试] ❌ 有效ICE不足2个，返回None")
        return None

    # 添加调试信息：显示识别到的ICE数量
    # print(f"[IEAR] 识别到 {len(ice_boundaries)} 个ICE（仅在生成区域之前，gen_start={gen_start}）")

    # 4. 对每个ICE，计算其内部token对intra/extra的注意力分配
    num_layers = len(attentions)


    #逐层存储指标
    # intra_attentions_per_layer:List[float] = []#每层所有ICE的平均intra注意力
    # extra_attentions_per_layer:List[float] = []#每层所有ICE的平均extra注意力
    iear_ratio_per_layer:List[float] = []#每层IEAR的整体比率
    iear_ratio_per_layer_individual:List[float] = []#每层所有ICE的平均个体注意力

    
    for l in range(num_layers):
        att_l=attentions[l] #[N,H,S,S]
        if att_l.dim() != 4 or att_l.size(-1) != S or att_l.size(-2) != S:
            continue

        #头平均注意力[S，S]
        ave_att=att_l[0].mean(dim=0)
        
        #对每个ICE分别计算
        ice_intra_scores=[]
        ice_extra_scores=[]
        ice_intra_scores_individual=[]
        ice_extra_scores_individual=[]

        for ice_idx,(ice_start,ice_end) in enumerate(ice_boundaries):
            #当前ICE的token索引
            current_ice_tokens=set(range(ice_start,ice_end+1))

            # 添加调试信息
            # print(f"ICE {ice_idx}: 原始范围 [{ice_start}, {ice_end}], 原始token数: {len(current_ice_tokens)}")


            current_ice_tokens = sorted(current_ice_tokens.difference(anchor_idx))
            
            # 添加更多调试信息 - 修复类型错误
            # original_tokens = set(range(ice_start, ice_end+1))
            # anchor_tokens = original_tokens - set(current_ice_tokens)
            # print(f"ICE {ice_idx}: 过滤anchor后token数: {len(current_ice_tokens)}")
            # print(f"ICE {ice_idx}: anchor token数: {len(anchor_tokens)}")

            #其他ICE的token索引
            other_ice_tokens=set()
            for other_idx,(other_start,other_end) in enumerate(ice_boundaries):
                if other_idx !=ice_idx:
                    other_ice_tokens.update(range(other_start,other_end+1))

            other_ice_tokens=sorted(other_ice_tokens.difference(anchor_idx))

            #转为张量
            current_tensor=torch.tensor(current_ice_tokens, device=ave_att.device)
            other_tensor=torch.tensor(other_ice_tokens, device=ave_att.device)

            #提取当前ICE token的注意力
            current_ice_att=ave_att.index_select(dim=0,index=current_tensor)

            #计算总流入
            total_inflow=current_ice_att.sum(dim=1)  # [|current_valid|]
            total_inflow=torch.where(total_inflow==0,torch.ones_like(total_inflow),total_inflow)#防止都为0

            #Intra-ICE：当前ICE内部有效token之间的注意力
            intra_att=current_ice_att.index_select(dim=1,index=current_tensor)
            intra_score=(intra_att.sum(dim=1)/total_inflow).mean().item()

            #Extra-ICE:当前ICE对其他ICE有效token的注意力
            extra_att=current_ice_att.index_select(dim=1,index=other_tensor)   # [|current|, |current|]
            extra_score=(extra_att.sum(dim=1)/total_inflow).mean().item()
            
            # print("len(current_ice_tokens):",len(current_ice_tokens))
            # print("len(other_ice_tokens):",len(other_ice_tokens))

            #计算个体指标
            intra_score_individual=intra_score/len(current_ice_tokens)
            extra_score_individual=extra_score/len(other_ice_tokens)

            ice_intra_scores.append(intra_score)
            ice_extra_scores.append(extra_score)
            ice_intra_scores_individual.append(intra_score_individual)
            ice_extra_scores_individual.append(extra_score_individual)

        #计算该层所有ICE的平均
        #同理，这个也是整体，还是需要去看个体指标
        if len(ice_intra_scores)>0:
            #这个是整体
            avg_intra = np.mean(ice_intra_scores)
            avg_extra=np.mean(ice_extra_scores)
            #这个是个体
            avg_intra_individual=np.mean(ice_intra_scores_individual)
            avg_extra_individual=np.mean(ice_extra_scores_individual)
            #计算两种指标的比率
            ratio_collective = avg_intra / avg_extra if avg_extra >1e-9 else 0.0
            ratio_individual = avg_intra_individual / avg_extra_individual if avg_extra_individual >1e-9 else 0.0

            # intra_attentions_per_layer.append(intra_score)
            # extra_attentions_per_layer.append(extra_score)
            iear_ratio_per_layer.append(ratio_collective)
            iear_ratio_per_layer_individual.append(ratio_individual)
        else:
            # intra_attentions_per_layer.append(0.0)
            # extra_attentions_per_layer.append(0.0)
            iear_ratio_per_layer.append(0.0)
            iear_ratio_per_layer_individual.append(0.0)

    return {
        #主要指标
        # "intra_attentions_per_layer": intra_attentions_per_layer,
        # "extra_attentions_per_layer": extra_attentions_per_layer,
        "iear_ratio_per_layer": iear_ratio_per_layer,#整体比率
        "iear_ratio_per_layer_individual": iear_ratio_per_layer_individual,#个体比率
    }


def plot_iear_metrics_per_step(
    all_step_metrics:List[dict],
    # all_decoded_texts:List[str], 
    save_dir:str="IEAR_analysis_per_step"):
    """
    为每个去噪步骤生成IEAR指标图，包含两个子图：
    - 左图：Individual-level IEAR (点对点平均注意力比率)
    - 右图：Collective-level IEAR (整体影响力比率)
    
    Args:
        all_step_metrics: 包含每一步IEAR指标字典的列表
        all_decoded_texts: 包含每一步解码文本的列表
        save_dir: 图片保存目录
    """
    # --- 1. 预处理和检查 ---
    if not all_step_metrics:
        print("没有可供可视化的IEAR指标。")
        return

    # 过滤掉无效的步骤数据
    valid_metrics = [m for m in all_step_metrics if m and "iear_ratio_per_layer_individual" in m and "iear_ratio_per_layer" in m]
    if not valid_metrics:
        print("所有步骤均无可供可视化的IEAR指标。")
        return

    # --- 2. 创建保存图片的目录 ---
    save_directory = Path(save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)
    print(f"开始生成 {len(valid_metrics)} 张IEAR分析图，将保存至 '{save_directory}' 目录...")

    # --- 3. 遍历每一步，生成一张图 ---
    for step_idx, metrics in tqdm(enumerate(valid_metrics), total=len(valid_metrics), desc="生成IEAR分析图中"):
        # 提取当前步骤的两种IEAR指标数据
        iear_individual = metrics["iear_ratio_per_layer_individual"]
        iear_collective = metrics["iear_ratio_per_layer"]

        num_layers = len(iear_individual)
        layers_x_axis = range(num_layers)

        # --- 4. 创建包含两个子图的画布（1行2列） ---
        fig,(ax1,ax2) =plt.subplots(1,2,figsize=(16,6))

        # 为整张图设置一个总标题
        fig.suptitle(f'IEAR 指标分析 (去噪步骤 Step {step_idx})', fontsize=16, fontweight='bold')

        # --- 5. 绘制左子图：Individual-level IEAR ---
        ax1.plot(layers_x_axis, iear_individual, marker='o', linestyle='-', color='darkorange', label='Individual IEAR', linewidth=2)
        # 添加置信区间式的填充
        avg_val = np.mean(iear_individual)
        std_val = np.std(iear_individual)
        ax1.fill_between(layers_x_axis, 
                         np.array(iear_individual) - std_val, 
                         np.array(iear_individual) + std_val, 
                         color='darkorange', alpha=0.2)
        ax1.set_title('Individual-level IEAR (点对点平均比率)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Layer Number', fontsize=12)
        ax1.set_ylabel('IEAR Individual Value', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=11)

               # --- 6. 绘制右子图：Collective-level IEAR ---
        ax2.plot(layers_x_axis, iear_collective, marker='s', linestyle='-', color='dodgerblue', label='Collective IEAR', linewidth=2)
        avg_val_2 = np.mean(iear_collective)
        std_val_2 = np.std(iear_collective)
        ax2.fill_between(layers_x_axis, 
                         np.array(iear_collective) - std_val_2, 
                         np.array(iear_collective) + std_val_2, 
                         color='dodgerblue', alpha=0.2)
        ax2.set_title('Collective-level IEAR (整体影响力比率)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Layer Number', fontsize=12)
        ax2.set_ylabel('IEAR Collective Value', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=11)

                # --- 7. 调整布局并保存 ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
        filename = save_directory / f"step_{step_idx:03d}_iear_analysis.png"
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)  # 关键：在循环中关闭画布，防止内存泄漏

    # 循环结束后统一输出
    print(f"所有 {len(valid_metrics)} 张IEAR分析图已成功保存。")


# def plot_metrics_per_step(all_step_metrics:List[dict], all_decoded_texts:List[str], save_dir:str="ACAR_analysis_per_step"):
#     """
#     修改后的可视化函数：
#     为每个去噪步骤生成一张图，图中包含两个子图，
#     分别展示两种ACAR指标随“层数”的变化。
#     同时，我现在想看到的是每一步的解码过程
#     Args:
#         all_step_metrics: 包含每一步指标字典的列表。
#         save_path: 图片保存路径。
#     """
#     # --- 1. 预处理和检查 ---
#     if not all_step_metrics:
#         print("没有可供可视化的指标。")
#         return

#     # 过滤掉无效的步骤数据
#     valid_metrics = [m for m in all_step_metrics if m and "ratio_scaled_avg" in m and "ratio_collective" in m]
#     if not valid_metrics:
#         print("所有步骤均无可供可视化的指标。")
#         return

#     # --- 2. 创建保存图片的目录 ---
#     save_directory = Path(save_dir)
#     save_directory.mkdir(parents=True, exist_ok=True)
#     print(f"开始生成 {len(valid_metrics)} 张分析图，将保存至 '{save_directory}' 目录...")

#     # --- 3. 遍历每一步，生成一张图 ---
#     for step_idx, metrics in tqdm(enumerate(valid_metrics), total=len(valid_metrics), desc="生成分析图中"):
        
#         # 提取当前步骤的两种指标数据
#         ratio_scaled_avg = metrics["ratio_scaled_avg"]
#         ratio_collective = metrics["ratio_collective"]

#         num_layers = len(ratio_scaled_avg)
#         layers_x_axis = range(num_layers)


#         # # --- 4. 创建包含两个子图的画布 ---
#         # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

#         # --- 4. 创建包含两个子图和文本区域的画布 ---
#         fig = plt.figure(figsize=(18, 9))
        
#         # 创建子图布局：上方两个图表，下方文本区域
#         gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], hspace=0.3)
#         ax1 = fig.add_subplot(gs[0, 0])  # 左上
#         ax2 = fig.add_subplot(gs[0, 1])  # 右上
#         ax_text = fig.add_subplot(gs[1, :])  # 下方跨两列

#         # 为整张图设置一个总标题
#         fig.suptitle(f'ACAR 指标分析 (去噪步骤 Step {step_idx})', fontsize=16)
    

#         # --- 5. 绘制左子图：ratio_scaled_avg ---
#         ax1.plot(layers_x_axis, ratio_scaled_avg, marker='o', linestyle='-', color='darkorange', label='ACAR Value')
#         # 模仿你提供的图片风格，添加置信区间式的填充
#         avg_val = np.mean(ratio_scaled_avg)
#         std_val = np.std(ratio_scaled_avg)
#         ax1.fill_between(layers_x_axis, 
#                          np.array(ratio_scaled_avg) - std_val, 
#                          np.array(ratio_scaled_avg) + std_val, 
#                          color='darkorange', alpha=0.2)
#         ax1.set_title('指标一: Scaled Average Ratio')
#         ax1.set_xlabel('Number of Layers')
#         ax1.set_ylabel('ACAR_avg Value')
#         ax1.grid(True, linestyle=':', alpha=0.6)
#         ax1.legend()

#         # --- 6. 绘制右子图：ratio_collective ---
#         ax2.plot(layers_x_axis, ratio_collective, marker='s', linestyle='-', color='dodgerblue', label='ACAR Value')
#         avg_val_2 = np.mean(ratio_collective)
#         std_val_2 = np.std(ratio_collective)
#         ax2.fill_between(layers_x_axis, 
#                          np.array(ratio_collective) - std_val_2, 
#                          np.array(ratio_collective) + std_val_2, 
#                          color='dodgerblue', alpha=0.2)
#         ax2.set_title('指标二: Collective Influence Ratio')
#         ax2.set_xlabel('Number of Layers')
#         ax2.set_ylabel('ACAR_col Value')
#         ax2.grid(True, linestyle=':', alpha=0.6)
#         ax2.legend()

#         #将解码的过程标注在下方
#         ax_text.axis('off')  # 隐藏坐标轴
#         if step_idx < len(all_decoded_texts):
#             decoded_text = all_decoded_texts[step_idx]
#             # 对超长文本进行换行与截断，避免绘图时字形栅格溢出
#             max_chars = 6000  # 最大展示字符数
#             if len(decoded_text) > max_chars:
#                 decoded_text = decoded_text[:max_chars] + "\n...[截断]"
#             # 避免matplotlib把$当作mathtext解析导致报错：转义所有$
#             decoded_text = decoded_text.replace("$", r"\$")
#             wrapped = textwrap.fill(decoded_text, width=160)
#             ax_text.text(
#                 0.02,
#                 0.5,
#                 f"Step {step_idx} 解码结果:\n{wrapped}",
#                 fontsize=9,
#                 verticalalignment='center',
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
#                 wrap=True,
#                 clip_on=True,
#             )
        
#         else:
#             ax_text.text(0.02, 0.5, f"Step {step_idx} 解码结果: 无数据", 
#                 fontsize=10, verticalalignment='center',
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))     

#         # --- 7. 保存并关闭当前画布 ---
#         # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局为总标题留出空间
        
#         # # 使用补零命名，方便文件排序，例如 step_001.png
#         # filename = save_directory / f"step_{step_idx:03d}_analysis.png"
#         # plt.savefig(filename, dpi=120) # 使用适中的DPI以平衡清晰度和文件大小


#         # --- 8. 保存并关闭当前画布 ---
#         filename = save_directory / f"step_{step_idx:03d}_analysis.png"
#         plt.savefig(filename, dpi=96, bbox_inches='tight')
#         plt.close(fig) # 关键：在循环中关闭画布，防止内存泄漏

#     # 循环结束后统一输出与生成热图
#     print(f"所有 {len(valid_metrics)} 张分析图已成功保存。")
#     # if confidence_matrix:
#     #     print("\n[分析] 生成置信度热图...")
#     #     target_heatmap_dir = heatmap_dir if heatmap_dir else (save_dir + "_heatmap")
#     #     create_decode_heatmap(confidence_matrix, gen_length=256, save_dir=target_heatmap_dir)

#         # 循环内不再生成热力图，也不逐步打印“已保存”汇总


    

#需要进行改动，因为我这个generate是针对我这个prompt的，而不是针对原始的prompt，也就是说query一直在变化
@torch.no_grad()
def generate(
    model: AutoModel,
    prompt: torch.Tensor,
    gen_start: int,
    steps: int = 1024,
    gen_length: int = 1024,
    block_length: int = 1024,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    output_attentions: bool = False,  # 新增参数
    IEAR_analyse: bool = False,
    sample_idx: int = None,  # 新增参数
    query_position: int = 0  # 新增参数
) -> Union[torch.Tensor, tuple[torch.Tensor, List]]:
    """
    LLaDA生成函数，与原始仓库代码完全对齐
    
    Args:
        model: LLaDA模型
        prompt: 输入提示张量，形状为(1, L)
        steps: 采样步数，小于等于gen_length
        gen_length: 生成答案长度
        block_length: 块长度，小于等于gen_length。如果小于gen_length，表示使用半自回归重掩码
        temperature: 分类分布采样温度
        cfg_scale: 无监督分类器自由引导缩放
        remasking: 重掩码策略。'low_confidence'或'random'
        mask_id: [MASK]的token id，默认为126336
        
    Returns:
        生成的序列张量
    """
    # prompt 此时应是“已展开”的序列：
    # - 中间填充：prefix + [MASK]*gen_length + suffix
    # - 尾部补全：input_ids + [MASK]*gen_length
    # 这里不再追加 mask，而是直接在传入的 prompt 上操作



    x = prompt.clone().to(model.device)
    
    prompt_index = (x != mask_id)


    # 仅在需要时初始化列表
    trigger_analysis = output_attentions and IEAR_analyse
    all_step_metrics = [] if trigger_analysis else None
    #增加解码的架构，方便人工标注解码的当下步骤，更多的是关注什么
    # all_decoded_texts = [] if trigger_analysis else None  # 新增这行
    # 不再追踪每步解码的位置
    # confidence_matrix=[] if trigger_analysis else None#追踪每步生成区域的置信度
    
    # 对于非半自回归架构，block_length应该等于gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    


    for num_block in range(num_blocks):
# 取当前块中仍为 mask 的位置
        block_mask_index = (
            x[:, gen_start + num_block * block_length : gen_start + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        #先加入进度条，我要看一下每一步为什么解码这么慢
        progress_bar = tqdm(range(steps), desc=f"去噪 Block {num_block+1}/{num_blocks}", leave=False)
        #for i in range(steps):
        for i in progress_bar:
            mask_index = (x == mask_id)
            
            # 分类器自由引导
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                outputs = model(x_, output_attentions=output_attentions)  # 修改这里
                logits = outputs.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                if output_attentions and outputs.attentions:
                    cond_attentions = tuple(torch.chunk(att, 2, dim=0)[0] for att in outputs.attentions)
                    attentions_to_analyze = cond_attentions

            else:
                outputs=model(x,output_attentions=output_attentions)
                logits=outputs.logits
                if output_attentions and outputs.attentions:
                    attentions_to_analyze = outputs.attentions



            #每一步都计算，但是我现在想对每一步中的每一层进行头平均
            if trigger_analysis and attentions_to_analyze:
                # 注意：将x和attentions都移动到CPU进行计算，可以进一步减少GPU显存峰值
                metrics = calculate_iear_metrics(
                    attentions=tuple(att.cpu() for att in attentions_to_analyze), 
                    current_sequence_ids=x.cpu(), 
                    tokenizer=tokenizer, 
                    gen_start=gen_start,
                    gen_length=gen_length,  # 传入生成长度
                )
                if metrics:
                    all_step_metrics.append(metrics)



            # 添加Gumbel噪声
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
            
            # 重掩码策略
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            # 仅允许在“当前块”内采样：块前、块后都设为 -inf
            x0_p[:, : gen_start + num_block * block_length] = -np.inf
            x0_p[:, gen_start + (num_block + 1) * block_length :] = -np.inf
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # 选择置信度最高的token进行转移
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]


            #记录解码的位置主要是想做heatmap
            # 记录当前步解码的位置
            # if trigger_analysis:
            #     #获取生成区域的置信度（相对于gen_start)
            #     gen_region_confidence=confidence[0,gen_start:gen_start+gen_length].to(torch.float32).cpu().numpy()
            #     #替换-inf为一个合理的最小值（用于可视化）
            #     gen_region_confidence=np.where(gen_region_confidence == -np.inf,np.nan,gen_region_confidence)
            #     confidence_matrix.append(gen_region_confidence)


                # 不再记录每步解码位置

            # --- 内存释放的关键步骤（删掉大量的权重）---
            # 在循环的末尾，显式删除不再需要的大张量
            if outputs is not None:
                del outputs
            if attentions_to_analyze is not None:
                del attentions_to_analyze
            
            # 如果使用GPU，强制清理PyTorch的缓存内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_generated_tokens = x[0][gen_start : gen_start + gen_length]
            # 使用 skip_special_tokens=False 来看到 <|mdm_mask|>
            current_text = tokenizer.decode(current_generated_tokens, skip_special_tokens=False)
            # 直接打印，让终端自动处理换行
            print(current_text)
            print("-" * (len(f"--- [Block {num_block+1}, Step {i+1}/{steps}] ---"))) # 分隔线

            # 收集解码文本用于绘图
            # if trigger_analysis:
            #     all_decoded_texts.append(current_text)
        
            # --- 在生成循环结束后，执行保存和绘图 ---
        if trigger_analysis and all_step_metrics:
            print("\n[分析] 生成过程结束，开始生成IEAR分析图...")
            if sample_idx is not None:
                base_dir = Path(f"IEAR_results_{query_position}") / f"test_{sample_idx}"
                iear_dir = base_dir / "IEAR_analysis_output"
                # heatmap_dir = base_dir / "heatmap"
            else:
                base_dir = Path(f"IEAR_results_{query_position}")
                iear_dir = base_dir / "IEAR_analysis_output"
                # heatmap_dir = base_dir / "heatmap"
            iear_dir.mkdir(parents=True, exist_ok=True)
            # heatmap_dir.mkdir(parents=True, exist_ok=True)
            plot_iear_metrics_per_step(
                all_step_metrics,
                # all_decoded_texts,
                save_dir=str(iear_dir)
            )
    
    return x


class LLaDAInference:
    """
    LLaDA推理类，用于测试双向能力 - 优化版本
    """

    def __init__(
    self,
    model_path: str,
    device: str = "cuda",
    mask_id: int = 126336,
    max_length: int = 4096,
    use_accelerate: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    tokenizer: AutoTokenizer = None,
    model: AutoModel = None,
    **kwargs
    ):
        """
        初始化LLaDA推理类，用于测试双向能力
        Args:
            model_path: 模型路径
            device: 设备
            mask_id: [MASK]的token id
            max_length: 最大长度
            use_accelerate: 是否使用accelerate
            kwargs: 其他参数
        """
        self.model_path = model_path
        self.device = device
        self.mask_id = mask_id
        self.max_length = max_length
        self.use_accelerate = use_accelerate
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.model = model
        self.torch_dtype = torch_dtype
        
        # 性能优化：缓存机制
        self._mask_position_cache = {}  # 缓存mask位置信息
        self._tensor_cache = {}  # 缓存tensor转换结果

    def _get_mask_positions(self, input_ids: torch.Tensor, prompt_hash: str):
        """获取mask位置信息，使用缓存优化"""
        if prompt_hash in self._mask_position_cache:
            return self._mask_position_cache[prompt_hash]
        
        # 找到mask token的位置
        mask_positions = (input_ids == self.mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")

        # 找到第一个和最后一个mask token的位置
        first_mask_pos = mask_positions[1][0].item()
        last_mask_pos = mask_positions[1][-1].item()

        # 验证mask token是连续的
        expected_mask_count = last_mask_pos - first_mask_pos + 1
        actual_mask_count = len(mask_positions[1])
        if actual_mask_count != expected_mask_count:
            raise ValueError(f"Mask tokens are not continuous. Expected {expected_mask_count}, got {actual_mask_count}")
        
        result = (first_mask_pos, last_mask_pos)
        self._mask_position_cache[prompt_hash] = result
        return result

    def _process_stop_tokens(self, text: str, stop_tokens: Optional[List[str]]) -> str:
        """优化的停止token处理"""
        if not stop_tokens:
            return text
        
        # 找到最早出现的停止token
        min_pos = len(text)
        for stop_token in stop_tokens:
            pos = text.find(stop_token)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        if min_pos < len(text):
            return text[:min_pos]
        return text

    def generate_text(
        self,
        prompt: Union[str, List[int]],
        answer_length: int = 1024,
        sampling_steps: int = 1024,
        block_length: int = 1024,
        remask_strategy: str = "low_confidence",
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        stop_tokens: Optional[List[str]] = None,
        output_attentions: bool = False,  # 保留这个参数
        IEAR_analyse: bool = False,
        sample_idx:int|None=None,#新增，后期要删掉
        query_position:int=0,#新增，后期要删掉
    ) -> str:
        """
        通用方法：生成文本 - 优化版本
        Args:
            prompt: 提示（字符串中已包含mask token）
            answer_length: 答案长度
            sampling_steps: 采样步数
            block_length: 块长度
            remask_strategy: 重掩码策略
            temperature: 温度
            cfg_scale: 分类器自由引导缩放
            stop_tokens: 停止token
        Returns:
            生成的文本
        """
        # 优化：创建prompt的hash用于缓存
        if isinstance(prompt, str):
            input_ids = self.tokenizer(prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        else:
            input_ids = torch.tensor(prompt).to(self.device).unsqueeze(0)

        # 找到mask token的位置
        mask_positions = (input_ids == self.mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")

        # 找到第一个和最后一个mask token的位置
        first_mask_pos = mask_positions[1][0].item()
        last_mask_pos = mask_positions[1][-1].item()

        # 验证mask token是连续的
        expected_mask_count = last_mask_pos - first_mask_pos + 1
        actual_mask_count = len(mask_positions[1])
        if actual_mask_count != expected_mask_count:
            raise ValueError(f"Mask tokens are not continuous. Expected {expected_mask_count}, got {actual_mask_count}")

        # 执行生成（核心逻辑保持不变）
        generated = generate(
            model=self.model,
            prompt=input_ids,
            gen_start=first_mask_pos,
            steps=sampling_steps,
            gen_length=answer_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remask_strategy,
            mask_id=self.mask_id,
            output_attentions=output_attentions,  # 保留这个参数，因为函数定义中有
            IEAR_analyse=IEAR_analyse,
            sample_idx=sample_idx,  #新增，后期得删掉  
            query_position=query_position  # 新增参数
        )

        # # 处理返回结果
        # if output_attentions:
        #     generated, all_attentions = result
        # else:
        #     generated = result
        #     all_attentions = None
        
        # 优化：直接提取mask填充区域，减少索引操作
        mask_filled_tokens = generated[0][first_mask_pos:last_mask_pos+1]
        
        # 优化：一次性解码，减少重复的tokenizer调用
        generated_text = self.tokenizer.decode(mask_filled_tokens, skip_special_tokens=False)

        # 优化：使用更高效的停止token处理
        generated_text = self._process_stop_tokens(generated_text, stop_tokens)
        
        # 移除特殊token（保持原有逻辑）
        generated_text = generated_text.replace("<|mdm_mask|>", "").strip()

        # # 可选：保存注意力权重
        # if output_attentions and save_attentions_path:

        #     save_dir = Path(save_attentions_path)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     torch.save(all_attentions, save_dir / "attention_weights.pt")
        #     print(f"注意力权重已保存到: {save_dir / 'attention_weights.pt'}")

        # # 返回结果
        # if output_attentions:
        #     return generated_text, all_attentions
        # else:

        return generated_text

   


def create_llada_inference(
    model_path: str = None,
    device: str = "cuda",
    use_accelerate: bool = False,
    tokenizer: AutoTokenizer = None,
    model: AutoModel = None,
    mask_id: int = 126336,
    max_length: int = 4096,
    torch_dtype: torch.dtype = torch.bfloat16
) -> LLaDAInference:
    """
    便捷函数：创建LLaDA推理器
    
    Args:
        model_path: LLaDA模型路径（如果提供了tokenizer和model，此参数可选）
        device: 设备类型
        use_accelerate: 是否使用Accelerate
        tokenizer: 已加载的分词器（可选）
        model: 已加载的模型（可选）
        mask_id: 掩码token ID
        max_length: 最大长度
        torch_dtype: 模型精度
        
    Returns:
        LLaDAInference实例
    """
    return LLaDAInference(
        model_path=model_path,
        device=device,
        use_accelerate=use_accelerate,
        tokenizer=tokenizer,
        model=model,
        mask_id=mask_id,
        max_length=max_length,
        torch_dtype=torch_dtype
    )



# 测试用例
if __name__ == "__main__":
    # 导入llada_loader
    import sys
    import os
    import random
    import json
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from llada_loader import load_model
    from prompt_constructor_gsm8k import GSM8KPromptConstructor
    from utils import extract_gsm8k_answer
    from gsm8k_handler_v2 import GSM8KHandler
    
    print("=== LLaDA 批量ACAR分析测试 ===")
    
    #配置参数
    N_SAMPLES = 20 #从测试集随机抽取的样本数
    RANDOM_SEED = 1234 #随机种子
    random.seed(RANDOM_SEED)


    # 1. 使用llada_loader加载模型和分词器
    print("步骤1: 使用llada_loader加载模型...")
    model_path = "/home/share/model_weight/llada/LLaDA-8B-Base/"
    device = "cuda:3"
    
    model, tokenizer = load_model(
        model_path=model_path,
        device=device,
        use_accelerate=False,
        mask_id=126336,
        max_length=4096,
        torch_dtype=torch.bfloat16
    )
    
    print(f"模型加载完成，设备: {device}")
    print(f"分词器词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 使用已加载的模型和分词器创建推理器
    print("\n步骤2: 创建推理器...")
    inference = create_llada_inference(
        model_path=model_path,
        device=device,
        tokenizer=tokenizer,  # 传入已加载的分词器
        model=model,          # 传入已加载的模型
        mask_id=126336,
        max_length=4096,
        torch_dtype=torch.bfloat16
    )
    
    print("推理器创建完成")
    

    # 3. 加载GSM8K数据集
    print("\n步骤3: 加载GSM8K数据集...")
    data_handler = GSM8KHandler(data_dir="/home/share/datasets/gsm8k/")
    train_dataset, test_dataset = data_handler.prepare_for_evaluation("test", n_shots=4)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 4. 随机抽取测试样本
    print(f"\n步骤4: 从测试集随机抽取 {N_SAMPLES} 个样本...")
    test_indices = random.sample(range(len(test_dataset)), N_SAMPLES)
    test_samples = [test_dataset[i] for i in test_indices]
    print(f"抽取的测试样本索引: {test_indices[:10]}...")  # 只显示前10个

    # 5. 固定使用前4个训练样本作为few-shot示例
    print("\n步骤5: 准备few-shot示例...")
    train_samples = [train_dataset[i] for i in range(4)]

    for position in range(4,5):
        print(f"\n=======开始Position={position}的测试=======")
        # 6. 创建prompt构造器
        print("\n步骤6: 创建prompt构造器...")
        prompt_constructor = GSM8KPromptConstructor(n_shots=4, query_position=position)  # 使用默认位置
        
        # 7. 批量推理与分析 
        print(f"\n步骤6: 开始批量推理与ACAR分析 ({N_SAMPLES} 个样本)...")

        results = []

        for idx, test_sample in enumerate(test_samples, start=1):
            print(f"\n{'='*80}")
            print(f"处理样本 {idx}/{N_SAMPLES} (测试集索引: {test_indices[idx-1]})")
            print(f"{'='*80}")
            print(f"问题: {test_sample['question'][:200]}...")

            try:
                #构建prompt
                prompt = prompt_constructor.construct_prompt(train_samples, test_sample, mask_length=256)

                # 修改generate函数调用，传入sample_idx
                # 注意：需要在generate内部根据sample_idx构建输出目录
                generated_text = inference.generate_text(
                    prompt=prompt,
                    answer_length=256,
                    sampling_steps=256,
                    block_length=256,
                    temperature=0.0,
                    stop_tokens=["Question:", "Answer:"],
                    output_attentions=True,
                    IEAR_analyse=True,
                    sample_idx=idx, #表示这是第几个，后期要删掉
                    query_position=position #表示位置
                )

                # 提取答案
                predicted_answer = extract_gsm8k_answer(generated_text)
                true_answer = extract_gsm8k_answer(test_sample['answer'])
                is_correct = (predicted_answer == true_answer)

                result = {
                    "sample_idx": idx,
                    "test_dataset_idx": test_indices[idx-1],
                    "question": test_sample['question'],
                    "predicted_answer": predicted_answer,
                    "true_answer": true_answer,
                    "is_correct": is_correct,
                    "generated_text": generated_text
                }
                results.append(result)
                print(f"预测答案: {predicted_answer}")
                print(f"真实答案: {true_answer}")
                print(f"正确性: {'✓ 正确' if is_correct else '✗ 错误'}")

            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                
                result = {
                    "sample_idx": idx,
                    "test_dataset_idx": test_indices[idx-1],
                    "question": test_sample['question'],
                    "error": str(e),
                    "is_correct": False
                }
                results.append(result)


        # 8. 保存汇总结果
        print(f"\n{'='*80}")
        print("步骤7:保存汇总结果...")
        print(f"{'='*80}")

        # 创建results目录
        results_dir = Path(f"batch_analysis_iear_results_{position}")
        results_dir.mkdir(exist_ok=True)

        # 保存详细结果
        with open(results_dir / "all_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 计算并保存统计信息
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0
        
        summary = {
            "total_samples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "random_seed": RANDOM_SEED,
            "test_indices": test_indices,
            "query_position": position
        }

        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nposition={position}的最终统计:")
        print(f"- 总样本数: {total}")
        print(f"- 正确数: {correct}")
        print(f"- 错误数: {total - correct}")
        print(f"- 准确率: {accuracy:.2%}")
        print(f"\n结果已保存到: {results_dir}/")
        print(f"- 汇总统计: summary.json")
        print(f"- IEAR分析图: test_1/ ~ test_{N_SAMPLES}/ (各自的IEAR_analysis_output目录)")
    
    print("\n=== 批量测试完成 ===")
