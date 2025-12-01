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

#计算ACAR(AR 版本和NAR 版本)
def calculate_step_metrics(
    attentions: tuple[torch.Tensor],
    current_sequence_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    mask_id: int
) -> Optional[Dict]:
    """
    针对 LLaDA（NAR）在“单个去噪步骤”上，计算所有层的 Sap、Scp 和 Sap/Scp。
    - anchor: 解码后为 '.' 或 '\n' 的 token
    - prediction: 当前仍为 mask 的所有位置
    - context: 已解码且非 anchor 的位置
    attentions: 长度 = 层数；每层张量形状 [B, H, S, S]
    """
    if attentions is None or len(attentions) == 0:
        return None

    # 当前序列（去掉 batch 维）
    seq_ids = current_sequence_ids[0].tolist()  # 长度 S
    S = len(seq_ids)

    # 找集合：anchor / prediction / context
    # 逐 token 解码以识别 '.' 和 '\n'
    decoded = [tokenizer.decode([tid]) for tid in seq_ids]
    anchor_chars = {'.', '\n'}
    anchor_idx = {i for i, s in enumerate(decoded) if (s.strip() == '.' or s == '\n')}
    pred_idx = {i for i, tid in enumerate(seq_ids) if tid == mask_id}
    all_idx = set(range(S))
    context_idx = all_idx.difference(anchor_idx, pred_idx)

    #我要计算两个比率，一个是看向个体的，一个是看向群体的
    num_context_tokens = len(context_idx)
    num_anchor_tokens = len(anchor_idx)

    # 若没有预测 token（例如最后一步），返回 None
    if not pred_idx:
        return None

    # 转为有序 list，便于索引张量
    anchor_idx = sorted(anchor_idx)
    pred_idx = sorted(pred_idx)
    context_idx = sorted(context_idx)

    num_layers = len(attentions)
    Sap_lis: List[float] = []
    Scp_lis: List[float] = []

    #2是看向群体的，1是看向个体的
    ratio2_per_layer: List[float] = []
    ratio1_per_layer: List[float] = []

    # 按层计算
    for l in range(num_layers):
        # att: [B, H, S, S]，取 batch=0 并在 head 维平均 → [S, S]
        att_l = attentions[l]             # [B, H, S, S]
        if att_l.dim() != 4 or att_l.size(-1) != S or att_l.size(-2) != S:
            return None  # 形状不一致直接跳过
        ave = att_l[0].mean(dim=0)        # [S, S]


        # 仅在预测位置上取列向量：ave[pred_idx, :] → [|P|, S]
        if len(pred_idx) == 0:
            Sap_lis.append(0.0)
            Scp_lis.append(0.0)
            ratio2_per_layer.append(0.0)
            continue

        pred_mat = ave.index_select(dim=0, index=torch.tensor(pred_idx, device=ave.device))  # [|P|, S] 


        # 每个预测 token 的总流入（归一化分母）
        whole_flows = pred_mat.sum(dim=1)  # [|P|]
        whole_flows = torch.where(whole_flows == 0, torch.ones_like(whole_flows), whole_flows)

        # 来自 anchor 的流入占比
        if len(anchor_idx) > 0:
            anchor_cols = pred_mat.index_select(dim=1, index=torch.tensor(anchor_idx, device=ave.device))  # [|P|, |A|]
            sap_vec = anchor_cols.sum(dim=1) / whole_flows  # [|P|]
            Sap = sap_vec.mean().item() #这个就是取平均值了
        else:
            Sap = 0.0

        
        # 来自 context 的流入占比
        if len(context_idx) > 0:
            ctx_cols = pred_mat.index_select(dim=1, index=torch.tensor(context_idx, device=ave.device))  # [|P|, |C|]
            scp_vec = ctx_cols.sum(dim=1) / whole_flows  # [|P|]
            Scp = scp_vec.mean().item() #这个也是取平均值了
        else:
            Scp = 0.0

        Sap_lis.append(Sap)
        Scp_lis.append(Scp)
        ratio2 = Sap / Scp if Scp > 1e-9 else 0.0
        # 计算 Ratio₂ (集体影响力占比的比率)
        ratio2_per_layer.append(ratio2)

        #推算出Ration1(点对点平均占比的比率)
        scaling_factor = num_context_tokens / num_anchor_tokens if num_anchor_tokens > 0 else 0.0   
        ratio1 = ratio2 * scaling_factor
        ratio1_per_layer.append(ratio1)


    return {
        "sap_per_layer": Sap_lis,
        "scp_per_layer": Scp_lis,
        "ratio_collective": ratio2_per_layer, # 第二种方法的结果
        "ratio_scaled_avg": ratio1_per_layer   # 第一种方法的结果
    }

def create_decode_heatmap(confidence_matrix: List[np.ndarray], gen_length: int, save_dir: str = "decode_heatmap"):
    """
    创建解码位置热图
    Args:
        confidence_matrix: 每一步生成区域的置信度列表[steps,gen_length]
        gen_length: 生成长度
        save_dir: 保存目录
    """
    if not confidence_matrix:
        print("没有置信度数据，跳过热图生成")
        return

    # total_steps = len(decode_positions)
    total_steps = len(confidence_matrix)
    print(f"生成置信度热图：{total_steps}步 x {gen_length} tokens")

    #创建热图矩阵：steps x gen_length
    # 0=未解码（黄），1=已解码（蓝）
    # heatmap_matrix=np.zeros((total_steps,gen_length))
    heatmap_matrix=np.array(confidence_matrix) #shape:[steps,gen_length]

    #处理nan值，用于可视化
    heatmap_matrix=np.nan_to_num(heatmap_matrix,nan=-10.0,posinf=1.0,neginf=-10.0)

    #创建保存目录
    save_directory = Path(save_dir)
    save_directory.mkdir(parents=True,exist_ok=True)

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(16, 12))

    # 绘制置信度热图（蓝色=低置信度，黄色=高置信度）
    im = ax.imshow(heatmap_matrix, cmap='viridis', aspect='auto', interpolation='nearest')



    # 不再标注解码位置，只展示置信度热图

    #设置标签
    ax.set_xlabel('Token Position',fontsize=14,fontweight='bold')
    ax.set_ylabel('Decode Step',fontsize=14,fontweight='bold')
    ax.set_title('Confidence Heatmap with Decode Positions',fontsize=16,fontweight='bold',pad=20)

    # 设置坐标轴刻度
    # X轴：每20个token显示一次
    x_ticks = np.arange(0, gen_length, 20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=10)
    
    # Y轴：每20步显示一次
    y_ticks = np.arange(0, total_steps, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence Status', fontsize=12, fontweight='bold')
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['Not Decoded', 'Decoded'], fontsize=10)

    #创建可视化
    # fig ,ax=plt.subplots(figsize=(16,12))

    #绘制热图
    # im=ax.imshow(heatmap_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1, interpolation='nearest')


    # 添加图例说明  
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='yellow', alpha=0.7, label='High Confidence'),
        Patch(facecolor='blue', alpha=0.7, label='Low Confidence'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 保存图片
    heatmap_file = save_directory / "confidence_heatmap.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"置信度热图已保存: {heatmap_file}")
    
    # 生成统计信息
    print(f"\n解码统计:")
    print(f"- 总步数: {total_steps}")
    print(f"- 总token数: {gen_length}")
    
    # 位置统计移除，仅展示置信度整体分布

    # 计算平均置信度变化
    valid_conf = heatmap_matrix[heatmap_matrix > -5]  # 排除-inf
    print(f"- 平均置信度: {valid_conf.mean():.4f}")
    print(f"- 置信度标准差: {valid_conf.std():.4f}")
    
    # 保存详细统计
    stats_file = save_directory / "confidence_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"置信度热图统计信息\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"总步数: {total_steps}\n")
        f.write(f"总token数: {gen_length}\n")
        f.write(f"平均置信度: {valid_conf.mean():.4f}\n")
        f.write(f"置信度标准差: {valid_conf.std():.4f}\n\n")
        # 移除位置明细，仅保存整体统计
    
    print(f"置信度统计信息已保存: {stats_file}")



def plot_metrics_per_step(all_step_metrics:List[dict], all_decoded_texts:List[str], confidence_matrix:List[np.ndarray], save_dir:str="ACAR_analysis_per_step", heatmap_dir: str | None = None):
    """
    修改后的可视化函数：
    为每个去噪步骤生成一张图，图中包含两个子图，
    分别展示两种ACAR指标随“层数”的变化。
    同时，我现在想看到的是每一步的解码过程
    Args:
        all_step_metrics: 包含每一步指标字典的列表。
        save_path: 图片保存路径。
    """
    # --- 1. 预处理和检查 ---
    if not all_step_metrics:
        print("没有可供可视化的指标。")
        return

    # 过滤掉无效的步骤数据
    valid_metrics = [m for m in all_step_metrics if m and "ratio_scaled_avg" in m and "ratio_collective" in m]
    if not valid_metrics:
        print("所有步骤均无可供可视化的指标。")
        return

    # --- 2. 创建保存图片的目录 ---
    save_directory = Path(save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)
    print(f"开始生成 {len(valid_metrics)} 张分析图，将保存至 '{save_directory}' 目录...")

    # --- 3. 遍历每一步，生成一张图 ---
    for step_idx, metrics in tqdm(enumerate(valid_metrics), total=len(valid_metrics), desc="生成分析图中"):
        
        # 提取当前步骤的两种指标数据
        ratio_scaled_avg = metrics["ratio_scaled_avg"]
        ratio_collective = metrics["ratio_collective"]

        num_layers = len(ratio_scaled_avg)
        layers_x_axis = range(num_layers)


        # # --- 4. 创建包含两个子图的画布 ---
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        # --- 4. 创建包含两个子图和文本区域的画布 ---
        fig = plt.figure(figsize=(18, 9))
        
        # 创建子图布局：上方两个图表，下方文本区域
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # 左上
        ax2 = fig.add_subplot(gs[0, 1])  # 右上
        ax_text = fig.add_subplot(gs[1, :])  # 下方跨两列

        # 为整张图设置一个总标题
        fig.suptitle(f'ACAR 指标分析 (去噪步骤 Step {step_idx})', fontsize=16)
    

        # --- 5. 绘制左子图：ratio_scaled_avg ---
        ax1.plot(layers_x_axis, ratio_scaled_avg, marker='o', linestyle='-', color='darkorange', label='ACAR Value')
        # 模仿你提供的图片风格，添加置信区间式的填充
        avg_val = np.mean(ratio_scaled_avg)
        std_val = np.std(ratio_scaled_avg)
        ax1.fill_between(layers_x_axis, 
                         np.array(ratio_scaled_avg) - std_val, 
                         np.array(ratio_scaled_avg) + std_val, 
                         color='darkorange', alpha=0.2)
        ax1.set_title('指标一: Scaled Average Ratio')
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('ACAR_avg Value')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # --- 6. 绘制右子图：ratio_collective ---
        ax2.plot(layers_x_axis, ratio_collective, marker='s', linestyle='-', color='dodgerblue', label='ACAR Value')
        avg_val_2 = np.mean(ratio_collective)
        std_val_2 = np.std(ratio_collective)
        ax2.fill_between(layers_x_axis, 
                         np.array(ratio_collective) - std_val_2, 
                         np.array(ratio_collective) + std_val_2, 
                         color='dodgerblue', alpha=0.2)
        ax2.set_title('指标二: Collective Influence Ratio')
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('ACAR_col Value')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

        #将解码的过程标注在下方
        ax_text.axis('off')  # 隐藏坐标轴
        if step_idx < len(all_decoded_texts):
            decoded_text = all_decoded_texts[step_idx]
            # 对超长文本进行换行与截断，避免绘图时字形栅格溢出
            max_chars = 6000  # 最大展示字符数
            if len(decoded_text) > max_chars:
                decoded_text = decoded_text[:max_chars] + "\n...[截断]"
            # 避免matplotlib把$当作mathtext解析导致报错：转义所有$
            decoded_text = decoded_text.replace("$", r"\$")
            wrapped = textwrap.fill(decoded_text, width=160)
            ax_text.text(
                0.02,
                0.5,
                f"Step {step_idx} 解码结果:\n{wrapped}",
                fontsize=9,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                wrap=True,
                clip_on=True,
            )
        
        else:
            ax_text.text(0.02, 0.5, f"Step {step_idx} 解码结果: 无数据", 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))     

        # --- 7. 保存并关闭当前画布 ---
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局为总标题留出空间
        
        # # 使用补零命名，方便文件排序，例如 step_001.png
        # filename = save_directory / f"step_{step_idx:03d}_analysis.png"
        # plt.savefig(filename, dpi=120) # 使用适中的DPI以平衡清晰度和文件大小


        # --- 8. 保存并关闭当前画布 ---
        filename = save_directory / f"step_{step_idx:03d}_analysis.png"
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        plt.close(fig) # 关键：在循环中关闭画布，防止内存泄漏

    # 循环结束后统一输出与生成热图
    print(f"所有 {len(valid_metrics)} 张分析图已成功保存。")
    if confidence_matrix:
        print("\n[分析] 生成置信度热图...")
        target_heatmap_dir = heatmap_dir if heatmap_dir else (save_dir + "_heatmap")
        create_decode_heatmap(confidence_matrix, gen_length=256, save_dir=target_heatmap_dir)

        # 循环内不再生成热力图，也不逐步打印“已保存”汇总


    

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
    ACAR_analyse: bool = False,
    sample_idx: int = None,  # 新增参数
    query_position: int = 0,  # 新增参数
    tokenizer: AutoTokenizer = None  # 添加这个参数
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
    trigger_analysis = output_attentions and ACAR_analyse
    all_step_metrics = [] if trigger_analysis else None
    #增加解码的架构，方便人工标注解码的当下步骤，更多的是关注什么
    all_decoded_texts = [] if trigger_analysis else None  # 新增这行
    # 不再追踪每步解码的位置
    confidence_matrix=[] if trigger_analysis else None#追踪每步生成区域的置信度
    
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
            
            ###调试信息
            #在每一步推理前打印“输入给模型的prompt“
            if tokenizer is not None:
                try:
                    input_text = tokenizer.decode(x[0],skip_special_tokens=False)
                    print(input_text)
                    print("-"*80)
                except Exception as e:
                    print(f"解码当前步输入给模型的prompt失败: {e}")


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
                metrics = calculate_step_metrics(
                    attentions=tuple(att.cpu() for att in attentions_to_analyze), 
                    current_sequence_ids=x.cpu(), 
                    tokenizer=tokenizer, 
                    mask_id=mask_id
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
            if trigger_analysis:
                #获取生成区域的置信度（相对于gen_start)
                gen_region_confidence=confidence[0,gen_start:gen_start+gen_length].to(torch.float32).cpu().numpy()
                #替换-inf为一个合理的最小值（用于可视化）
                gen_region_confidence=np.where(gen_region_confidence == -np.inf,np.nan,gen_region_confidence)
                confidence_matrix.append(gen_region_confidence)


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
            # print(current_text)
            # print("-" * (len(f"--- [Block {num_block+1}, Step {i+1}/{steps}] ---"))) # 分隔线

            # 收集解码文本用于绘图
            if trigger_analysis:
                all_decoded_texts.append(current_text)
        
            # --- 在生成循环结束后，执行保存和绘图 ---
        if trigger_analysis and all_step_metrics:
            print("\n[分析] 生成过程结束，开始生成分析图...")
            if sample_idx is not None:
                base_dir = Path(f"results_{query_position}") / f"test_{sample_idx}"
                acar_dir = base_dir / "ACAR_analysis_output"
                heatmap_dir = base_dir / "heatmap"
            else:
                base_dir = Path(f"results_{query_position}")
                acar_dir = base_dir / "ACAR_analysis_output"
                heatmap_dir = base_dir / "heatmap"
            acar_dir.mkdir(parents=True, exist_ok=True)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            plot_metrics_per_step(
                all_step_metrics,
                all_decoded_texts,
                confidence_matrix,
                save_dir=str(acar_dir),
                heatmap_dir=str(heatmap_dir)
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
        ACAR_analyse: bool = False,
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
            ACAR_analyse=ACAR_analyse,
            sample_idx=sample_idx,  #新增，后期得删掉  
            query_position=query_position, # 新增参数
            tokenizer=self.tokenizer 
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

    # def generate_text_batch(
    #     self,
    #     prompts: List[str],
    #     answer_length: int = 1024,
    #     sampling_steps: int = 1024,
    #     block_length: int = 1024,
    #     temperature: float = 0.0,
    #     stop_tokens: Optional[List[str]] = None
    # ) -> List[str]:
    #     """
    #     批量生成文本 - 优化版本（伪batch，因为LLaDA的mask机制限制）
        
    #     Args:
    #         prompts: 输入prompt列表
    #         answer_length: 答案长度
    #         sampling_steps: 采样步数
    #         block_length: 块长度
    #         temperature: 采样温度
    #         stop_tokens: 停止token列表
            
    #     Returns:
    #         生成的文本列表
    #     """
    #     if not prompts:
    #         return []
        
    #     generated_texts = []
        
    #     # 优化：预分配列表大小
    #     generated_texts = [""] * len(prompts)
        
    #     # 优化：减少异常处理开销，将错误处理移到外层
    #     for i, prompt in enumerate(prompts):
    #         generated_texts[i] = self.generate_text(
    #             prompt=prompt,
    #             answer_length=answer_length,
    #             sampling_steps=sampling_steps,
    #             block_length=block_length,
    #             temperature=temperature,
    #             stop_tokens=stop_tokens
    #         )
        
    #     return generated_texts

    
    #ai生成的测试LLaDA的icl双向推理能力
    # def generate_for_icl_testing(
    #     self,
    #     prompt: str,
    #     answer_length: int = 1024,
    #     sampling_steps: int = 1024,
    #     block_length: int = 1024,
    #     remask_strategy: str = "low_confidence",
    #     temperature: float = 0.0,
    #     cfg_scale: float = 0.0,
    #     stop_tokens: Optional[List[str]] = None
    # ) -> str:
    #     """
    #     专门用于测试双向能力的in-context learning的生成函数
    #     使用非半自回归架构（block_length = answer_length）
        
    #     Args:
    #         prompt: 输入提示
    #         answer_length: 答案长度
    #         sampling_steps: 采样步数
    #         block_length: 块长度（应等于answer_length以实现非半自回归）
    #         remask_strategy: 重掩码策略（应使用"low_confidence"）
    #         temperature: 采样温度
    #         cfg_scale: 分类器自由引导缩放
    #         stop_tokens: 停止token列表
            
    #     Returns:
    #         生成的文本
    #     """
    #     # 确保使用非半自回归架构
    #     if block_length != answer_length:
    #         print(f"Warning: For ICL testing, block_length should equal answer_length. "
    #                 f"Setting block_length from {block_length} to {answer_length}")
    #         block_length = answer_length
        
    #     # 确保使用低置信度掩码
    #     if remask_strategy != "low_confidence":
    #         print(f"Warning: For ICL testing, remask_strategy should be 'low_confidence'. "
    #                 f"Using 'low_confidence' instead of '{remask_strategy}'")
    #         remask_strategy = "low_confidence"
        
    #     return self.generate_text(
    #         prompt=prompt,
    #         answer_length=answer_length,
    #         sampling_steps=sampling_steps,
    #         block_length=block_length,
    #         remask_strategy=remask_strategy,
    #         temperature=temperature,
    #         cfg_scale=cfg_scale,
    #         stop_tokens=stop_tokens
    #     )


    def batch_generate(
        self,
        prompts: List[str],
        answer_length: int = 1024,
        sampling_steps: int = 1024,
        block_length: int = 1024,
        remask_strategy: str = "low_confidence",
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        stop_tokens: Optional[List[str]] = None
    ) -> List[str]:   
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            answer_length: 答案长度
            sampling_steps: 采样步数
            block_length: 块长度
            remask_strategy: 重掩码策略
            temperature: 采样温度
            cfg_scale: 分类器自由引导缩放
            stop_tokens: 停止token列表
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            result = self.generate_text(
                prompt=prompt,
                answer_length=answer_length,
                sampling_steps=sampling_steps,
                block_length=block_length,
                remask_strategy=remask_strategy,
                temperature=temperature,
                cfg_scale=cfg_scale,
                stop_tokens=stop_tokens
            )
            results.append(result)
        return results


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

    for position in range(5):
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
                    ACAR_analyse=True,
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
        results_dir = Path(f"batch_analysis_results_{position}")
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
        print(f"- 详细结果: all_results.json")
        print(f"- 汇总统计: summary.json")
        print(f"- ACAR分析图: test_1/ ~ test_{N_SAMPLES}/ (各自的ACAR_analysis_output和heatmap目录)")
    
    print("\n=== 批量测试完成 ===")



    # # 4. 准备测试数据
    # print("\n步骤4: 准备测试数据...")
    # train_samples = [
    #     {
    #         "question": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.",
    #         "answer": """If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43"""
    #     },
    #     {
    #         "question": "Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?",
    #         "answer": """The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files.\nThe number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240\nWhen he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again.\nThe total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160\nTo write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research.\n#### 400"""
    #     },
    #     {
    #         "question": "Paityn has 20 red hats and 24 blue hats. Her friend Zola has 4/5 times as many red hats as she has and twice the number of blue hats. If they combine all the hats together and share them equally between themselves, calculate the number of hats each gets.",
    #         "answer": """Paityn has a total of 20 hats + 24 hats = <<20+24=44>>44 hats.\nThe number of red hats that Zola has is 4/5 * 20 hats = <<4/5*20=16>>16 hats\nZola also has 2 * 24 hats = <<2*24=48>>48 blue hats.\nZola has a total of 48 hats + 16 hats = <<48+16=64>>64 hats.\nWhen they combine their hats, they have 64 hats + 44 hats = <<64+44=108>>108 hats\nIf they share the hats equally, each get 108 hats / 2 people = <<108/2=54>>54 hats/person\n#### 54"""
    #     },
    #     {
    #         "question": "John works a job that offers performance bonuses. He makes $80 a day and works for 8 hours. He has the option of working hard to earn the performance bonus of an extra $20 a day, but the extra effort results in a 2-hour longer workday. How much does John make per hour if he decides to earn the bonus?",
    #         "answer": """First, we need to determine the length of John's workday if he decides to earn the bonus. We do this by performing 8+2= <<8+2=10>>10 hours for his workday.\nNext, we need to determine his overall pay. We do this by performing 80+20=<<80+20=100>>100 dollars a day.\nWe then determine John's hourly rate by dividing his pay by the number of hours worked, performing 100/10= <<100/10=10>>10 dollars an hour.\n#### 10"""
    #     },
    #     {
    #         "question": "Last year Jessica paid $1000 for rent, $200 for food, and $100 for car insurance each month. This year her rent goes up by 30%, food costs increase by 50%, and the cost of her car insurance triples because she was at fault in an accident. How much more does Jessica pay for her expenses over the whole year compared to last year?",
    #         "answer": "First find the increase in rent by multiplying last year's rent by 30%: $1000 * .3 = $<<1000*.3=300>>300\nThen find the food cost increase by multiplying last year's costs by 50%: $200 * .5 = $<<200*.5=100>>100\nThen find the new car insurance price by multiplying last year's price by 3: $100 * 3 = $<<100*3=300>>300\nThen subtract the cost of car insurance last year from this year's price to find the increase: $300 - $100 = $<<300-100=200>>200\nNow find how much Jessica's monthly expenses increased by adding the increases in each of the three costs: $300 + $100 + $200 = $<<300+100+200=600>>600\nNow multiply the increase per month by the number of months in a year to find the annual increase: $600/month * 12 months/year = $<<600*12=7200>>7200/year\n#### 7200"
    #     }

    # ]
    
    # test_sample = {
    #     "question": "Stephen placed an online order for groceries. His final bill came to $40.00. Because this was through a delivery vendor, they tacked on a 25% fee to his final total and charged him $3.00 in delivery fees. Stephen also added a $4.00 tip. After the extra fees, what was the final price of Stephen's groceries?",
    #     "answer": "He spent $40.00 on groceries but they charged him a 25% fee so that's 40*.25 = $10.00\nThere is also a $3.00 delivery fee and a $4.00 tip for an extra $3 + $4 = $<<3+4=7.00>>7.00 in fees\nHis groceries were $40.00, there's a $10.00 fee plus another $7.00 in delivery/tip fees for a final total of $40 + $10 + $7 = $<<40+10+7=57.00>>57.00\n#### 57"
    # }
    
    # 5. 构建prompt
    # print("\n步骤5: 构建prompt...")
    # prompt = prompt_constructor.construct_prompt(train_samples, test_sample, mask_length=256)
    # print(f"Prompt长度: {len(prompt)} 字符")
    # print("完整Prompt:")
    # print(prompt)
    
    # # 6. 生成答案
    # print("\n步骤6: 生成答案...")
    # try:
    #     generated_text = inference.generate_text(
    #         prompt=prompt,
    #         answer_length=256,
    #         sampling_steps=256,
    #         block_length=256, 
    #         temperature=0.0,
    #         stop_tokens=["Question:", "Answer:"],  # 设置停止条件，当遇到新的问题时停止
    #         output_attentions=True,  # 开启注意力捕获
    #         ACAR_analyse=True
    #     )
        
    #     print("生成完成")
    #     print(f"生成文本长度: {len(generated_text)} 字符")
        
    #     # 7. 打印生成的文本和真实答案
    #     print("\n步骤7: 调试信息...")
    #     print("=" * 50)
    #     print("生成的文本:")
    #     print(f"'{generated_text}'")
    #     print("=" * 50)
    #     print("真实答案:")
    #     print(f"'{test_sample['answer']}'")
    #     print("=" * 50)
        
    #     # 8. 提取和比较答案
    #     print("\n步骤8: 提取和比较答案...")
    #     predicted_answer = extract_gsm8k_answer(generated_text)
    #     true_answer = extract_gsm8k_answer(test_sample['answer'])
        
    #     print(f"提取的预测答案: {predicted_answer}")
    #     print(f"提取的真实答案: {true_answer}")
    #     print(f"答案正确: {predicted_answer == true_answer}")
        
    #     print("\n=== 串联测试完成 ===")
        
    # except Exception as e:
    #     print(f"生成过程中出现错误: {e}")
    #     print(f"错误类型: {type(e).__name__}")
    #     import traceback
    #     print("详细错误信息:")
    #     traceback.print_exc()
    #     print("请检查模型配置和参数设置")