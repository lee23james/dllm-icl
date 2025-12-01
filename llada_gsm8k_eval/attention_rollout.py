"""
Attention Rollout 计算模块
参考：https://arxiv.org/abs/2005.00928

Attention Rollout 通过聚合所有层的注意力权重，并考虑残差连接，
来可视化模型在做出预测时关注哪些输入 token。
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any
import numpy as np


@torch.no_grad()
def accumulate_attn_rollout(
    attentions: List[torch.Tensor],
    head_average: bool = True,
    add_residual: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    计算 Attention Rollout
    
    Args:
        attentions: List of attention weights from each layer
                   Each tensor shape: [batch, num_heads, seq_len, seq_len]
                   or [num_heads, seq_len, seq_len] (single batch)
        head_average: 是否对多头注意力进行平均
        add_residual: 是否添加残差连接（单位矩阵）
        normalize: 是否归一化注意力矩阵
    
    Returns:
        rollout: [batch, seq_len, seq_len] or [seq_len, seq_len] (single batch)
                注意力 rollout 矩阵，表示从输入到输出的全局注意力流
    """
    if not attentions:
        raise ValueError("attentions list cannot be empty")
    
    # 过滤掉 None 值
    attentions = [attn for attn in attentions if attn is not None]
    if not attentions:
        raise ValueError("All attention weights are None")
    
    # 处理第一个 batch，假设所有层都有相同的 batch size
    first_attn = attentions[0]
    if first_attn.dim() == 3:
        # [num_heads, seq_len, seq_len] - 单 batch
        batch_size = 1
        single_batch = True
        seq_len = first_attn.shape[1]
    else:
        # [batch, num_heads, seq_len, seq_len]
        batch_size = first_attn.shape[0]
        single_batch = False
        seq_len = first_attn.shape[2]
    
    # 初始化 rollout 为单位矩阵（残差连接）
    if single_batch:
        rollout = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype)
    else:
        rollout = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 逐层处理注意力权重
    for layer_idx, attn in enumerate(attentions):
        if single_batch:
            # [num_heads, seq_len, seq_len]
            if head_average:
                # 对多头进行平均
                attn_matrix = attn.mean(dim=0)  # [seq_len, seq_len]
            else:
                # 使用第一个头
                attn_matrix = attn[0]
        else:
            # [batch, num_heads, seq_len, seq_len]
            if head_average:
                attn_matrix = attn.mean(dim=1)  # [batch, seq_len, seq_len]
            else:
                attn_matrix = attn[:, 0]  # [batch, seq_len, seq_len]
        
        # 添加残差连接（如果启用）
        if add_residual:
            if single_batch:
                attn_matrix = attn_matrix + torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype)
            else:
                identity = torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                attn_matrix = attn_matrix + identity
        
        # 归一化（确保每行和为1）
        if normalize:
            attn_matrix = attn_matrix / (attn_matrix.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 与上一层的 rollout 矩阵相乘
        rollout = torch.matmul(rollout, attn_matrix)
    
    return rollout


@torch.no_grad()
def compute_layer_wise_rollout(
    attentions: List[torch.Tensor],
    head_average: bool = True,
    add_residual: bool = True,
    normalize: bool = True
) -> List[torch.Tensor]:
    """
    计算逐层的 rollout（累积到每一层为止的 rollout）
    
    Args:
        attentions: List of attention weights from each layer
        head_average: 是否对多头注意力进行平均
        add_residual: 是否添加残差连接
        normalize: 是否归一化
    
    Returns:
        layer_rollouts: List of rollout matrices at each layer
    """
    layer_rollouts = []
    
    if not attentions:
        return layer_rollouts
    
    # 处理第一个 batch
    first_attn = attentions[0]
    if first_attn.dim() == 3:
        batch_size = 1
        single_batch = True
        seq_len = first_attn.shape[1]
    else:
        batch_size = first_attn.shape[0]
        single_batch = False
        seq_len = first_attn.shape[2]
    
    # 初始化
    if single_batch:
        rollout = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype)
    else:
        rollout = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    layer_rollouts.append(rollout.clone())
    
    # 逐层累积
    for layer_idx, attn in enumerate(attentions):
        if single_batch:
            if head_average:
                attn_matrix = attn.mean(dim=0)
            else:
                attn_matrix = attn[0]
        else:
            if head_average:
                attn_matrix = attn.mean(dim=1)
            else:
                attn_matrix = attn[:, 0]
        
        if add_residual:
            if single_batch:
                attn_matrix = attn_matrix + torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype)
            else:
                identity = torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                attn_matrix = attn_matrix + identity
        
        if normalize:
            attn_matrix = attn_matrix / (attn_matrix.sum(dim=-1, keepdim=True) + 1e-9)
        
        rollout = torch.matmul(rollout, attn_matrix)
        layer_rollouts.append(rollout.clone())
    
    return layer_rollouts


def extract_cls_token_attention(
    rollout: torch.Tensor,
    cls_position: int = 0
) -> torch.Tensor:
    """
    从 rollout 矩阵中提取 CLS token（或特定位置 token）对输入序列的注意力
    
    Args:
        rollout: [seq_len, seq_len] or [batch, seq_len, seq_len] - rollout 矩阵
        cls_position: CLS token 的位置（通常是 0）
    
    Returns:
        cls_attention: [seq_len] or [batch, seq_len] - CLS token 对每个 token 的注意力
    """
    if rollout.dim() == 2:
        return rollout[cls_position]
    else:
        return rollout[:, cls_position]


def visualize_rollout(
    rollout: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer: Optional[Any] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Rollout"
):
    """
    可视化 attention rollout 矩阵
    
    Args:
        rollout: [seq_len, seq_len] or [batch, seq_len, seq_len]
        token_ids: Token IDs for decoding
        tokenizer: Tokenizer for decoding tokens
        save_path: Path to save visualization
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    # 处理 batch 维度
    if rollout.dim() == 3:
        rollout = rollout[0]  # 取第一个 batch
    
    rollout_np = rollout.detach().cpu().numpy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(rollout_np, cmap='viridis', aspect='auto')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Input Token Position', fontsize=12)
    ax.set_ylabel('Output Token Position', fontsize=12)
    
    # 如果提供了 tokenizer，尝试添加 token 标签
    if tokenizer is not None and token_ids is not None:
        try:
            tokens = tokenizer.convert_ids_to_tokens(token_ids[0] if token_ids.dim() > 1 else token_ids)
            # 只显示部分 token 以避免拥挤
            step = max(1, len(tokens) // 20)
            tick_positions = list(range(0, len(tokens), step))
            tick_labels = [tokens[i] if i < len(tokens) else '' for i in tick_positions]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8)
        except Exception as e:
            print(f"Warning: Failed to decode tokens for visualization: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rollout visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

