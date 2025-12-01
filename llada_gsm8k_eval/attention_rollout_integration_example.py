"""
示例：如何在 llada_inference.py 中集成 Attention Rollout

在你的 generate 函数中，可以这样添加：
"""

# 在文件顶部导入
from attention_rollout import (
    accumulate_attn_rollout,
    compute_layer_wise_rollout,
    extract_cls_token_attention,
    visualize_rollout
)


def example_integration_in_generate_function():
    """
    示例：在 generate 函数中集成 attention rollout
    
    在 llada_inference.py 的 generate 函数中，在获取注意力权重后添加：
    """
    
    # 假设你已经有了这些变量：
    # outputs.attentions: List of attention tensors from model
    # x: input token IDs
    # tokenizer: tokenizer object
    # save_root: path to save results
    
    # ========== 示例代码：添加到 generate 函数中 ==========
    
    # 在你的代码中，当 output_attentions=True 且获取到注意力权重后：
    
    if output_attentions and outputs.attentions is not None:
        # 1. 准备注意力权重列表（转换为统一格式）
        # outputs.attentions 的格式可能是: List[Tuple[Tensor]] 或 List[Tensor]
        # 每个元素形状: [batch, num_heads, seq_len, seq_len] 或 [num_heads, seq_len, seq_len]
        
        attention_list = []
        for attn in outputs.attentions:
            # 处理不同的格式
            if isinstance(attn, tuple):
                attn = attn[0]  # 取第一个元素（如果有多个）
            
            # 如果是 [batch, num_heads, seq_len, seq_len]，取第一个 batch
            if attn.dim() == 4:
                attn = attn[0]  # [num_heads, seq_len, seq_len]
            
            attention_list.append(attn)
        
        # 2. 计算完整的 attention rollout
        rollout = accumulate_attn_rollout(
            attention_list,
            head_average=True,    # 对多头进行平均
            add_residual=True,     # 添加残差连接
            normalize=True         # 归一化
        )
        
        # 3. 保存 rollout 矩阵
        rollout_path = save_root / f"step_{step_counter}_rollout.pt"
        torch.save(rollout.detach().cpu(), rollout_path)
        print(f"Attention rollout saved to: {rollout_path}")
        
        # 4. （可选）计算逐层 rollout
        layer_rollouts = compute_layer_wise_rollout(
            attention_list,
            head_average=True,
            add_residual=True,
            normalize=True
        )
        
        # 保存每一层的 rollout
        for layer_idx, layer_rollout in enumerate(layer_rollouts):
            layer_rollout_path = save_root / f"step_{step_counter}_layer_{layer_idx}_rollout.pt"
            torch.save(layer_rollout.detach().cpu(), layer_rollout_path)
        
        # 5. （可选）提取特定位置的注意力（例如 CLS token）
        if rollout.dim() == 2:  # [seq_len, seq_len]
            # 提取第一个位置（通常是 CLS）对所有 token 的注意力
            cls_attention = extract_cls_token_attention(rollout, cls_position=0)
            cls_attention_path = save_root / f"step_{step_counter}_cls_attention.pt"
            torch.save(cls_attention.detach().cpu(), cls_attention_path)
        
        # 6. （可选）可视化 rollout
        if tokenizer is not None:
            try:
                viz_path = save_root / f"step_{step_counter}_rollout_visualization.png"
                visualize_rollout(
                    rollout=rollout,
                    token_ids=x[0] if x.dim() > 1 else x,
                    tokenizer=tokenizer,
                    save_path=str(viz_path),
                    title=f"Attention Rollout - Step {step_counter}"
                )
                print(f"Rollout visualization saved to: {viz_path}")
            except Exception as e:
                print(f"Failed to visualize rollout: {e}")
        
        # 7. （可选）分析 rollout：找到最重要的输入 token
        # rollout[i, j] 表示输出位置 i 对输入位置 j 的累积注意力
        if rollout.dim() == 2:
            # 计算每个输入 token 的总重要性（所有输出位置对它关注的总和）
            token_importance = rollout.sum(dim=0)  # [seq_len]
            
            # 找到最重要的 token
            top_k_tokens = 10
            top_indices = torch.topk(token_importance, k=min(top_k_tokens, len(token_importance)))[1]
            
            if tokenizer is not None:
                try:
                    tokens = tokenizer.convert_ids_to_tokens(x[0] if x.dim() > 1 else x)
                    print(f"\nTop {top_k_tokens} most important tokens (by rollout):")
                    for idx in top_indices:
                        print(f"  Position {idx.item()}: {tokens[idx.item()]} (importance: {token_importance[idx].item():.4f})")
                except Exception as e:
                    print(f"Failed to decode tokens: {e}")


# ========== 在函数参数中添加 rollout 选项 ==========

"""
在你的 generate 函数签名中添加参数：

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
    output_attentions: bool = False,
    compute_rollout: bool = False,  # 新增：是否计算 rollout
    visualize_rollout: bool = False,  # 新增：是否可视化 rollout
    tokenizer: AutoTokenizer = None
) -> Union[torch.Tensor, tuple[torch.Tensor, List]]:
    ...
"""


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例：如何使用
    
    # 假设你有一个模型的输出，包含注意力权重
    # attentions = outputs.attentions  # List of tensors
    
    # 计算 rollout
    # rollout = accumulate_attn_rollout(attentions)
    
    # 可视化
    # visualize_rollout(rollout, token_ids=x, tokenizer=tokenizer, save_path="rollout.png")
    
    pass

