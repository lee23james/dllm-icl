#!/usr/bin/env python3
"""
简洁的Attention权重可视化工具
只需要tokenizer，不需要加载整个模型
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_tokenizer():
    """加载LLaDA-8B-Base tokenizer（使用Hugging Face）"""
    try:
        print("Loading LLaDA-8B-Base tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
        print("Hugging Face tokenizer loaded successfully!")
        return tokenizer
    except Exception as e:
        print(f"Error loading Hugging Face tokenizer: {e}")
        print("Will use token IDs directly without decoding")
        return None

def load_attention_weights(file_path):
    """加载attention权重文件"""
    try:
        print(f"Loading attention weights from {file_path}...")
        attention_weights = torch.load(file_path, map_location='cpu').to(torch.float32)
        print(f"Attention weights shape: {attention_weights.shape}")
        return attention_weights
    except Exception as e:
        print(f"Error loading attention weights: {e}")
        return None

def load_step_token_ids(attention_file_path):
    """加载当前步的token ID序列"""
    try:
        # 从attention文件路径推断当前步的token ID文件路径
        attention_dir = Path(attention_file_path).parent
        file_name = Path(attention_file_path).stem
        
        # 解析文件名获取step信息: step_{step}_last_layer_mean 
        parts = file_name.split('_')
        if len(parts) >= 2:
            step = parts[1]
            step_token_ids_file = attention_dir / f"step_{step}_input_ids.pt"
            
            if step_token_ids_file.exists():
                print(f"Loading step {step} token IDs from {step_token_ids_file}...")
                step_token_ids = torch.load(step_token_ids_file, map_location='cpu')
                print(f"Step {step} token IDs shape: {step_token_ids.shape}")
                return step_token_ids
            else:
                print(f"Step {step} token IDs file not found: {step_token_ids_file}")
                # 回退到原始方法
                return load_original_token_ids_fallback(attention_file_path)
        else:
            print(f"Cannot parse step from filename: {file_name}")
            return load_original_token_ids_fallback(attention_file_path)
    except Exception as e:
        print(f"Error loading step token IDs: {e}")
        return load_original_token_ids_fallback(attention_file_path)

def load_original_token_ids_fallback(attention_file_path):
    """回退方法：加载原始token ID序列"""
    try:
        attention_dir = Path(attention_file_path).parent
        token_ids_file = attention_dir / "original_token_ids.pt"
        
        if token_ids_file.exists():
            print(f"Loading fallback original token IDs from {token_ids_file}...")
            original_token_ids = torch.load(token_ids_file, map_location='cpu')
            print(f"Original token IDs shape: {original_token_ids.shape}")
            return original_token_ids
        else:
            print(f"No token IDs file found")
            return None
    except Exception as e:
        print(f"Error loading fallback token IDs: {e}")
        return None

def get_top_attention_tokens(attention_weights, top_k=4):
    """计算平均attention权重最大的前k个token"""
    # 计算每个token的平均attention权重（被其他token关注的程度）
    avg_attention = torch.mean(attention_weights, dim=0)  # 对列求平均
    
    # 获取前k个最大的attention权重
    top_values, top_indices = torch.topk(avg_attention, top_k)
    
    return top_indices.tolist(), top_values.tolist()

def decode_tokens(tokenizer, token_ids, original_token_ids=None):
    """解码token ID为可读文字"""
    if original_token_ids is None:
        # 如果没有原始token ID序列，只能显示位置信息
        return [f"Pos_{id}" for id in token_ids]
    
    if tokenizer is None:
        # 如果有原始token ID但没有tokenizer，显示token ID
        return [f"Token_{original_token_ids[0, id].item()}" for id in token_ids]
    
    try:
        decoded_tokens = []
        for pos_id in token_ids:
            try:
                # 获取该位置的实际token ID
                actual_token_id = original_token_ids[0, pos_id].item()
                # 解码实际的token ID
                token_text = tokenizer.decode([actual_token_id])
                # 处理特殊字符
                if token_text == ' ':
                    token_text = '␣'  # 用可见的空格符号
                elif token_text == '\n':
                    token_text = '↵'  # 换行符
                elif token_text == '\t':
                    token_text = '⇥'  # 制表符
                elif len(token_text) == 1 and ord(token_text) < 32:
                    # 控制字符 (ASCII 0-31)
                    control_names = {
                        0: 'NUL', 1: 'SOH', 2: 'STX', 3: 'ETX', 4: 'EOT', 5: 'ENQ', 6: 'ACK', 7: 'BEL',
                        8: 'BS', 9: 'TAB', 10: 'LF', 11: 'VT', 12: 'FF', 13: 'CR', 14: 'SO', 15: 'SI',
                        16: 'DLE', 17: 'DC1', 18: 'DC2', 19: 'DC3', 20: 'DC4', 21: 'NAK', 22: 'SYN', 23: 'ETB',
                        24: 'CAN', 25: 'EM', 26: 'SUB', 27: 'ESC', 28: 'FS', 29: 'GS', 30: 'RS', 31: 'US'
                    }
                    control_name = control_names.get(ord(token_text), f'CTRL{ord(token_text)}')
                    token_text = f'[{control_name}]'
                elif not token_text.strip():
                    token_text = f'[{actual_token_id}]'  # 不可见字符用token ID表示
                
                decoded_tokens.append(token_text)
            except Exception as e:
                decoded_tokens.append(f"[ERROR_{pos_id}]")
        return decoded_tokens
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        return [f"Pos_{id}" for id in token_ids]

def decode_full_sequence(tokenizer, seq_len, original_token_ids=None):
    """解码整个序列的token"""
    if original_token_ids is None:
        # 如果没有原始token ID序列，只能显示位置信息
        return [f"Pos_{i}" for i in range(seq_len)]
    
    if tokenizer is None:
        # 如果有原始token ID但没有tokenizer，显示token ID
        return [f"Token_{original_token_ids[0, i].item()}" for i in range(seq_len)]
    
    try:
        # 解码整个序列
        decoded_sequence = []
        for i in range(seq_len):
            try:
                # 获取该位置的实际token ID
                actual_token_id = original_token_ids[0, i].item()
                # 解码实际的token ID
                token_text = tokenizer.decode([actual_token_id])
                # 处理特殊字符
                if token_text == ' ':
                    token_text = '␣'  # 用可见的空格符号
                elif token_text == '\n':
                    token_text = '↵'  # 换行符
                elif token_text == '\t':
                    token_text = '⇥'  # 制表符
                elif len(token_text) == 1 and ord(token_text) < 32:
                    # 控制字符 (ASCII 0-31)
                    control_names = {
                        0: 'NUL', 1: 'SOH', 2: 'STX', 3: 'ETX', 4: 'EOT', 5: 'ENQ', 6: 'ACK', 7: 'BEL',
                        8: 'BS', 9: 'TAB', 10: 'LF', 11: 'VT', 12: 'FF', 13: 'CR', 14: 'SO', 15: 'SI',
                        16: 'DLE', 17: 'DC1', 18: 'DC2', 19: 'DC3', 20: 'DC4', 21: 'NAK', 22: 'SYN', 23: 'ETB',
                        24: 'CAN', 25: 'EM', 26: 'SUB', 27: 'ESC', 28: 'FS', 29: 'GS', 30: 'RS', 31: 'US'
                    }
                    control_name = control_names.get(ord(token_text), f'CTRL{ord(token_text)}')
                    token_text = f'[{control_name}]'
                elif not token_text.strip():
                    token_text = f'[{actual_token_id}]'  # 不可见字符用token ID表示
                
                decoded_sequence.append(token_text)
            except Exception as e:
                decoded_sequence.append(f"[ERROR_{i}]")
        return decoded_sequence
    except Exception as e:
        print(f"Error decoding full sequence: {e}")
        return [f"Pos_{i}" for i in range(seq_len)]

def create_visualization(attention_weights, tokenizer, output_path, step, layer, head, top_k=4, original_token_ids=None):
    """创建attention权重可视化图"""
    # 获取前k个attention权重最大的token
    top_indices, top_scores = get_top_attention_tokens(attention_weights, top_k)
    
    # 解码token
    decoded_tokens = decode_tokens(tokenizer, top_indices, original_token_ids)
    
    # 解码整个序列
    seq_len = attention_weights.shape[0]
    full_sequence = decode_full_sequence(tokenizer, seq_len, original_token_ids)
    
    # 创建图形 - 增加高度以容纳第三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 20))
    
    # 第一个子图：attention权重热力图
    im1 = ax1.imshow(attention_weights.numpy(), cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Attention Weights - Step {step}, Layer {layer}, Head {head}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position', fontsize=12)
    ax1.set_ylabel('Query Position', fontsize=12)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Attention Weight', fontsize=12)
    
    # 第二个子图：平均attention权重条形图
    avg_attention = torch.mean(attention_weights, dim=0).numpy()
    
    # 创建条形图
    bars = ax2.bar(range(seq_len), avg_attention, alpha=0.7, color='skyblue')
    
    # 突出显示前k个token
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        if idx < len(bars):
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.9)
            # 添加标签
            ax2.annotate(f'{i+1}', xy=(idx, score), xytext=(0, 5),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='red')
    
    ax2.set_title(f'Average Attention Weights (Top {top_k} tokens highlighted)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('Average Attention Weight', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.9, label=f'Top {top_k} tokens'),
        plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.7, label='Other tokens')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # 第三个子图：完整序列文本显示
    ax3.set_title('Full Token Sequence (Top tokens highlighted)', fontsize=14, fontweight='bold')
    
    # 创建完整序列的文本显示
    text_content = ""
    for i, token in enumerate(full_sequence):
        if i in top_indices:
            # 用中括号标注重要token
            text_content += f"[{token}]"
        else:
            text_content += token
    
    # 分多行显示，每行显示更多token以填满空间
    tokens_per_line = 100  # 增加每行的token数量
    lines = []
    for i in range(0, len(text_content), tokens_per_line):
        lines.append(text_content[i:i+tokens_per_line])
    
    # 计算需要显示的行数，调整字体大小
    max_lines = len(lines)  # 显示所有行，不截断
    # 如果行数太多，调整字体大小
    if len(lines) > 30:
        fontsize = 3  # 更小的字体
    elif len(lines) > 20:
        fontsize = 4
    else:
        fontsize = 5
    
    # 显示文本，使用等宽字体，横向填满
    ax3.text(0.01, 0.98, '\n'.join(lines), fontsize=fontsize, verticalalignment='top',
             horizontalalignment='left', transform=ax3.transAxes, 
             fontfamily='monospace', fontweight='normal',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # 添加token信息文本
    token_info = f"Top {top_k} tokens with highest average attention:\n"
    for i, (idx, token_text, score) in enumerate(zip(top_indices, decoded_tokens, top_scores)):
        token_info += f"{i+1}. Position {idx}: '{token_text}' (score: {score:.4f})\n"
    
    # 在图上添加文本信息
    fig.text(0.02, 0.02, token_info, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.95, hspace=0.3)  # 调整子图间距
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印结果
    print(f"\n可视化完成！图片已保存到: {output_path}")
    print(f"\nStep {step}, Layer {layer}, Head {head} 的Top {top_k}个token:")
    print("-" * 60)
    for i, (idx, token_text, score) in enumerate(zip(top_indices, decoded_tokens, top_scores)):
        print(f"{i+1}. 位置 {idx}: '{token_text}' (平均attention权重: {score:.4f})")
    print("-" * 60)
    
    # # 打印完整序列（带标注）
    # print(f"\n完整序列（重要token用中括号标注）:")
    # print("-" * 60)
    # sequence_text = ""
    # for i, token in enumerate(full_sequence):
    #     if i in top_indices:
    #         sequence_text += f"[{token}]"
    #     else:
    #         sequence_text += token
    
    # # 分多行显示，每行120个字符（增加显示长度）
    # tokens_per_line = 120
    # for i in range(0, len(sequence_text), tokens_per_line):
    #     print(sequence_text[i:i+tokens_per_line])
    # print("-" * 60)
    
    return top_indices, decoded_tokens, top_scores

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简洁的Attention权重可视化工具")
    parser.add_argument("--attention_file", type=str, required=True,
                       help="Attention权重.pt文件路径")
    parser.add_argument("--output_dir", type=str, default="./attention_visualizations",
                       help="输出目录")
    parser.add_argument("--top_k", type=int, default=8,
                       help="突出显示的token数量")
    
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = load_tokenizer()
    
    # 加载attention权重
    attention_weights = load_attention_weights(args.attention_file)
    if attention_weights is None:
        print("Failed to load attention weights!")
        return
    
    # 加载当前步的token ID序列
    original_token_ids = load_step_token_ids(args.attention_file)
    
    # 从文件名解析step信息: step_{step}_last_layer_mean
    file_name = Path(args.attention_file).stem
    try:
        parts = file_name.split('_')
        step = int(parts[1])
        # 对于last_layer_mean文件，layer和head固定为"last"和"mean"
        layer = "last"
        head = "mean"
    except (IndexError, ValueError):
        print("Warning: Could not parse step from filename, using defaults")
        step, layer, head = 0, "last", "mean"
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    output_file = output_dir / f"attention_step_{step}_last_layer_mean_top{args.top_k}.png"
    
    # 创建可视化
    create_visualization(
        attention_weights=attention_weights,
        tokenizer=tokenizer,
        output_path=str(output_file),
        step=step,
        layer=layer,
        head=head,
        top_k=args.top_k,
        original_token_ids=original_token_ids
    )

if __name__ == "__main__":
    main()