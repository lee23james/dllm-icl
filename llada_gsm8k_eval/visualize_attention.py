import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

def visualize_attention_weights(input_dir: str, output_dir: str):
    """
    将存储为 .pt 文件的注意力权重可视化为 .png 图片。

    Args:
        input_dir (str): 包含 .pt 文件的输入目录路径。
        output_dir (str): 保存 .png 图片的输出目录路径。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录 '{output_path}' 已创建或已存在。")

    # 2. 获取所有 .pt 文件
    pt_files = sorted(list(input_path.glob("*.pt")))
    if not pt_files:
        print(f"在 '{input_path}' 中没有找到 .pt 文件。")
        return

    print(f"找到 {len(pt_files)} 个 .pt 文件，开始转换...")

    # 3. 遍历文件并进行可视化
    for pt_file in tqdm(pt_files, desc="Visualizing Attention Weights"):
        try:
            # 加载张量
            # 注意：原始张量保存为 float16，加载后转为 float32 以便绘图
            attn_tensor = torch.load(pt_file, map_location='cpu').to(torch.float32)

            # 创建图像
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_tensor.numpy(), cmap='viridis', interpolation='nearest')
            
            # 添加颜色条
            fig.colorbar(im, ax=ax)
            
            # 设置标题
            ax.set_title(pt_file.name, fontsize=12)
            
            # 移除坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])

            # 构建输出文件名
            output_filename = output_path / f"{pt_file.stem}.png"
            
            # 保存图像
            plt.savefig(output_filename, bbox_inches='tight', dpi=150)
            
            # 关闭图像以释放内存
            plt.close(fig)

        except Exception as e:
            print(f"处理文件 {pt_file} 时出错: {e}")

    print(f"可视化完成！所有图片已保存到 '{output_path}' 目录。")

if __name__ == "__main__":
    # 设置输入和输出目录
    # 脚本位于 llada_gsm8k_eval/，所以使用相对路径
    base_dir = Path(__file__).parent
    input_directory = base_dir / "temp_attention_weights"
    output_directory = base_dir / "attention_images"
    
    visualize_attention_weights(str(input_directory), str(output_directory))
