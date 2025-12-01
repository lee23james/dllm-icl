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
from dataset.sudoku.sudoku_handler import SudokuHandler
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

import os
# 导入可视化函数

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from simple_attention_visualizer import (
    get_top_attention_tokens, 
    decode_tokens, 
    decode_full_sequence, 
    create_visualization
)

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
    visualize_attentions: bool = False,  # 新增参数：是否实时可视化
    top_k: int = 8,  # 新增参数：可视化时显示的top k tokens
    tokenizer: AutoTokenizer = None  # 新增参数：用于解码的tokenizer
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

# 初始化step_counter（如果任一功能开启）
    if output_attentions or visualize_attentions:
        step_counter = 0
    else:
        step_counter = None
    
    if output_attentions:
        save_root = Path(__file__).parent / "temp_attention_weights"
        save_root.mkdir(parents=True, exist_ok=True)
        
    # 创建可视化输出目录
    if visualize_attentions:
        viz_root = Path(__file__).parent / "attention_visualizations"
        viz_root.mkdir(parents=True, exist_ok=True)   


    x = prompt.clone().to(model.device)
    
    prompt_index = (x != mask_id)
    
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

            #在每一步推理前打印“输入给模型的prompt“
            # if tokenizer is not None:
            #     try:
            #         input_text = tokenizer.decode(x[0],skip_special_tokens=False)
            #         print(input_text)
            #         print("-"*80)
            #     except Exception as e:
            #         print(f"解码当前步输入给模型的prompt失败: {e}")
            
            # 分类器自由引导
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                outputs = model(x_, output_attentions=output_attentions)  # 修改这里
                logits = outputs.logits

                # CFG 分支里（紧接 outputs = model(x_, output_attentions=True); logits = outputs.logits 之后）
                if output_attentions and outputs.attentions is not None:
                    print(f"保存CFG分支 step_{step_counter}_input_ids.pt")
                    # 保存当前步的输入序列（CFG时保存条件分支x，因为注意力权重取的是第一个batch）
                    torch.save(x.clone(), save_root / f"step_{step_counter}_input_ids.pt")
                    torch.save(torch.tensor(x.shape[1]), save_root / f"step_{step_counter}_seq_len.pt")
                    
                    #保存全部注意力权重（每一步每一层每一头）
                    # for layer_idx, attn in enumerate(outputs.attentions):       # attn: [B, H, S, S]
                    #     attn_b = attn[0]                                        # 只取第一个 batch
                    #     n_heads = attn_b.shape[0]
                    #     for head_idx in range(n_heads):
                    #         attn_mat = attn_b[head_idx].detach().to('cpu', dtype=torch.float16)
                    #         torch.save(attn_mat, save_root / f"step_{step_counter}_layer_{layer_idx}_head_{head_idx}.pt")
                    #         del attn_mat
                    #     del attn_b
                    # step_counter += 1
                    # del outputs  # 及时释放
                    #保存最后一层的注意力权重（并且进行头平均）
                    #保存和可视化制定层（6，7，26，27）的头平均注意力
                    target_layers=[6,7,26,27]
                    for layer_idx in target_layers:
                        if layer_idx < len(outputs.attentions):
                            layer_attn_b = outputs.attentions[layer_idx][0] #[H,S,S]
                            mean_attn=layer_attn_b.mean(dim=0).detach().to('cpu',dtype=torch.float16)#[S,S]
                            torch.save(mean_attn,save_root/f"step_{step_counter}_layer_{layer_idx}_mean.pt")

                            #实时可视化
                            if visualize_attentions and tokenizer is not None:
                                try:
                                    #加载当前步的token ID 序列
                                    step_token_ids = x.clone()
                                    #生成可视化图片
                                    # output_file = viz_root/f"attention_step_{step_counter}_layer_{layer_idx}_mean_top{top_k}.png"
                                    # create_visualization(
                                    #     attention_weights=mean_attn.to(torch.float32),
                                    #     tokenizer=tokenizer,
                                    #     output_path=str(output_file),
                                    #     step=step_counter,
                                    #     layer=layer_idx,
                                    #     head="mean",
                                    #     top_k=top_k,
                                    #     original_token_ids=step_token_ids
                                    # )
                                    print(f"Step {step_counter} Layer {layer_idx} 可视化完成: {output_file}")
                                except Exception as e:
                                    print(f"Step {step_counter} Layer {layer_idx} 可视化失败: {e}")

                            del mean_attn
                            del layer_attn_b
                        
                    step_counter += 1


                    last_layer_attn_b = outputs.attentions[-1][0]  # [H, S, S]
                    mean_attn = last_layer_attn_b.mean(dim=0).detach().to('cpu', dtype=torch.float16)  # [S, S]
                    torch.save(mean_attn, save_root / f"step_{step_counter}_last_layer_mean.pt")
                        
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                outputs = model(x, output_attentions=output_attentions)  # 修改这里
                logits = outputs.logits

                # 普通分支里（紧接 outputs = model(x, output_attentions=True); logits = outputs.logits 之后）
                if output_attentions and outputs.attentions is not None:
                    print(f"保存普通分支 step_{step_counter}_input_ids.pt")
                    # 保存当前步的输入序列
                    torch.save(x.clone(), save_root / f"step_{step_counter}_input_ids.pt")
                    torch.save(torch.tensor(x.shape[1]), save_root / f"step_{step_counter}_seq_len.pt")
                    
                    #保存全部注意力权重（每一步每一层每一头）
                    # for layer_idx, attn in enumerate(outputs.attentions):
                    #     attn_b = attn[0]
                    #     n_heads = attn_b.shape[0]
                    #     for head_idx in range(n_heads):
                    #         attn_mat = attn_b[head_idx].detach().to('cpu', dtype=torch.float16)
                    #         torch.save(attn_mat, save_root / f"step_{step_counter}_layer_{layer_idx}_head_{head_idx}.pt")
                    #         del attn_mat
                    #     del attn_b
                    # step_counter += 1
                    # del outputs
                    # 仅保存最后一层注意力在所有头上的平均（只取第一个 batch）
                    #保存对应层的注意力权重（并且进行头平均）
                    # last_layer_attn_b = outputs.attentions[-1][0]  # [H, S, S]
                    # mean_attn = last_layer_attn_b.mean(dim=0).detach().to('cpu', dtype=torch.float16)  # [S, S]
                    # torch.save(mean_attn, save_root / f"step_{step_counter}_last_layer_mean.pt")
                    target_layers = [6, 7, 26, 27]
                    for layer_idx in target_layers:
                        if layer_idx < len(outputs.attentions):
                            layer_attn_b = outputs.attentions[layer_idx][0]  # [H, S, S]
                            mean_attn = layer_attn_b.mean(dim=0).detach().to('cpu', dtype=torch.float16)  # [S, S]
                            torch.save(mean_attn, save_root / f"step_{step_counter}_layer_{layer_idx}_mean.pt")

                            # 实时可视化
                            if visualize_attentions and tokenizer is not None:
                                try:
                                    # 加载当前步的token ID序列
                                    step_token_ids = x.clone()
                                    
                                    # 生成可视化图片
                                    output_file = viz_root / f"attention_step_{step_counter}_layer_{layer_idx}_mean_top{top_k}.png"
                                    
                                    create_visualization(
                                        attention_weights=mean_attn.to(torch.float32),
                                        tokenizer=tokenizer,
                                        output_path=str(output_file),
                                        step=step_counter,
                                        layer=layer_idx,
                                        head="mean",
                                        top_k=top_k,
                                        original_token_ids=step_token_ids
                                    )
                                    print(f"Step {step_counter} 可视化完成: {output_file}")
                                except Exception as e:
                                    print(f"Step {step_counter} 可视化失败: {e}")

                            del mean_attn
                            del layer_attn_b
                    step_counter += 1
            
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

            # current_generated_tokens = x[0][gen_start : gen_start + gen_length]
            # # 使用 skip_special_tokens=False 来看到 <|mdm_mask|>
            # current_text = tokenizer.decode(current_generated_tokens, skip_special_tokens=False)
            # # 直接打印，让终端自动处理换行
            # print(current_text)
            # print("-" * (len(f"--- [Block {num_block+1}, Step {i+1}/{steps}] ---"))) # 分隔线
    
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
        output_attentions: bool = False,  # 新增参数
        visualize_attentions: bool = False,  # 新增参数：是否实时可视化
        top_k: int = 8,  # 新增参数：可视化时显示的top k tokens
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
            output_attentions=output_attentions,  # 新增参数
            visualize_attentions=visualize_attentions,  # 新增参数
            top_k=top_k,  # 新增参数
            tokenizer=self.tokenizer  # 新增参数
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

    def generate_text_batch(
        self,
        prompts: List[str],
        answer_length: int = 1024,
        sampling_steps: int = 1024,
        block_length: int = 1024,
        temperature: float = 0.0,
        stop_tokens: Optional[List[str]] = None
    ) -> List[str]:
        """
        批量生成文本 - 优化版本（伪batch，因为LLaDA的mask机制限制）
        
        Args:
            prompts: 输入prompt列表
            answer_length: 答案长度
            sampling_steps: 采样步数
            block_length: 块长度
            temperature: 采样温度
            stop_tokens: 停止token列表
            
        Returns:
            生成的文本列表
        """
        if not prompts:
            return []
        
        generated_texts = []
        
        # 优化：预分配列表大小
        generated_texts = [""] * len(prompts)
        
        # 优化：减少异常处理开销，将错误处理移到外层
        for i, prompt in enumerate(prompts):
            generated_texts[i] = self.generate_text(
                prompt=prompt,
                answer_length=answer_length,
                sampling_steps=sampling_steps,
                block_length=block_length,
                temperature=temperature,
                stop_tokens=stop_tokens
            )
        
        return generated_texts

    
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
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from llada_loader import load_model
    from prompt_constructor_gsm8k import GSM8KPromptConstructor
    from utils import extract_gsm8k_answer
    
    print("=== LLaDA Loader + Inference 串联测试 ===")
    
    # 1. 使用llada_loader加载模型和分词器
    print("步骤1: 使用llada_loader加载模型...")
    model_path = "/home/share/model_weight/llada/LLaDA-8B-Base/"
    device = "cuda:0"
    
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
    
    # 3. 创建prompt构造器
    print("\n步骤3: 创建prompt构造器...")
    prompt_constructor = GSM8KPromptConstructor(n_shots=4, query_position=0)  # 使用默认位置
    
    # 4. 准备测试数据
    print("\n步骤4: 准备测试数据...")
    train_samples = [
        {
            "question": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.",
            "answer": """If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43"""
        },
        {
            "question": "Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?",
            "answer": """The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files.\nThe number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240\nWhen he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again.\nThe total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160\nTo write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research.\n#### 400"""
        },
        {
            "question": "Paityn has 20 red hats and 24 blue hats. Her friend Zola has 4/5 times as many red hats as she has and twice the number of blue hats. If they combine all the hats together and share them equally between themselves, calculate the number of hats each gets.",
            "answer": """Paityn has a total of 20 hats + 24 hats = <<20+24=44>>44 hats.\nThe number of red hats that Zola has is 4/5 * 20 hats = <<4/5*20=16>>16 hats\nZola also has 2 * 24 hats = <<2*24=48>>48 blue hats.\nZola has a total of 48 hats + 16 hats = <<48+16=64>>64 hats.\nWhen they combine their hats, they have 64 hats + 44 hats = <<64+44=108>>108 hats\nIf they share the hats equally, each get 108 hats / 2 people = <<108/2=54>>54 hats/person\n#### 54"""
        },
        {
            "question": "John works a job that offers performance bonuses. He makes $80 a day and works for 8 hours. He has the option of working hard to earn the performance bonus of an extra $20 a day, but the extra effort results in a 2-hour longer workday. How much does John make per hour if he decides to earn the bonus?",
            "answer": """First, we need to determine the length of John's workday if he decides to earn the bonus. We do this by performing 8+2= <<8+2=10>>10 hours for his workday.\nNext, we need to determine his overall pay. We do this by performing 80+20=<<80+20=100>>100 dollars a day.\nWe then determine John's hourly rate by dividing his pay by the number of hours worked, performing 100/10= <<100/10=10>>10 dollars an hour.\n#### 10"""
        },
        {
            "question": "Last year Jessica paid $1000 for rent, $200 for food, and $100 for car insurance each month. This year her rent goes up by 30%, food costs increase by 50%, and the cost of her car insurance triples because she was at fault in an accident. How much more does Jessica pay for her expenses over the whole year compared to last year?",
            "answer": "First find the increase in rent by multiplying last year's rent by 30%: $1000 * .3 = $<<1000*.3=300>>300\nThen find the food cost increase by multiplying last year's costs by 50%: $200 * .5 = $<<200*.5=100>>100\nThen find the new car insurance price by multiplying last year's price by 3: $100 * 3 = $<<100*3=300>>300\nThen subtract the cost of car insurance last year from this year's price to find the increase: $300 - $100 = $<<300-100=200>>200\nNow find how much Jessica's monthly expenses increased by adding the increases in each of the three costs: $300 + $100 + $200 = $<<300+100+200=600>>600\nNow multiply the increase per month by the number of months in a year to find the annual increase: $600/month * 12 months/year = $<<600*12=7200>>7200/year\n#### 7200"
        }

    ]
    
    test_sample = {
        "question": "Jackson is buying school supplies for his third grade class, which has 30 students. Each student needs 5 pens, 3 notebooks, 1 binder and 2 highlighters. Pens cost $0.50, notebooks cost $1.25, binders cost $4.25, and highlighters cost $0.75. If Jackson gets a $100 teacher discount, how much does he spend on school supplies?",
        "answer": "First calculate the cost of the pens for one student by multiplying the number of pens by the cost per pen: 5 pens * $0.50/pen = $<<5*0.5=2.50>>2.50\nThen calculate the cost of the notebooks for one student by multiplying the number of notebooks by the cost per notebook: 3 notebooks * $1.25/notebook = $<<3*1.25=3.75>>3.75\nThen calculate the cost of the highlighters for one student by multiplying the number of highlighters by the cost per highlighters: 2 highlighters * $0.75/highlighter = $<<2*0.75=1.50>>1.50\nNow add the cost of all the different supplies to find the total cost for one student: $2.50 + $3.75 + $1.50 + $4.25 = $<<2.5+3.75+1.5+4.25=12.00>>12.00\nNow multiply the cost per student by the number of students to find the total cost before the discount: 30 students * $12.00/student = $<<30*12=360.00>>360.00\nNow subtract the teacher discount to find the final cost: $360.00 - $100.00 = $<<360-100=260.00>>260.00\n#### 260"
    }
    
    # 5. 构建prompt
    print("\n步骤5: 构建prompt...")
    prompt = prompt_constructor.construct_prompt(train_samples, test_sample, mask_length=256)
    print(f"Prompt长度: {len(prompt)} 字符")
    print("完整Prompt:")
    print(prompt)
    
    # 6. 生成答案
    print("\n步骤6: 生成答案...")
    try:
        generated_text = inference.generate_text(
            prompt=prompt,
            answer_length=128,
            sampling_steps=128,
            block_length=128, 
            temperature=0.0,
            stop_tokens=["Question:", "Answer:"],  # 设置停止条件，当遇到新的问题时停止
            output_attentions=False, # 开启注意力捕获
            visualize_attentions=False,  # 开启实时可视化
            top_k=8  # 显示前8个最重要的token
        )
        
        print("生成完成")
        print(f"生成文本长度: {len(generated_text)} 字符")
        
        # 7. 打印生成的文本和真实答案
        print("\n步骤7: 调试信息...")
        print("=" * 50)
        print("生成的文本:")
        print(f"'{generated_text}'")
        print("=" * 50)
        print("真实答案:")
        print(f"'{test_sample['answer']}'")
        print("=" * 50)
        
        # # 8. 提取和比较答案
        # print("\n步骤8: 提取和比较答案...")
        # predicted_answer = extract_gsm8k_answer(generated_text)
        # true_answer = extract_gsm8k_answer(test_sample['answer'])
        
        # print(f"提取的预测答案: {predicted_answer}")
        # print(f"提取的真实答案: {true_answer}")
        # print(f"答案正确: {predicted_answer == true_answer}")
        
        # print("\n=== 串联测试完成 ===")
        
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        print("请检查模型配置和参数设置")