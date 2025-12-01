import os, json, re, csv, sys
from pathlib import Path
from typing import List, Optional, Tuple, Any
from transformers import AutoTokenizer
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
#默认进行头平均
@torch.no_grad()
def accumulate_attn_rollout(
    #每一个list是一层,每一个tensor是[S,S]
    attentions:List[torch.Tensor],
    # head_average:bool=True,
    add_residual:bool=True,
    normalize:bool=True
) -> torch.Tensor:
    """
    计算注意力矩阵的累积rollout
    
    Args:
        attentions: List of attention weights from each layer
    Returns:
        rollout: [batch, seq_len, seq_len] or [seq_len, seq_len] (single batch)
        注意力 rollout 矩阵，表示从输入到输出的全局注意力流
    """
    # if not attentions:
    #     raise ValueError("attentions list cannot be empty")

    #过滤掉None值
    #防止有一层出现none的情况
    # attentions = [attn for attn in attentions if attn is not None]
    # if len(attentions)!=32:
    #     raise ValueError("attention weights length is not 32")
    
    #list[tensor] ,这里取出的是第一层attention权重
    first_attn=attentions[0]
    #[B,H,S,S]
    if first_attn.dim()==2:
        #[h,s,s]单batch的情况
        # batch_size=1
        single_batch=True
        seq_len=first_attn.shape[1]
    else:
        raise ValueError("函数处理到仙子只有单batch的情况了,代码肯定有异常")
    
    #初始化rollout为单位矩阵(残差连接)
    if single_batch:
        #先创建一个单位矩阵用于后期的rollout的计算
        rollout=torch.eye(seq_len,device=first_attn.device,dtype=first_attn.dtype)
    else:
        raise ValueError("没适配多个batch的情况")
    
    #逐层处理注意力权重
    #开始处理每一层进行传递
    for _,attn_matrix in enumerate(attentions):
        # if single_batch:
        #     #[h,s,s]单batch的情况
        #     if head_average:
        #         attn_matrix=attn.mean(dim=0)#进行头平均
        #     else:
        #         attn_matrix=attn[0]
        # else:
        #     raise ValueError("没适配多个batch的情况")
        #添加残差连接
        if add_residual:
            if single_batch:
                attn_matrix=attn_matrix+torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype)
            else:
                raise ValueError("没适配多个batch的情况")
        
        #归一化(确保每行和为1)
        if normalize:
            attn_matrix = attn_matrix / (attn_matrix.sum(dim=-1,keepdim=True) + 1e-9)

        #与上一层的rollout矩阵进行右乘
        rollout=torch.matmul(rollout,attn_matrix)
    #返回的是[S.S]的tensor
    return rollout

#只能查看累积到当前层关注的是什么模式
@torch.no_grad()
def compute_layer_wise_rollout(
    attentions:List[torch.Tensor],
    head_average:bool = True,
    add_residual: bool = True,
    normalize: bool = True
)->List[torch.Tensor]:
    """
    计算每层注意力权重对最终输出的贡献
    """
    layer_rollouts=[]

    if not attentions:
        return layer_rollouts

    #处理第一层
    first_attn=attentions[0]
    if first_attn.dim()==3:
        batch=1
        single_batch=True
        seq_len=first_attn.shape[1]#取出序列长度数值
    else:
        raise ValueError("函数处理到只有单batch的情况了,代码肯定有异常")

    #初始化rollout为单位矩阵(残差连接)
    if single_batch:
        rollout=torch.eye(seq_len,device=first_attn.device,dtype=first_attn.dtype)
    else:
        raise ValueError("没适配多个batch的情况")
    
    #逐层处理注意力权重
    for _,attn in enumerate(attentions):
        if single_batch:
            #[h,s,s]单batch的情况
            if head_average:
                attn_matrix=attn.mean(dim=0)#进行头平均
            else:
                attn_matrix=attn[0]
        else:
            raise ValueError("没适配多个batch的情况")
        if add_residual:
            if single_batch:
                attn_matrix=attn_matrix+torch.eye(seq_len, device=attn_matrix.device, dtype=attn_matrix.dtype)
            else:
                raise ValueError("没适配多个batch的情况")
        if normalize:
            attn_matrix = attn_matrix / (attn_matrix.sum(dim=-1,keepdim=True) + 1e-9)
        #与上一层的rollout矩阵进行右乘
        rollout=torch.matmul(rollout,attn_matrix)
        #记录每一层的rollout的值
        layer_rollouts.append(rollout.clone())
    return layer_rollouts

#绘制rollout矩阵,展示关注哪个部分
def visualize_rollout(
    rollout:torch.Tensor,
    save_path:Optional[str]=None,
    title: str = "Attention Rollout"
):
    """
    可视化 attention rollout矩阵
    """
    import matplotlib.pyplot as plt

    #处理batch维度
    if rollout.dim()==3:
        raise ValueError("函数处理到只有单batch的情况了,代码肯定有异常")

    rollout_np=rollout.detach().cpu().numpy()
    #创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(rollout_np, cmap='viridis', aspect='auto')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('key token Position', fontsize=12)
    ax.set_ylabel('query token Position', fontsize=12)

    # #如果提供了tokenizer,尝试添加token标签
    # if tokenizer is not None and token_ids is not None:
    #     try:
    #         tokens=tokenizer.convert_ids_to_tokens(token_ids[0] if token_ids.dim() > 1 else token_ids)

    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
        print(f"Rollout visualzation saved to:{save_path}")
    
    else:
        plt.show()
    plt.close()

#绘制attention权重矩阵,展现哪个attention score占比更加大
#可以自主选择去可视化哪一层,这样子代码方便解耦
def visualize_attention(
    attention:torch.Tensor,#[S,S]的tensor
    save_path:Optional[str]=None,
    title: str = "Attention Weights",
    answer_token_positions:Tuple[int,int]=None,
    question_token_positions:Tuple[int,int]=None,
    example1_token_positions:Tuple[int,int]=None,
    example2_token_positions:Tuple[int,int]=None,
    example3_token_positions:Tuple[int,int]=None,
    example4_token_positions:Tuple[int,int]=None,
    example5_token_positions:Tuple[int,int]=None,
    system_token_positions:Tuple[int,int]=None,
):
    """
    可视化 attention权重矩阵
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    #attention一定要是二维的,即S,S
    #处理batch维度
    if attention.dim()>=3:
        raise ValueError("函数处理到只有单batch的情况了,代码肯定有异常")

    attention_np=attention.detach().cpu().numpy()
    seq_len=attention_np.shape[1]

    #收集所有非None的位置段
    segments=[]
    if system_token_positions is not None:
        segments.append(("S", system_token_positions, '#87CEEB'))  # 天蓝色
    if question_token_positions is not None:
        segments.append(("Q", question_token_positions, '#FFA500'))  # 橙色
    if example1_token_positions is not None:
        segments.append(("E1", example1_token_positions, '#90EE90'))  # 浅绿色
    if example2_token_positions is not None:
        segments.append(("E2", example2_token_positions, '#FFB6C1'))  # 浅粉色
    if example3_token_positions is not None:
        segments.append(("E3", example3_token_positions, '#DDA0DD'))  # 梅红色
    if example4_token_positions is not None:
        segments.append(("E4", example4_token_positions, '#F0E68C'))  # 卡其色
    if example5_token_positions is not None:
        segments.append(("E5", example5_token_positions, '#ADD8E6'))  # 浅蓝色
    if answer_token_positions is not None:
        segments.append(("A", answer_token_positions, '#FFD700'))  # 金色

    #创建图形布局:主图+底部分段条
    fig=plt.figure(figsize=(12,10))
    gs=GridSpec(2,1,height_ratios=[10,1],hspace=0.1)

    #主heatmap
    ax=fig.add_subplot(gs[0])
    im=ax.imshow(attention_np,cmap='viridis',aspect='auto')


    #添加颜色条
    plt.colorbar(im,ax=ax)

    #设置标题和标签
    ax.set_title(title,fontsize=14,fontweight='bold')
    ax.set_xlabel('key token Position',fontsize=12)
    ax.set_ylabel('query token Position',fontsize=12)

    # 底部分段条
    if segments:
        ax_seg = fig.add_subplot(gs[1])
        ax_seg.set_xlim(0, seq_len)
        ax_seg.set_ylim(0, 1)
        ax_seg.axis('off')  # 隐藏坐标轴
        
        # 绘制分段矩形和标签
        for label, (start, end), color in segments:
            # 绘制矩形
            rect = mpatches.Rectangle(
                (start, 0), 
                end - start + 1, 
                1, 
                facecolor=color, 
                edgecolor='black', 
                linewidth=0.5
            )
            ax_seg.add_patch(rect)
            
            # 添加标签（居中）
            mid_x = (start + end + 1) / 2
            ax_seg.text(
                mid_x, 0.5, label,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='black'
            )

        #添加图例(可选,显示在右侧)
        legend_elements=[
            mpatches.Patch(facecolor=color, edgecolor='black', label=label)
            for label, _, color in segments
        ]
        ax_seg.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    if save_path:
        #创建目录(如果不存在)
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
        print(f"Attention weights visualzation saved to:{save_path}")
    else:
        plt.show()
    plt.close()

#处理注意力权重(因为rollout再计算的时候已经全部处理好了,但是attention并没有全部处理)
# def process_attention_weights(
#     attentions:list[torch.Tensor]#注意力权重,每个元素是一个tensor,tensor的形状是[H,S,S]
# )->List[torch.Tensor]:#返回的是list[tensor],每个tensor的形状是[S,S]
#     #进行后验判断,先把他们变成可以计算转化的的东西,并且逐一验证是否满足需求
#     if not attentions:
#         raise ValueError("attentions list cannot be empty")  # pyright: ignore[reportUnreachable, reportUnreachable]

#     #过滤掉None值
#     #防止有一层出现none的情况
#     # attentions_list = [attn for attn in attentions if attn is not None]
#     attentions_list=[]
#     for attn in attentions:
#         if attn is not None:
#             #对每一层的注意力分数进行头平均,并且返回list[tensor]
#             attn=attn.mean(dim=0)
#             attentions_list.append(attn)
#     #进行后续的检查,进行硬编码
#     if len(attentions_list)!=32:
#         raise ValueError("attention weights length is not 32")
#     #现在要对每一个头进行头平均转化成[S,S]
#     return attentions_list

#找出所有满足条件的首个位置列表
def find_token_sequence_position(token_ids: List[int], target_ids: List[int]):
    # 如果输入是tensor，转换为list
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)
    
    target_len=len(target_ids)
    if target_len==0:
        return -1
    first_position=[]
    #滑动窗口进行查找
    for i in range(len(token_ids)-target_len+1):
        #检查从位置i开始的字列表是否完全匹配
        if token_ids[i:i+target_len]==target_ids:
            first_position.append(i)
    if first_position:
        return first_position#返回的是list[int],表示的是token的position
    else:
        return []

#返回的是匹配的所有最后位置的集合列表
def find_last_token_sequence_position(token_ids: List[int], target_ids: List[int]):
    # 如果输入是tensor，转换为list
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)
    
    target_len=len(target_ids)
    if target_len==0:
        return -1
    last_position=[]
    #滑动窗口进行查找
    for i in range(len(token_ids)-target_len+1):
        #检查从位置i开始的字列表是否完全匹配
        if token_ids[i:i+target_len]==target_ids:
            last_position.append(i+target_len-1)
    if last_position:
        return last_position#返回的是list[int],表示的是token的position
    else:
        return []

#输入进开头来进行计算
def get_token_position(
    token_ids:List[int],
    tokenizer:AutoTokenizer,
    mask_start_position:int,
    position:int
):#返回dict[str,tuple[int,int]]
    """
    获取token的元祖坐标
    """
    #结果为list[int],表示的是token的id
    start_ids = tokenizer.encode('Puzzle:', add_special_tokens=False)
    end_ids = tokenizer.encode('</answer>', add_special_tokens=False)
    start_token_positions_list=find_token_sequence_position(token_ids,start_ids)
    end_token_positions_list=find_last_token_sequence_position(token_ids,end_ids)
    example_token_positions=[]
    if start_token_positions_list and end_token_positions_list and len(start_token_positions_list)==6 and len(end_token_positions_list)==6:
        #这里采取硬编码,只适合5shot的情况,后期需要在这里补实验
        question_token_positions=(start_token_positions_list[5-position],mask_start_position-1)
        system_token_positions=(0,end_token_positions_list[0])
        del start_token_positions_list[5-position]
        del end_token_positions_list[0]
        for start_position,end_position in zip(start_token_positions_list,end_token_positions_list):
            #list[tuple[int,int]],表示的是token的元祖坐标
            example_token_positions.append((start_position,end_position))
        return{
            "question_token_positions": question_token_positions,
            "system_token_positions": system_token_positions,
            "example1_token_positions": example_token_positions[0],#tuple[int,int]
            "example2_token_positions": example_token_positions[1],#tuple[int,int]
            "example3_token_positions": example_token_positions[2],#tuple[int,int]
            "example4_token_positions": example_token_positions[3],#tuple[int,int]
            "example5_token_positions": example_token_positions[4],#tuple[int,int]
        }
    else:
        raise ValueError("token positions not found")

#关注特定的token关注的是哪些部分,这里我想把query token限定在answer token上,让它判断是哪里的最有效
#我要不要特化一下,就是专门适配sudoku的,标出所有位置,方便进行探究
#这个我想想魔改一下,变成专门看对于answertoken五个例子的贡献度
#我想要计算的是四个例子的占比
#所以我现在需要的是五个例子的元祖坐标还有answer token的元祖坐标来去比对贡献度
#适配
def extract_token_attention_or_rollout(
    #如果后期有需要计算层的我再考虑三种维度
    rollout:torch.Tensor,#其实也可以是rollout矩阵,并且已经放在了cpu上面,但是要确保是tensor
    answer_token_positions:Tuple[int,int],#answer token的元祖坐标,前面的int是开始,后面的int是结束
    question_token_positions:Tuple[int,int],#question token的元祖坐标,前面的int是开始,后面的int是结束
    example1_token_positions:Tuple[int,int],#example token的元祖坐标,每个元祖是一个例子,前面的int是开始,后面的int是结束,都是idx
    example2_token_positions:Tuple[int,int],#example token的元祖坐标,每个元祖是一个例子,前面的int是开始,后面的int是结束,都是idx
    example3_token_positions:Tuple[int,int],#example token的元祖坐标,每个元祖是一个例子,前面的int是开始,后面的int是结束,都是idx
    example4_token_positions:Tuple[int,int],#example token的元祖坐标,每个元祖是一个例子,前面的int是开始,后面的int是结束,都是idx
    example5_token_positions:Tuple[int,int],#example token的元祖坐标,每个元祖是一个例子,前面的int是开始,后面的int是结束,都是idx
    top_3_record:List[int],#list是前3步关注的解码位置数,其实就是需要去关注到东西
)->Tuple[dict[str,torch.Tensor],dict[str,float]]:
    """
    从rollout矩阵中提取特定位置的token来计算输入序列的注意力
    """
    #一定要要确保输入的rollout是二维的,占比为[S,S]
    if rollout.dim() == 2:
        #因为其实已经行归一化了,其实我只用去记录分数就好了,因为最后相加都是1
        #然后不管是attention还是rollout都已经行归一化了,所以不需要再进行归一化
        # 选择答案区间对应的行（闭区间）
        answer_token_attention = rollout[answer_token_positions[0]:answer_token_positions[1]+1, :]
        M,S=answer_token_attention.shape
        device=answer_token_attention.device
        dtype=answer_token_attention.dtype
        cols = torch.arange(S, device=device)
        # attention_sum_dict=defaultdict(torch.tensor)
        #这里是对第二个维度的制定列开始结束进行相加
        def seg_sum(start,end):
            mask=(cols>=start)&(cols<=end)#闭区间
            if mask.any():
                return answer_token_attention[:,mask].sum(dim=-1)
            else:
                return torch.zeros(M,device=device,dtype=dtype)

        query_attention=answer_token_attention.sum(dim=-1)
        
        # 筛选最近3步解码的答案token (top_3_record已经是相对索引)
        if top_3_record:
            # 过滤有效的索引（确保在范围内）
            valid_indices = [idx for idx in top_3_record if 0 <= idx < M]
            if valid_indices:
                top3_query_attention = query_attention[valid_indices].sum()  # 最近3步的总注意力
            else:
                # top3_query_attention = query_attention.sum()  # 回退到全部
                # valid_indices = list(range(M))
                raise ValueError("top_3_record is out of range or empty")
        else:
            raise ValueError("top_3_record is none")
            # top3_query_attention = query_attention.sum()  # 回退到全部
            # valid_indices = list(range(M))
    
        #取第二维的索引,也就是key token的索引
        #这里相当于进行了归一化,因为要涉及到关注的部分
        #这里对token级别的贡献度进行了计算
        #全部转化为list,方便进行存储,后期进行画图
        all_attention=query_attention.sum()#全部的注意力
        #细粒度压缩到token级别(这里放到了前3个)
        # 注意：只关注top_3_record指定的答案token
        attention_dict={
            "answer_token":  (seg_sum(answer_token_positions[0],  answer_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "example1_token": (seg_sum(example1_token_positions[0], example1_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "example2_token": (seg_sum(example2_token_positions[0], example2_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "example3_token": (seg_sum(example3_token_positions[0], example3_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "example4_token": (seg_sum(example4_token_positions[0], example4_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "example5_token": (seg_sum(example5_token_positions[0], example5_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "question_token": (seg_sum(question_token_positions[0], question_token_positions[1])[valid_indices].sum()/(top3_query_attention+1e-9)).item(),
            "other_token": ((top3_query_attention-seg_sum(question_token_positions[0], question_token_positions[1])[valid_indices].sum()
            -seg_sum(answer_token_positions[0],  answer_token_positions[1])[valid_indices].sum()
            -seg_sum(example1_token_positions[0], example1_token_positions[1])[valid_indices].sum()
            -seg_sum(example2_token_positions[0], example2_token_positions[1])[valid_indices].sum()
            -seg_sum(example3_token_positions[0], example3_token_positions[1])[valid_indices].sum()
            -seg_sum(example4_token_positions[0], example4_token_positions[1])[valid_indices].sum()
            -seg_sum(example5_token_positions[0], example5_token_positions[1])[valid_indices].sum())/(top3_query_attention+1e-9)).item(),
        }
        #这里是对总体的计算,也就是对区域内的注意力计算
        attention_sum_dict={
            "answer_token": (seg_sum(answer_token_positions[0],  answer_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "question_token": (seg_sum(question_token_positions[0], question_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "example1_token": (seg_sum(example1_token_positions[0], example1_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "example2_token": (seg_sum(example2_token_positions[0], example2_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "example3_token": (seg_sum(example3_token_positions[0], example3_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "example4_token": (seg_sum(example4_token_positions[0], example4_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "example5_token": (seg_sum(example5_token_positions[0], example5_token_positions[1]).sum()/(all_attention+1e-9)).item(),
            "other_token": ((all_attention-seg_sum(question_token_positions[0], question_token_positions[1]).sum()
            -seg_sum(answer_token_positions[0],  answer_token_positions[1]).sum()
            -seg_sum(example1_token_positions[0], example1_token_positions[1]).sum()
            -seg_sum(example2_token_positions[0], example2_token_positions[1]).sum()
            -seg_sum(example3_token_positions[0], example3_token_positions[1]).sum()
            -seg_sum(example4_token_positions[0], example4_token_positions[1]).sum()
            -seg_sum(example5_token_positions[0], example5_token_positions[1]).sum())/(all_attention+1e-9)).item(),
        }
        return attention_dict,attention_sum_dict

#这里我想要的是给定一个问题,根据问题的tokenid,去找对应的0的位置来看答案是怎么变化的
#这里要分成两个区域一个是answer区域,还有一个是0对应的区域
#返程4种吧,一种是answer区域,一种是换行和空格区域,一种是照搬上答案区域,还有一种就是0对应的位置区域
#这四种应该是意义对应的关系,但是后必须在左边变出来
#在第一个<answer>和最后一个</answer>之间都是答案区域,我先记录在这个范围内,然后我先从原来的问题中提取0是在第几个位置然后就是搜集数字
#我可以先计算这几个都熵值平均值和置信度的平均值并进行比较,来计算平均的熵值,看看是否有什么变化
#针对的是sudoku的特定查找范式
def find_answer_token_positions(
    answer_id,#回答的prompt的token_ids，可以是list[int]或torch.Tensor
    # query_position,
    tokenizer:AutoTokenizer,
    space_id:int=220,
    enter_id:int=198,
    # zero_position_list:List[int]=None,
    ):
    # 如果输入是tensor，转换为list
    if isinstance(answer_id, torch.Tensor):
        answer_id = answer_id.cpu().tolist()
    elif not isinstance(answer_id, list):
        # 如果是其他类型（如numpy array），也转换为list
        answer_id = list(answer_id)
    
    start_ids = tokenizer.encode('<answer>', add_special_tokens=False)
    end_ids = tokenizer.encode('</answer>', add_special_tokens=False)
    #先找到模型回答答案的开始和结尾(要找第一个)
    
    end_token_positions_list=find_token_sequence_position(answer_id,end_ids)#找到第一个</answer>的第一个匹配位置
    start_token_positions_list=find_last_token_sequence_position(answer_id,start_ids)#找到第一个<answer>的最后一个匹配位置
    
    #进行健壮性检查,如果发现找不到,那么则返回0个字段(相当于跳过该样本)
    # (新增) 检查是否找到了标签
    if not start_token_positions_list or not end_token_positions_list:
        print(f"Warning: 在 token 序列中未能找到 <answer> 或 </answer> 标签。")
        # 按照您的要求，在找不到时，将所有字段设为 None
        return {
            # "answer_start": [], "answer_end": [],
            # "content_start": [], "content_end": [],
            # "space_enter_positions": [], "digit_positions": [],
            # "zero_mapped_positions": [], "other_digit_positions": [],
            # "answer_tag_positions": [],
            "answer_token_positions": [],
            "other_positions": [],
        }
    
    
    start_position=start_token_positions_list[0]+1
    end_position=end_token_positions_list[0]-1#开始和结束都是闭区间
    #在这个区间内搜索数字空格的情况,先排除空格,再从数字里面取出原来就有的东西
    space_enter_position_list=[]
    digit_position_list=[]
    # answerid_clone=answer_id.copy()
    
    # 判断是否为数字的辅助函数
    def is_digit_token(token_id):
        """判断token是否为数字0-9"""
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            return decoded.strip() in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        except:
            return False
    
    # 遍历答案内容区域，分类所有token
    #相当于在答案区域收集了所有答案和其他数字还有空格标点符号的位置
    for i in range(start_position,end_position+1):
        if answer_id[i]==space_id:
            space_enter_position_list.append(i)
        elif answer_id[i]==enter_id:
            space_enter_position_list.append(i)
        elif is_digit_token(answer_id[i]):
            # 收集所有数字的位置（按顺序）
            digit_position_list.append(i)
    
    # 建立映射：原问题中0的位置 -> 答案区域中对应数字的位置
    # zero_mapped_positions = []
    # other_digit_positions = []
    
    # if zero_position_list is not None and len(digit_position_list) > 0:
    #     # 原问题中0的位置索引（在纯数字序列中）
    #     zero_indices_in_original = set(zero_position_list)
        
    #     # 答案区域中纯数字序列的索引（排除空格换行后的顺序）,digit记录的是所有数字的顺序
    #     for idx, digit_pos in enumerate(digit_position_list):
    #         #找到原来的0对应的位置,然后开始记录
    #         if idx in zero_indices_in_original:#然后zero_list记录的是原始的0在数字中的相对位置
    #             # 这是原问题中0位置对应的数字
    #             zero_mapped_positions.append(digit_pos)
    #         else:
    #             # 这是其他数字
    #             other_digit_positions.append(digit_pos)#其他数字也要记录
    # else:
    #     # 如果没有提供zero_position_list，所有数字都算other
    #     if zero_position_list is None:
    #         print("[find_answer_token_positions] zero_position_list is None. "
    #               "Make sure find_zero_position() returned a valid list.")
    #     if len(digit_position_list) == 0:
    #         print("[find_answer_token_positions] digit_position_list is empty. "
    #               "Decoded answer segment:", tokenizer.decode(answer_id, skip_special_tokens=False))
    #     # 不中断流程，让上层判断 None 后自行跳过
    #     return None
    
    # # 收集<answer>和</answer>标签的所有token位置
    # answer_tag_positions = []
    # # <answer>标签的所有token位置
    # for i in range(start_token_positions_list[0] - len(start_ids) + 1, start_token_positions_list[0] + 1):
    #     answer_tag_positions.append(i)
    # # </answer>标签的所有token位置
    # for i in range(end_token_positions_list[0], end_token_positions_list[0] + len(end_ids)):
    #     answer_tag_positions.append(i)
    
    # 计算answer_tag_positions的闭区间边界
    answer_start = start_token_positions_list[0] - len(start_ids) + 1
    answer_end = end_token_positions_list[0] + len(end_ids) - 1
    
    # 计算other_positions：除了answer_tag_positions闭区间之外的所有位置
    other_positions = []
    for i in range(len(answer_id)):
        if i < answer_start or i > answer_end:
            other_positions.append(i)
    
    # 返回所有位置信息
    #统一一下所有字段,只关注想关注的部分
    return {
        # "answer_start": answer_start,  # <answer>标签的起始位置
        # "answer_end": answer_end,  # </answer>标签的结束位置
        # "content_start": start_position,  # 答案内容的起始位置（排除标签）
        # "content_end": end_position,  # 答案内容的结束位置（排除标签，闭区间）
        # "space_enter_positions": space_enter_position_list,  # 空格和换行符的位置
        # "digit_positions": digit_position_list,  # 所有数字的位置（按顺序，排除空格换行）
        # "zero_mapped_positions": zero_mapped_positions,  # 原问题中0位置对应的答案区域位置
        # "other_digit_positions": other_digit_positions,  # 其他数字的位置
        # "answer_tag_positions": answer_tag_positions,  # <answer>和</answer>标签的所有token位置
        "other_positions": other_positions,  # 除了answer_tag_positions闭区间之外的所有位置
        "answer_token_positions": digit_position_list,  # 统一字段接口
    }
    
#给出sudoku上面数字中0的位置,取的是input['Puzzle']中的0的位置
def find_zero_position(
    prompt:str,
) -> List[int]:
    """
    找到字符串中所有0的位置索引
    
    Args:
        prompt: 输入字符串，例如 "2100031004211000"
    
    Returns:
        所有0的位置索引列表，例如 [2, 3, 4, 7, 8, 13, 14, 15]
    
    Example:
        >>> find_zero_position("2100031004211000")
        [2, 3, 4, 7, 8, 13, 14, 15]
    """
    zero_positions = []
    for i, char in enumerate(prompt):
        if char == '0':
            zero_positions.append(i)
    return zero_positions

#现在需要推广到countdown的任务范式
def find_countdown_answer_token_positions(
    answer_id,#回答的prompt的token_ids，可以是list[int]或torch.Tensor,目前这里就是包含模型的输出
    tokenizer:AutoTokenizer,
    space_id:int=220,
    enter_id:int=198,
):
    """
    找到countdown的答案区域的位置
    """
    # 如果输入是tensor，转换为list
    if isinstance(answer_id, torch.Tensor):
        answer_id = answer_id.cpu().tolist()
    elif not isinstance(answer_id, list):
        # 如果是其他类型（如numpy array），也转换为list
        answer_id = list(answer_id)
    
    start_ids = tokenizer.encode('<answer>', add_special_tokens=False)
    end_ids = tokenizer.encode('</answer>', add_special_tokens=False)

    end_token_positions_list=find_token_sequence_position(answer_id,end_ids)#找到第一个</answer>的第一个匹配位置
    start_token_positions_list=find_last_token_sequence_position(answer_id,start_ids)#找到第一个<answer>的最后一个匹配位置

    #进行健壮性检查,如果发现找不到,那么则返回0个字段(相当于跳过该样本)
    # (新增) 检查是否找到了标签
    if not start_token_positions_list or not end_token_positions_list:
        print(f"Warning: 在 token 序列中未能找到 <answer> 或 </answer> 标签。")
        # 按照您的要求，在找不到时，将所有字段设为空列表（包括 symbol_positions）
        return {
            # "answer_start": [], "answer_end": [],
            # "content_start": [], "content_end": [],
            # "space_enter_positions": [], "digit_positions": [],
            # "symbol_positions": [],  # 添加缺失的 symbol_positions
            # "answer_tag_positions": [],
            "other_positions": [],
            "answer_token_positions": [],
        }
    
    start_position=start_token_positions_list[0]+1
    end_position=end_token_positions_list[0]-1#开始和结束都是闭区间
    #在这个区间内搜索数字空格的情况,先排除空格,再从数字里面取出原来就有的东西
    space_enter_position_list=[]
    # digit_position_list=[]#题干部分
    # symbol_position_list=[]#符号部分,这个是真正的问题对应的数字
    answer_token_position_list=[]#按原始顺序收集数字和符号的位置

    # 判断是否为数字的辅助函数
    # def is_digit_token(token_id):
    #     """判断token是否为数字0-9"""
    #     try:
    #         decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    #         return decoded.strip() in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #     except:
    #         return False
    
    #回答只是记录三种区域
    # 遍历答案内容区域，分类所有token
    for i in range(start_position,end_position+1):
        if answer_id[i]==space_id:
            space_enter_position_list.append(i)
        elif answer_id[i]==enter_id:
            space_enter_position_list.append(i)
        # elif is_digit_token(answer_id[i]):
        #     digit_position_list.append(i)
        #     answer_token_position_list.append(i)  # 按顺序收集数字位置
        else:
            # symbol_position_list.append(i)
            answer_token_position_list.append(i)  # 按顺序收集符号位置

    # 收集<answer>和</answer>标签的所有token位置
    # answer_tag_positions = []
    # <answer>标签的所有token位置
    # for i in range(start_token_positions_list[0] - len(start_ids)+1,start_token_positions_list[0]+1):
    #     answer_tag_positions.append(i)
    # # </answer>标签的所有token位置
    # for i in range(end_token_positions_list[0],end_token_positions_list[0]+len(end_ids)):
    #     answer_tag_positions.append(i)

    # 计算answer_tag_positions的闭区间边界
    answer_start = start_token_positions_list[0] - len(start_ids) + 1
    answer_end = end_token_positions_list[0] + len(end_ids) - 1

    # 计算other_positions：除了answer_tag_positions闭区间之外的所有位置
    other_positions = []
    for i in range(len(answer_id)):
        if i < answer_start or i > answer_end:
            other_positions.append(i)

    # 返回所有位置信息
    return {
        # "answer_start": answer_start,  # <answer>标签的起始位置
        # "answer_end": answer_end,  # </answer>标签的结束位置
        # "content_start": start_position,  # 答案内容的起始位置（排除标签）
        # "content_end": end_position,  # 答案内容的结束位置（排除标签，闭区间）
        # "space_enter_positions": space_enter_position_list,  # 空格和换行符的位置
        # "digit_positions": digit_position_list,  # 所有数字的位置（按顺序，排除空格换行）
        # "symbol_positions": symbol_position_list,  # 符号的位置
        # "answer_tag_positions": answer_tag_positions,  # <answer>和</answer>标签的所有token位置
        "answer_token_positions": answer_token_position_list,  # 所有数字和符号的位置（按原始顺序，排除空格换行）
        "other_positions": other_positions,  # 除了answer_tag_positions闭区间之外的所有位置
    }

#这里专门适配countdown,去计算想看到的量(这里直接看平均即可),发现差别很大就用case by case的去看
#这次主要关注的是第一个answer之前的所有box的输出,做到token级别的
#这个我想做一个通用的,成熟之后,我们只用关注answer token即可
def find_math_answer_token_positions(
    answer_id,#回答的prompt的token_ids，可以是list[int]或torch.Tensor,目前这里就是包含模型的输出
    tokenizer:AutoTokenizer,
    space_id:int=220,
    enter_id:int=198,
):
    """
    找到math的答案区域的位置
    """
    # 如果输入是tensor 转化为list
    if isinstance(answer_id, torch.Tensor):
        answer_id = answer_id.cpu().tolist()
    elif not isinstance(answer_id, list):
        # 如果是其他类型（如numpy array），也转换为list
        answer_id = list(answer_id)
    
    #找到</answer>位置,然后从这个位置之前开始找boxed的内容,里面的内容即为答案的内容
    #这个是虚假的标记,需要从开始去找
    start_ids = tokenizer.encode('<answer>', add_special_tokens=False)
    end_ids = tokenizer.encode('</answer>', add_special_tokens=False)

    # boxed_ids = tokenizer.encode(r'\boxed', add_special_tokens=False)
    # boxed_ids_with_curly = tokenizer.encode(r'\boxed{', add_special_tokens=False)
    # curly_end = tokenizer.encode(r'}', add_special_tokens=False)
    # curly_start = tokenizer.encode(r'{', add_special_tokens=False)

    end_token_positions_list=find_token_sequence_position(answer_id,end_ids)#找到第一个</answer>的第一个匹配位置
    start_token_positions_list=find_last_token_sequence_position(answer_id,start_ids)#找到第一个<answer>的最后一个匹配位置

    #进行健壮性检查,万一答案区域太短,根本来不及回答,只能进行健壮性检查
    #如果找不到,因为我字数太短了,也就50个,只能舍弃(没事,后面会开1024的)
    if not start_token_positions_list or not end_token_positions_list:
        # print(f"Warning: 在 token 序列中未能找到 <answer> 或 </answer> 标签。")
        return {
            #我这下之区分需要去关注的位置和其他位置,我还是把boxed考虑进来吧
            "answer_token_positions": [],#这个包含的是/boxed(主要是太难分开了)
            "other_token_positions": [],#这个包含space_enter和其他东西,还包含<answer>和</answer>tag
            # "space_enter_positions": [],
            # "boxed_answer_positions":[],
        }
    start_position=start_token_positions_list[0]+1
    end_position=end_token_positions_list[0]-1

    # 从前向后找到第一个非空白/非回车的 token 作为答案起点
    answer_start_idx = None
    for i in range(start_position, end_position + 1):
        if answer_id[i] != space_id and answer_id[i] != enter_id:
            answer_start_idx = i
            break

    # 如果整段都是空白，直接返回
    if answer_start_idx is None:
        return {
            "answer_token_positions": [],
            "other_token_positions": list(range(len(answer_id)))
        }

    # 从后向前找到最后一个非空白/非回车的 token 作为答案终点
    answer_end_idx = None
    for i in range(end_position, answer_start_idx - 1, -1):
        if answer_id[i] != space_id and answer_id[i] != enter_id:
            answer_end_idx = i
            break

    if answer_end_idx is None:
        return {
            "answer_token_positions": [],
            "other_token_positions": list(range(len(answer_id)))
        }
    # 判断是否满足起点《终点的情况
    if answer_start_idx<=answer_end_idx:
    # 记录答案区间（保留内部的空白和换行）
        answer_token_positions = list(range(answer_start_idx, answer_end_idx + 1))
    else:
        raise ValueError("answer_start_idx is greater than answer_end_idx")
    
    # 找到 <think></think> 之间的内容
    think_start_ids = tokenizer.encode('<think>', add_special_tokens=False)
    think_end_ids = tokenizer.encode('</think>', add_special_tokens=False)
    
    think_start_positions_list = find_last_token_sequence_position(answer_id, think_start_ids)
    think_end_positions_list = find_token_sequence_position(answer_id, think_end_ids)
    
    if think_start_positions_list and think_end_positions_list:
        # 找到 <think> 之后和 </think> 之前的位置（不包括两边）
        think_start_position = think_start_positions_list[0] + 1
        think_end_position = think_end_positions_list[0] - 1
        
        # 边界检查
        if think_start_position <= think_end_position and think_start_position >= 0 and think_end_position < len(answer_id):
            # 将 <think></think> 之间的所有token位置加入到answer_token_positions
            answer_token_positions.extend(range(think_start_position, think_end_position + 1))

    # 计算 other_token_positions：除去答案 token 的所有位置
    answer_token_set = set(answer_token_positions)
    other_token_positions = [idx for idx in range(len(answer_id)) if idx not in answer_token_set]
    
    return {
        "answer_token_positions": sorted(answer_token_positions),
        "other_token_positions": other_token_positions,
    }

#后验来看答案token的生成位置
# def find_gsm8k_answer_token_positions
def find_gsm8k_answer_token_positions(
    answer_id,
    tokenizer:AutoTokenizer,
    space_id:int=220,
    enter_id:int=198,
):
    """
    为gsm8k找到应该去探寻的位置,其实就是the answer is 部分
    """
    if isinstance(answer_id, torch.Tensor):
        answer_id = answer_id.cpu().tolist()
    elif not isinstance(answer_id, list):
        # 如果是其他类型（如numpy array），也转换为list
        answer_id = list(answer_id)
    #找到开始和结束的位置
    start_ids = tokenizer.encode('<answer>', add_special_tokens=False)
    end_ids = tokenizer.encode('</answer>', add_special_tokens=False)
    #任务开始寻找的东西,现在需要注意的应该就是数字了
    end_token_positions_list=find_token_sequence_position(answer_id,end_ids)#找到第一个</answer>的第一个匹配位置
    start_token_positions_list=find_last_token_sequence_position(answer_id,start_ids)#找到第一个<answer>的最后一个匹配位置

    if not start_token_positions_list or not end_token_positions_list:
        #这里我也分成两个版本:带the answer is 和不带the answer is的版本
        return{
            "answer_token_positions": [],
            "other_token_positions": [],#这个和answer_token_positions是反义词
            "struct_answer_token_positions": [],#这里包含了一部分结构化的答案(就是包含the answer is)#我感觉这里其实可以不用太关注的
        }
        #因为答案是数字所以不包含空格和回车
    # 这里找到了寻找
    start_position=start_token_positions_list[0]+1
    end_position=end_token_positions_list[0]-1

    # 边界检查：确保start_position <= end_position，且都在有效范围内
    if start_position > end_position or start_position < 0 or end_position >= len(answer_id):
        return {
            "answer_token_positions": [],
            "other_token_positions": list(range(len(answer_id))),
            #但是感觉the answer is没啥用,并不是结构常见的
            "struct_answer_token_positions": [],
        }

    #其实struct就比非other多了the answer is的位置部分
    struct_answer_token_positions=[]
    answer_token_positions=[]
    other_token_positions=[]

    # 还是得记录,-,.这些
    def is_digit_symbol_token(token_id):
        "判断token是否为数字0-9或者是一些符号"
        try:
            decoded = tokenizer.decode([token_id],skip_special_tokens=False)
            return decoded.strip() in ['0','1','2','3','4','5','6','7','8','9','.','-',',']
        except:
            return False
    
    # 1. 首先收集<answer></answer>内的数字token
    for i in range(start_position,end_position+1):
        if answer_id[i]!=space_id and answer_id[i]!=enter_id:
            #这里是基本常识,我跳过了一些比较基础的空白和回车
            if is_digit_symbol_token(answer_id[i]):
                answer_token_positions.append(i)
                struct_answer_token_positions.append(i)
            else:
                struct_answer_token_positions.append(i)

    # 2. 从第一个位置到<answer>之前的所有内容（因为模型回答时没有target:）
    # <answer>的开始位置（<answer>序列的第一个token位置）
    answer_start_position = start_token_positions_list[0] - len(start_ids) + 1
    
    # 将从位置0到<answer>之前的所有token位置加入到answer_token_positions
    if answer_start_position > 0:
        for i in range(0, answer_start_position):
            answer_token_positions.append(i)
            struct_answer_token_positions.append(i)
    
    # 去重并排序 struct_answer_token_positions
    struct_answer_token_positions = sorted(list(set(struct_answer_token_positions)))
    
    # 计算 other_token_positions：除去答案 token 的所有位置
    answer_token_set = set(answer_token_positions)
    other_token_positions = [idx for idx in range(len(answer_id)) if idx not in answer_token_set]
    
    return {
        "answer_token_positions": sorted(answer_token_positions),
        "other_token_positions": other_token_positions,
        "struct_answer_token_positions": struct_answer_token_positions,
    }

#适配answer_token_positions的位置(精准找到需要关注的地方)
#这里提前计算好mbpp所有需要的位置,但是展示的时候并不需要出现,因为发现后面可能用不到
def find_mbpp_answer_token_positions(
    answer_id,
    tokenizer:AutoTokenizer,
    space_id:int=220,
    enter_id:int=198,
):
#这里应该是通过找[begin]和[done]之间的位置,来去获取答案的区域
    if isinstance(answer_id, torch.Tensor):
        answer_id = answer_id.cpu().tolist()
    elif not isinstance(answer_id, list):
        # 如果是其他类型（如numpy array），也转换为list
        answer_id = list(answer_id)
    #找到开始和结束的位置,进而提取关键信息
    begin_ids = tokenizer.encode('[BEGIN]', add_special_tokens=False)
    done_ids = tokenizer.encode('[DONE]', add_special_tokens=False)
    begin_token_positions_list=find_last_token_sequence_position(answer_id,begin_ids)#找到第一个[BEGIN]的第一个匹配位置
    done_token_positions_list=find_token_sequence_position(answer_id,done_ids)#找到第一个[DONE]的第一个匹配位置
    if not begin_token_positions_list or not done_token_positions_list:
        return {
            "answer_token_positions": [],
            "other_token_positions": [],
        }
    begin_position=begin_token_positions_list[0]+1
    done_position=done_token_positions_list[0]-1

    # 从前向后找到第一个非空白/非回车的 token 作为答案起点
    answer_start_idx = None
    for i in range(begin_position, done_position + 1):
        if answer_id[i] != space_id and answer_id[i] != enter_id:
            answer_start_idx = i
            break
        # 如果整段都是空白，直接返回
    if answer_start_idx is None:
        return {
            "answer_token_positions": [],
            "other_token_positions": list(range(len(answer_id)))
        }
    
     # 从后向前找到最后一个非空白/非回车的 token 作为答案终点
    answer_end_idx = None
    for i in range(done_position, answer_start_idx - 1, -1):
        if answer_id[i] != space_id and answer_id[i] != enter_id:
            answer_end_idx = i
            break

    if answer_end_idx is None:
        return {
            "answer_token_positions": [],
            "other_token_positions": list(range(len(answer_id)))
        }   

    # 判断是否满足起点《终点的情况,这里记录答案位置
    if answer_start_idx<=answer_end_idx:
    # 记录答案区间（保留内部的空白和换行）
        answer_token_positions = list(range(answer_start_idx, answer_end_idx + 1))
    else:
        raise ValueError("answer_start_idx is greater than answer_end_idx")
    answer_token_set = set(answer_token_positions)
    other_token_positions = [idx for idx in range(len(answer_id)) if idx not in answer_token_set]
    return{
        "answer_token_positions": sorted(answer_token_positions),
        "other_token_positions": other_token_positions,
    }