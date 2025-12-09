from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from tqdm import tqdm
import os, sys, json
import numpy as np
from typing import List, Dict, Tuple
from transformers.models.electra.modeling_electra import ElectraSelfAttention
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available, multi-GPU support disabled")
#加入煮目录,方便搜索到相应模块
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import query_extract, load_dataset, eval
from utils.attention_utils import get_token_position,find_answer_token_positions,find_token_sequence_position,find_last_token_sequence_position,find_zero_position,find_countdown_answer_token_positions,find_gsm8k_answer_token_positions,find_math_answer_token_positions,find_mbpp_answer_token_positions
from utils.conf_utils import cal_token_change_positive_total,cal_accmulate_conf,paint_conf,cal_accmulate_conf_countdown,cal_accmulate_conf_gsm8k,cal_accmulate_conf_math
from scripts.eval import generate
#现在就是根据输入进模型的来去看有没有效果,属于是后验的情况
#我先进行模块化吧,把输入输进去
#根据输入进去的来进行选择
#我先改进一下sudoku一下,因为sudoku的token变化是最大的,我想通过这个
#这个我想进行在线排序
#这里我需要修改一下,计算一下upperbound,同时最好看一下关系到的是什么东西,这样才能解决问题
def generate_improve(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot,mask_id=126336):
    #例子输入进了模型中,我先在要返回的不是答案,而是所有的步骤和answer token的位置
    #对于sudoku的一种适配而已,现在我要做的就是两版本
    #这里已经有根据输入进来的位置进行评测
    # zero_position_list = None
    # if task=='sudoku':
    #     zero_position_list=find_zero_position(input['Puzzle'])
    query=query_extract(input,task,query_position,gen_length,nshot)
    if situation=='base':
        user_input=query
    elif situation=='instruct':
        m=[{"role":"user","content":query}]
        user_input=tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    else:
        user_input = query
    
    # tokenize并转换为tensor
    prompt = tokenizer(user_input, return_tensors='pt')['input_ids']
    mask_positions=(prompt==mask_id).nonzero(as_tuple=True)
    if len(mask_positions[0])==0:
        raise ValueError("No mask tokens found in prompt")
    #找到开始的位置
    first_mask_pos=mask_positions[1][0].item()
    last_mask_pos=mask_positions[1][-1].item()
    if mode=='original':
        #这个只取出想要观测的指标,同时为了模版的统一化,我现在想的就是能不能把
        from src.generate import generate_with_conf
        x,token_change,conf=generate_with_conf(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence',
        return_order=False,return_conf_diff=False,return_entropy=False,return_token_change=True,return_conf=True)
        x_answer=x[:,first_mask_pos:last_mask_pos+1]
        if task=='sudoku':
            answer_token_positions=find_answer_token_positions(x_answer[0],tokenizer,space_id=220,enter_id=198)
        elif task=='countdown':
            #找到了需要去关注的地方
            answer_token_positions=find_countdown_answer_token_positions(x_answer[0],tokenizer,space_id=220,enter_id=198)
        elif task=='gsm8k':
            answer_token_positions=find_gsm8k_answer_token_positions(x_answer[0],tokenizer)
        elif task=='math500':
            answer_token_positions=find_math_answer_token_positions(x_answer[0],tokenizer)
        elif task=='mbpp':
            #这里mbpp回答的就是完整答案,只用返回全部坐标即可,找到关键的地方去画图
            answer_token_positions=find_mbpp_answer_token_positions(x_answer[0],tokenizer)
        else:
            raise NotImplementedError(f"Task {task} not implemented.")
    elif mode=='conf_sampler':
        
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")
    #这个是后验的阶段


    return token_change,conf,answer_token_positions

#这里应该是针对一个地方进行探索:就是找到最适合嵌入的位置.计算出来的值,进行根据版本计算平均值
#根据version版本按照相应字段进行平均值的计算
def compute_position(token_change, conf, answer_position, version):
    """
    计算conf在指定位置的所有步平均值和token_change的总和
    
    Args:
        token_change: 每一步的token变化列表，每个元素是[gen_length]的numpy数组
        conf: 每一步的置信度列表，每个元素是[gen_length]的numpy数组
        answer_position: 答案位置信息字典，包含answer_token_positions字段
        version: 模式，'all' 或 'answer_token'
    
    Returns:
        tuple: (conf_mean, token_span) 两个float值
            - conf_mean: conf在指定位置的所有步平均值
            - token_span: token_change在指定位置的所有步总和
    """
    # 检查输入
    assert len(conf) == len(token_change), "conf和token_change列表长度必须相同"
    #紧急避险,防止出现错误的情况
    if answer_position is None or token_change is None or conf is None:
        return np.nan, 0.0
    if len(conf) == 0 or len(token_change) == 0:
        # 如果列表为空，返回默认值而不是报错
        return np.nan, 0.0
    if len(conf) != len(token_change):
        print(f"Warning: conf和token_change列表长度不一致: conf={len(conf)}, token_change={len(token_change)}，返回默认值")
        return np.nan, 0.0
    
    # 根据version确定要使用的位置
    if version == 'all':
        # 使用所有位置，需要从第一步获取长度
        if len(conf) > 0:
            gen_length = len(conf[0])
            positions = list(range(gen_length))
        else:
            return np.nan, 0.0
    elif version == 'answer_token':
        # 使用answer_position中的answer_token_positions字段
        positions = answer_position.get('answer_token_positions', [])
        #得做好找不到的情况,健全代码
        if not positions:
            # 如果没有找到位置，返回nan和0
            return np.nan, 0.0
    else:
        raise ValueError(f"Unknown version: {version}. Must be 'all' or 'answer_token'")
    
    # 存储所有步中指定位置的conf值
    conf_values = []
    # 存储每一步的token_change变化范围
    step_token_spans = []
    
    # 遍历所有步
    for step_idx in range(len(conf)):
        step_conf = conf[step_idx]
        step_token_change = token_change[step_idx]
        
        # 确保是一维数组
        assert step_conf.ndim == 1, f"step_conf的维度应该是1,当前维度是{step_conf.ndim}"
        assert step_token_change.ndim == 1, f"step_token_change的维度应该是1,当前维度是{step_token_change.ndim}"
        
        # 提取指定位置的值
        conf_at_positions = step_conf[positions]
        token_change_at_positions = step_token_change[positions]
        
        # 过滤掉 -np.inf（对于conf）
        valid_conf_mask = conf_at_positions != -np.inf
        valid_conf_values = conf_at_positions[valid_conf_mask]
        
        # 收集conf的有效值
        if len(valid_conf_values) > 0:
            conf_values.extend(valid_conf_values.tolist())
        
        # 对于token_change，找到大于0的位置索引（在原始positions中的索引）
        # 首先找到在指定位置范围内，token_change > 0 且不是-inf的位置
        valid_token_change_mask = (token_change_at_positions != -np.inf) & (token_change_at_positions > 0)
        valid_token_indices = np.where(valid_token_change_mask)[0]
        
        # 计算该步的token变化范围
        if len(valid_token_indices) > 0:
            # 将相对索引转换为绝对位置索引
            absolute_indices = np.array(positions)[valid_token_indices]
            # 计算范围：max - min + 1
            step_span = int(absolute_indices.max() - absolute_indices.min() + 1)
            step_token_spans.append(step_span)
        else:
            step_token_spans.append(0)
    
    # 计算conf的平均值
    if len(conf_values) > 0:
        conf_mean = float(np.mean(conf_values)) if np.isfinite(np.mean(conf_values)) else np.nan
    else:
        conf_mean = np.nan
    
    # 计算token_span：取所有步中最大的变化范围
    token_span = float(max(step_token_spans)) if len(step_token_spans) > 0 else 0.0
    
    return conf_mean, token_span


# 开始进行位置的筛选
#这里其实是根据制定nshot来去选择每一个position放置进去搜集conf和token进行筛选排序
#进而据此来进行评估eval
#目前没有验证集,直接上手开始进行计算
#得出的应该是一种顺序,作为一种前置过程
#因为是针对每一种shot的位置input进行改善
def select_position(model,tokenizer,version,nshot,input,task,mode,
steps,gen_length,block_length,temperature,situation,lamda1=0.8,lamda2=0.2):
    #针对每一种输入,我都去找最好的顺序,来看看效果,同时我给自己了两版
    conf_list=[]
    token_span_list=[]
    #位置按照从0到nshot到位置进行区分,所以还要给对应位置计算排位分才行
    for position in range(nshot+1):
        #针对每一个position,先计算对应的conf和token_span

        token_change,conf,answer_position=generate_improve(model,tokenizer,input,task,steps,gen_length,block_length,
        temperature,mode,situation,position,nshot)
        #根据任务计算该计算的位置,不过以后可以直接提取,这一步都不太需要了
        conf_mean,token_span=compute_position(token_change,conf,answer_position,version)
        conf_list.append(conf_mean)
        token_span_list.append(token_span)
    
    #我可以用排位,我现在觉得每个值进行归一化也可以,但是就是万一两边差别太大就比较难说
    #计算排位分：排名第1（最大值）→ nshot分，排名第n（最小值）→ 0分
    conf_array = np.array(conf_list)
    token_span_array = np.array(token_span_list)
    
    # 对conf_list进行排名（降序：最大值排名第1，从0开始）
    conf_ranks = np.argsort(np.argsort(-conf_array, kind='stable'))
    conf_list_rank = [float(nshot - rank) if not np.isnan(val) else 0.0 for rank, val in zip(conf_ranks, conf_list)]
    
    # 对token_span_list进行排名（降序：最大值排名第1，从0开始）
    token_span_ranks = np.argsort(np.argsort(-token_span_array, kind='stable'))
    token_span_list_rank = [float(nshot - rank) if not np.isnan(val) else 0.0 for rank, val in zip(token_span_ranks, token_span_list)]
    
    score = lamda1 * np.array(conf_list_rank) + lamda2 * np.array(token_span_list_rank)
    
    # 找到score的最大值及其位置
    max_score = np.max(score)
    best_position = int(np.argmax(score))  # 最大值的索引位置
    #核心lamda1*conf+lamda2*token_span,然后取最大值的位置,选出了最好的位置
    return best_position

def main(args):
    task=args.task
    model_name=args.model_name
    device=args.device
    version=args.version
    nshot=args.nshot
    steps=args.steps
    gen_length=args.gen_length
    block_length=args.block_length
    temperature=args.temperature
    lamda1=args.lamda1
    lamda2=args.lamda2
    data_path=args.data_path
    samples_num=args.samples_num
    mode=args.mode
    result_path=args.result_path
    #查看是否是零位置有效
    compute_zero=args.compute_zero
    
    # 检查是否使用 Accelerate（通过环境变量判断是否通过 accelerate launch 启动）
    accelerator = None
    if ACCELERATE_AVAILABLE:
        # 检查是否通过 accelerate launch 启动（会设置 WORLD_SIZE 环境变量）
        if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
            accelerator = accelerate.Accelerator()
            print(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] Accelerate initialized")
        elif 'ACCELERATE_CONFIG' in os.environ:
            # 也可能通过配置文件启动
            accelerator = accelerate.Accelerator()
            if accelerator.num_processes > 1:
                print(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] Accelerate initialized")
            else:
                accelerator = None
    
    # 加载完整数据集（用于评估时）
    full_dataset = load_dataset(data_path, task)
    # 取出前面的部分sample,方便进行取值计算
    full_dataset = full_dataset[:samples_num]
    
    # 如果使用 Accelerate，分割数据集用于并行处理
    if accelerator is not None:
        with accelerator.split_between_processes(full_dataset) as subset_dataset:
            dataset = subset_dataset
            print(f'[Rank {accelerator.process_index}] Load dataset: {len(dataset)} samples (total: {len(full_dataset)})')
    else:
        dataset = full_dataset
        print('------------------Load dataset------------------')
    
    print('------------------ load model -----------------------')
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,local_files_only=True)
    
    # 如果使用 Accelerate，手动管理设备分配，避免 DDP 包装（推理任务不需要 DDP）
    if accelerator is not None:
        # 获取当前进程对应的设备
        device = accelerator.device
        print(f"[Rank {accelerator.process_index}] Loading model on {device}")
        
        # 加载模型到 CPU，然后手动移动到对应 GPU
        # 这样可以避免 DDP 包装，减少内存占用
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        # 手动移动到对应设备，不使用 prepare（避免 DDP 包装）
        model = model.to(device)
        print(f"[Rank {accelerator.process_index}] Model loaded on {device}")
    else:
        # 不使用 Accelerate 时，直接加载到指定设备
        model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    
    model.eval()
    print('--------------------start pre-processing----------------')
    answers=[]
    zero_answers=[]#对照组
    
    # 使用 tqdm 时，只在主进程显示进度条
    dataset_iter = dataset
    if accelerator is not None and accelerator.num_processes > 1:
        # 多进程时，只在主进程显示进度条
        if accelerator.is_main_process:
            dataset_iter = tqdm(dataset, desc=f"[Rank {accelerator.process_index}] Processing")
        else:
            dataset_iter = dataset
    else:
        dataset_iter = tqdm(dataset, desc="Processing")
    
    for idx, input_item in enumerate(dataset_iter):
        #这里我先硬编码成一半,之后需要多少可以通过别的来确定
        #针对每一种输入,我都要进行pre-processing,然后选择最好的位置和放在最后进行比较
        if 'Instruct' in model_name:
            situation='instruct'
        else:
            situation='base'
        
        best_position=select_position(model,tokenizer,version,nshot,input_item,task,mode,int(steps/4),gen_length,block_length,temperature,situation,lamda1,lamda2)
        if accelerator is not None:
            print(f"[Rank {accelerator.process_index}] Best position for input {idx} is {best_position}")
        else:
            print(f"Best position for input {idx} is {best_position}")
        #开始进行测试
        #我得设置对照组,看看放在最后对比结果
        answer=generate(model,tokenizer,input_item,task,steps,gen_length,block_length,temperature,mode,situation,best_position,nshot)
        #之后还要补充代码的健全性
        answers.append(answer)
        if compute_zero:
            #记录放置在最后的位置
            zero_answer=generate(model,tokenizer,input_item,task,steps,gen_length,block_length,temperature,mode,situation,0,nshot)
            zero_answers.append(zero_answer)
        else:
            zero_answers.append(None)
    
    # 如果使用 Accelerate，收集所有进程的结果
    if accelerator is not None and accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            # 收集 answers
            all_answers_list = [None] * accelerator.num_processes
            dist.all_gather_object(all_answers_list, answers)
            
            # 收集 zero_answers（如果需要）
            if compute_zero:
                all_zero_answers_list = [None] * accelerator.num_processes
                dist.all_gather_object(all_zero_answers_list, zero_answers)
            else:
                all_zero_answers_list = None
            
            # 在主进程中合并结果
            if accelerator.is_main_process:
                # 展平所有结果，按进程顺序合并
                merged_answers = []
                for proc_answers in all_answers_list:
                    if proc_answers is not None:
                        merged_answers.extend(proc_answers)
                answers = merged_answers
                
                if compute_zero:
                    merged_zero_answers = []
                    for proc_zero_answers in all_zero_answers_list:
                        if proc_zero_answers is not None:
                            merged_zero_answers.extend(proc_zero_answers)
                    zero_answers = merged_zero_answers
            else:
                # 非主进程不继续执行评估
                return
    
    # 只在主进程执行评估和保存
    if accelerator is None or accelerator.is_main_process:
        #写入对应文件
        # 使用 "auto-icl" 作为 position，表示这是自动选择的最佳位置
        acc=eval(task,answers,full_dataset,result_path,args,position="auto-icl")
        # zero_acc=eval(task,zero_answers,dataset,result_path,args)
        if acc is not None:
            print(f"Accuracy: {acc:.4f}")
        else:
            print("Accuracy: Not available (task may not return accuracy)")
        
        if compute_zero:
            zero_acc=eval(task,zero_answers,full_dataset,result_path,args,position=0)
            if zero_acc is not None:
                print(f"Zero Accuracy: {zero_acc:.4f}")
            else:
                print("Zero Accuracy: Not available (task may not return accuracy)")
        #把产生的结果记录一下

#这里我应该提供给
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='sudoku')
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--version',type=str,default='all')
    parser.add_argument('--nshot',type=int,default=4)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--gen_length',type=int,default=32)
    parser.add_argument('--block_length',type=int,default=32)
    parser.add_argument('--temperature',type=float,default=0.0)
    parser.add_argument('--data_path',type=str,default='./data/sudoku.csv')
    parser.add_argument('--samples_num',type=int,default=200)
    parser.add_argument('--mode',type=str,default='original')
    parser.add_argument('--result_path',type=str,default='./results/sudoku/improve')
    #我再想要不要学习一下学习率(或者我手动调参,让他们做到最好)
    parser.add_argument('--lamda1',type=float,default=0.8)
    parser.add_argument('--lamda2',type=float,default=0.2)
    #是否计算最后一个位置
    parser.add_argument('--compute_zero',action='store_true',default=False)
    args=parser.parse_args()
    
    # 使用 try-finally 确保在程序退出前清理分布式进程组
    try:
        main(args)
    finally:
        # 清理分布式进程组，避免资源泄漏
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()