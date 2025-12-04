#用这个来去选择最佳顺序
#依靠支持集来去选择更好的顺序
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import argparse
from tqdm import tqdm
import os, sys, json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
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
from scripts.eval import generate#利用generate去获得支持集的答案
from utils.judge_python_code import evaluate_python_files
#通过支持集来去看,这个直接从准确率看即可
#sample_num是一个关键.,可以从这里去探索用多少个去获得支持集的答案,准确率最高,这里额外需要支持集地址,方便筛选,默认thread是None
#dev_samples_num是一个重要参数,用来限制dev_dataset的样本数量,方便进行计算
def seclect_order(tokenizer,model_name,args,model,task,dev_samples_num,dev_data_path,nshot,steps,gen_length,block_length,temperature,mode,situation,thread=None,accelerator=None):
    # 先加载完整的数据集（用于评估时）
    full_dev_dataset = load_dataset(dev_data_path, task, dev_samples_num)
    
    # 如果使用 Accelerate，分割数据集用于并行处理
    if accelerator is not None:
        with accelerator.split_between_processes(full_dev_dataset) as subset_dataset:
            dev_dataset = subset_dataset
            print(f'[Rank {accelerator.process_index}] Load dev dataset: {len(dev_dataset)} samples (total: {len(full_dev_dataset)})')
    else:
        dev_dataset = full_dev_dataset
        print('------------------Load dev dataset------------------')
    # tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,local_files_only=True,local_files_only=True)
    # if 'LLaDA' in model_name:
    #     model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    # else:
    #     model=AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    # #进入评测环节,主要是用支持集去计算什么时候最好,加速模式更好
    # model.eval()

    print('------------------Start Answering------------------')
    #考虑是哪一个位置最好
    #默认不用gpqa,之后我会把gpqa进行下架
    #还要考虑fast-dllm这样的解码策略
    #怎么处理mbpp
    acc_list=[]
    for query_position in range(nshot+1):
        results=[]
        #对于每一个query_position都要选取最好的位置
        # 使用 tqdm 时，只在主进程显示进度条
        dataset_iter = dev_dataset
        if accelerator is not None and accelerator.num_processes > 1:
            # 多进程时，只在主进程显示进度条
            if accelerator.is_main_process:
                dataset_iter = tqdm(dev_dataset, desc=f"[Rank {accelerator.process_index}] Position {query_position}")
            else:
                dataset_iter = dev_dataset
        else:
            dataset_iter = tqdm(dev_dataset, desc=f"Position {query_position}")
        
        for input in dataset_iter:
            if 'Instruct' in model_name:
                situation='instruct'
            else:
                situation='base'
            #这里假设不等与gpqa,然后慢慢进行计算
            if task!='gpqa':
                answer=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot,thread=thread)
            else:
                raise NotImplementedError(f"Task {task} is gpqa, not implemented.")
            results.append(answer)
        
        # 如果使用 Accelerate，收集所有进程的结果
        if accelerator is not None and accelerator.num_processes > 1:
            # 使用 PyTorch 的分布式通信收集所有进程的结果
            import torch.distributed as dist
            if dist.is_initialized():
                # 使用 all_gather_object 收集所有进程的结果
                all_results_list = [None] * accelerator.num_processes
                dist.all_gather_object(all_results_list, results)
                
                # 在主进程中合并结果
                if accelerator.is_main_process:
                    # all_results_list 是一个包含所有进程结果的列表
                    # 展平所有结果，按进程顺序合并
                    merged_results = []
                    for proc_results in all_results_list:
                        if proc_results is not None:
                            merged_results.extend(proc_results)
                    results = merged_results
                else:
                    # 非主进程跳过评估，继续下一个 query_position
                    continue
            else:
                # 如果分布式未初始化，直接使用当前结果
                pass
        #注意这里一定要记录一下 ,这里返回准确率,对于每一个位置,都要去测试所有的准确度,mbpp也只是会生成对应的文件,我需要在结束的时候运行评估代码来记录正确与否
        #通过args记录要去评测的是什么,对于mbpp,acc没用,每个position都会生成最新的评估函数,所以只需要在之后进行计算即可
        # 只在主进程执行评估（因为只有主进程有完整的结果）
        if task == 'mbpp':
            if accelerator is None or accelerator.is_main_process:
                # 对于 mbpp，调用 eval_mbpp 生成 Python 文件
                from utils.eval_utils import eval_mbpp
                # 使用 result_path 如果存在，否则使用默认的 results 目录
                result_path = getattr(args, 'result_path', None)
                if result_path is None:
                    # 如果没有指定 result_path，使用默认的 results/{task}_results 目录
                    result_path = f'./results/{task}_results'
                # 使用完整的数据集进行评估（results 已经包含了所有进程的结果）
                eval_mbpp(results, full_dev_dataset, result_path, args, position=query_position)
        else:
            # 只在主进程执行评估（因为只有主进程有完整的结果）
            if accelerator is None or accelerator.is_main_process:
                # 使用 result_path 如果存在，否则使用默认的 results 目录
                result_path = getattr(args, 'result_path', None)
                if result_path is None:
                    # 如果没有指定 result_path，使用默认的 results/{task}_results 目录
                    result_path = f'./results/{task}_results'
                # 使用完整的数据集进行评估（results 已经包含了所有进程的结果）
                acc = eval(task, results, full_dev_dataset, result_path, args, position=query_position)
                acc_list.append(acc)
    
    # 如果是 mbpp 任务，在所有文件生成后统一评估
    # 只在主进程执行评估（因为只有主进程有完整的结果）
    if task == 'mbpp':
        # 如果使用 Accelerate，只在主进程执行评估
        if accelerator is None or accelerator.is_main_process:
            # 获取 result_path，用于查找生成的 Python 文件
            result_path = getattr(args, 'result_path', None)
            if result_path is None:
                # 如果没有指定 result_path，使用默认的 results/{task}_results 目录
                result_path = f'./results/{task}_results'
            # 调用 judge_python_code 进行评估，iswrite=False 只返回 Accuracy 列表
            judge_result = evaluate_python_files(
                folder_path=result_path,
                nshot=nshot,
                steps=steps,
                gen_length=gen_length,
                find_not_position=False,
                iswrite=False,#不让写入文件,只返回Accuracy列表
                return_json=False,#不返回json
                output_path=None,
                # find_best_position=True
            )
            if judge_result and 'Accuracy' in judge_result:
                acc_list = judge_result['Accuracy']
                print(f"MBPP Accuracy list: {acc_list}")
            else:
                print("Warning: Failed to get MBPP accuracy from judge_python_code")
        else:
            # 非主进程，acc_list 保持为空，稍后会返回 None
            acc_list = []
    
    #通过这样获得计算acc的方法,然后开始进行计算
    # 只在主进程打印完成信息
    if accelerator is None or accelerator.is_main_process:
        print('-------------------Finish----------------')

    #现在的acc_list是每一个位置的准确率按照0到nshot的顺序进行排列的,我要用这个顺序进行选择来运行了
    # 找出准确率最高的 position
    # 如果使用 Accelerate，只在主进程返回结果
    if accelerator is not None and accelerator.num_processes > 1:
        if not accelerator.is_main_process:
            return None
    
    if acc_list:
        best_acc = max(acc_list)
        best_position = acc_list.index(best_acc)  # 返回第一个最大值的索引
        print(f"Best position: {best_position} with accuracy: {best_acc:.4f}")
        print(f"All accuracies: {acc_list}")
        return best_position
    else:
        print("Warning: acc_list is empty, cannot determine best position")
        return None
    
#这里能不能只是记录准确率,并不会返回值,我需要进行记录
#这里返回了最快位置
def main(args):
    task=args.task
    model_name=args.model_name
    device=args.device
    nshot=args.nshot
    steps=args.steps
    gen_length=args.gen_length
    block_length=args.block_length
    temperature=args.temperature
    data_path=args.data_path
    situation=args.situation
    #如果之后要换解码策略,这个是阈值参数,但是我应该不需要
    thread=args.thread
    dev_data_path=args.dev_data_path
    samples_num=args.samples_num
    dev_samples_num=args.dev_samples_num
    result_path=getattr(args, 'result_path', None)
    if result_path is None:
        # 如果没有指定 result_path，使用默认的 results/{task}_results 目录
        result_path = f'./results/{task}_results'
    generate_file=args.generate_file#是否生成文件
    isvalidate=args.isvalidate#这里我只是需要一个参数,当得出了最好的位置之后,是否需要进行验证,默认是False,因为这和普通的evaluate其实没啥差别(感觉意义不大)
    mode=args.mode
    output_path=args.output_path#输出路径,用来记录结果的
    dataset=load_dataset(dev_data_path,task,samples_num)
    print('------------------Load dev dataset------------------')
    
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
    
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,local_files_only=True)
    
    # 如果使用 Accelerate，手动管理设备分配，避免 DDP 包装（推理任务不需要 DDP）
    if accelerator is not None:
        # 获取当前进程对应的设备
        device = accelerator.device
        print(f"[Rank {accelerator.process_index}] Loading model on {device}")
        
        # 加载模型到 CPU，然后手动移动到对应 GPU
        # 这样可以避免 DDP 包装，减少内存占用
        if 'LLaDA' in model_name:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
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
        if 'LLaDA' in model_name:
            model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True)
        else:
            model=AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True)
        model = model.to(device)
    
    model.eval()
    best_position=seclect_order(tokenizer,model_name,args,model,task,dev_samples_num,dev_data_path,nshot,steps,gen_length,block_length,temperature,mode,situation,thread,accelerator=accelerator)
    
    # 只在主进程执行后续操作（评估和保存）
    if accelerator is None or accelerator.is_main_process:
        print(f"Best position: {best_position}")
        #利用这个位置去跑,只有是true的时候才会生成正确结果
        if isvalidate is True:
            results=[]
            for idx,input in enumerate(tqdm(dataset)):
                answer=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,best_position,nshot,thread=thread)
                #还得看是哪一种任务
                results.append(answer)

            #计算准确率
            if task == 'mbpp':
                # 对于 mbpp，调用 eval_mbpp 生成 Python 文件
                #记录最好的位置
                from utils.eval_utils import eval_mbpp
                #生成测评文件
                #生成的是对应位置的
                #这里进行标记,看看是在哪个位置表现最好
                #直接生成相应的文件,如果之后要评估的话还是需要进行一些改进的
                print(f"task:{task} Best position: {best_position} dev_samples_num: {dev_samples_num}")
                if generate_file is True:
                    eval_mbpp(results, dataset, data_path, args, position=f"best_position_{dev_samples_num}")
            else:
                #当前位置直接生成即可，使用 result_path 而不是 data_path
                acc = eval(task, results, dataset, result_path, args, position=best_position)
                print(f"task:{task} Best position: {best_position} dev_samples_num: {dev_samples_num} Accuracy: {acc:.4f}")
                # 如果是 mbpp 任务，在所有文件生成后统一评估

        if output_path:
            try:
                base_dir = Path(output_path) / task / f"shot_{nshot}_step_{steps}_gen_{gen_length}" / f"dev_samples_num_{dev_samples_num}"
                base_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_path = base_dir / f"{timestamp}.json"
                summary_payload = {
                    'task': task,
                    'best_position': best_position,
                    'dev_samples_num': dev_samples_num,
                    'nshot': nshot,
                    'steps': steps,
                    'gen_length': gen_length,
                    'block_length': block_length,
                    'samples_num': samples_num
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_payload, f, ensure_ascii=False, indent=2)
                print(f"Best position summary saved to: {summary_path}")
            except Exception as exc:
                print(f"Warning: failed to save best position summary to {output_path}: {exc}")

        print('-------------------Finish----------------')
    #在这里记录一下输出路径
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='mbpp')
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--nshot',type=int,default=4)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--gen_length',type=int,default=32)
    parser.add_argument('--block_length',type=int,default=32)
    parser.add_argument('--temperature',type=float,default=0.0)
    parser.add_argument('--data_path',type=str,default='./data/mbpp.json')
    parser.add_argument('--dev_data_path',type=str,default='./data/mbpp.json')
    parser.add_argument('--samples_num',type=int,default=500)
    parser.add_argument('--dev_samples_num',type=int,default=50)
    parser.add_argument('--result_path',type=str,default=None,help='Results directory path. If not specified, defaults to ./results/{task}_results')
    parser.add_argument('--generate_file',action='store_true',default=False)
    parser.add_argument('--isvalidate',action='store_true',default=False)#默认不进行,只是进行计算
    parser.add_argument('--mode',type=str,default='original')
    parser.add_argument('--situation',type=str,default='base')
    parser.add_argument('--thread',type=int,default=0.9)
    parser.add_argument('--output_path',type=str,default='./results/best_position')
    args=parser.parse_args()
    
    # 使用 try-finally 确保在程序退出前清理分布式进程组
    try:
        main(args)
    finally:
        # 清理分布式进程组，避免资源泄漏
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()