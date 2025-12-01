from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from tqdm import tqdm
import os, sys, json
import numpy as np
from typing import List, Dict, Tuple
from transformers.models.electra.modeling_electra import ElectraSelfAttention
#加入煮目录,方便搜索到相应模块
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import eval_prompt,load_dataset
from src.eval_likelihood import get_log_likelihood 

#接受数据集,想办法调用底层函数进行
#这个不是生成单纯只是评估,所以是前向传播,而不是生成
#mode一般控制的是解码格式,所以可以不用考虑
#这个就已经返回模型生成的交叉熵了
def forward_likelihood(model,tokenizer,input,task,query_position,mc_num=128,batch_size=16,cfg_scale=0.,mask_id=126336):

    left_prompt,answer_prompt,right_prompt=eval_prompt(input,task,query_position)
    #这样的评估已经跟base还是instruct无关了,基本都是看的是是否是正确进行评测的
    prompt_left=torch.tensor(tokenizer(left_prompt)['input_ids']).to(model.device)
    answer_prompt=torch.tensor(tokenizer(answer_prompt)['input_ids']).to(model.device)
    if right_prompt is not None:
        prompt_right=torch.tensor(tokenizer(right_prompt)['input_ids']).to(model.device)
        likelihood=get_log_likelihood(model,prompt_left,answer_prompt,prompt_right,mc_num=mc_num,batch_size=batch_size,cfg_scale=cfg_scale,mask_id=mask_id)
    else:
        #我想测一下mc_num多少比较合适
        likelihood=get_log_likelihood(model,prompt_left,answer_prompt,mc_num=mc_num,batch_size=batch_size,cfg_scale=cfg_scale,mask_id=mask_id)
    #返回的是一个float,代表的是一个负数
    return likelihood

def main(args):
    #这里提前设定好参数
    task=args.task
    model_name=args.model_name
    device=args.device
    nshot=args.nshot
    data_path=args.data_path
    mc_num=args.mc_num
    batch_size=args.batch_size
    cfg_scale=args.cfg_scale
    samples_num=args.samples_num

    dataset=load_dataset(data_path,task)
    samples_num=min(samples_num,len(dataset))
    dataset=dataset[:samples_num]

    print('------------------ load model -----------------------')
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model=AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)

    print('--------------------start Eval----------------')
    #针对每一个样本我应该计算的是每一个位置
    #然后平均的是不同样本的同一个位置,列表推导式
    likelihood_avg=[0 for _ in range(nshot+1)]
    for position in range(nshot+1):
        likelihood_sum=0
        for _,input in enumerate(tqdm(dataset)):
            likelihood=forward_likelihood(model,tokenizer,input,task,position,mc_num=mc_num,batch_size=batch_size,cfg_scale=cfg_scale)
            likelihood_sum+=likelihood
        likelihood_avg[position]=likelihood_sum/len(dataset)
    
    print('--------------------end Eval----------------')
    print(likelihood_avg)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='sudoku')
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--nshot',type=int,default=5)
    parser.add_argument('--data_path',type=str,default='data/sudoku.csv')
    parser.add_argument('--mc_num',type=int,default=128)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--cfg_scale',type=float,default=0.)
    parser.add_argument('--samples_num',type=int,default=20)
    args=parser.parse_args()
    main(args)

