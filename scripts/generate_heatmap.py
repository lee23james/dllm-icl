# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from tqdm import tqdm
import os, sys, json
import numpy as np
#加入煮目录,方便搜索到相应模块
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import query_extract, load_dataset

def generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,mask_id=126336):
    #不想做gpqa的解码顺序表示,因为这个数据集并没有表现很好
    if task!='gpqa':
        query=query_extract(input,task,query_position,gen_length)
        if situation=='base':
            user_input=query
        
        elif situation=='instruct':
            m=[{"role":"user","content":query}]
            user_input=tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
        prompt=tokenizer(user_input)['input_ids']
        #这里已经把prompt进行升维处理了
        prompt=torch.tensor(prompt).to(model.device).unsqueeze(0)
        #开始取顺序
        #针对位置进行特异化
        mask_positions =(prompt==mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")
        #找到第一个和最后一个mask token的位置,item转化张量
        first_mask_pos = mask_positions[1][0].item()
        # last_mask_pos=mask_positions[1][-1].item()

        if mode=='original':
            from src.generate import generate
            _,orders=generate(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence',return_order=True)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        #一个list,每一个就的是每一步的转移结果(包括未转移的)
        orders_result=[]
        #每一个元素代表的是每一步转移的元素(不包括未转移的)
        determined=[]
        #取出block出来
        #对每一步进行探索
        #这里看的是每一个块,然后依次累积变成按时间的顺序
        for block_index in orders.keys():
            #每一个block块字段可能都包含多个步骤,需要step by step分析
            for step in range(len(orders[block_index])):
                this_step = torch.full((1,gen_length),0,dtype=torch.float16)
                #先记录当前步哪些被转移了
                this_step[0,orders[block_index][step]]=1
                if determined !=[]:
                    #然后记录的是总共有哪些token被转移(就是之前的)
                    #determined相当于记录的是之前所有确定转移的位置
                    this_step[0,determined]=1
                #添加的是每一步,而不局限于块内的步
                orders_result.append(this_step)
                #添加转移的元素
                determined.extend([j for j in orders[block_index][step]])
           #针对每一个block
    else:#就是gpqa
        raise NotImplementedError(f"Task {task} not implemented.")
    # 返回list[torch.Tensor],每一个torch.Tensor代表每一步的探索结果,一个list代表步数(和block无关)        
    return orders_result

def save_heatmap_params(confidence_result,task,position,mode:str,index:int=None):
    #将tensor转化为list
    #暂时只考虑记录这一个解码策略,后面其他的等到他们调研完再记录
    #mode可以选择single还是avg
    confidence_result_list = confidence_result.tolist() if isinstance(confidence_result, np.ndarray) else confidence_result
    data={"confidence_result":confidence_result_list}
    #使用项目根目录下的params文件夹
    params_dir=os.path.join(project_root,"params","heatmap_params")
    if mode=='single':
        # filename=f"heatmap_params/{task}_{position}_{mode}_{index}.json"
        filename=os.path.join(params_dir,f"{task}_{position}_{mode}_{index}.json")
    else:
        # filename=f"heatmap_params/{task}_{position}_{mode}.json"
        filename=os.path.join(params_dir,f"{task}_{position}_{mode}.json")
    if not os.path.exists(params_dir):
        os.makedirs(params_dir,exist_ok=True)
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    
def main(args):
    task=args.task
    model_name=args.model_name
    device=args.device
    gen_length=args.gen_length
    steps=args.steps
    block_length=args.block_length
    temperature =args.temperature
    data_path=args.data_path
    #想要关注的数据集数量
    samples_num=args.samples_num
    # mode=args.mode
    #换一个角度(我现在想关注的是位置的变化)
    nshot=args.nshot
    confidence_result={}#变成和位置相关的dict,key是位置,value是解码策略
    #目前只关注一个mode
    modes=['original']
    dataset=load_dataset(data_path,task)
    #取出前10个sample
    samples_num=min(samples_num,len(dataset))
    dataset=dataset[:samples_num]

    print('------------------ load model -----------------------')
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model=AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)

    print('--------------------start Answering----------------')

    #计算各种解码策略的效果(位置和解码效果的组合)
    #nshot有nshot+1的位置组合(目前我们只是锁定了一部分适合跑的shot,后期shot就是变化的量):从0到nshot
    #一次性生成所有位置的图片
    for position in range(nshot+1):
        #对每个解码进行排列组合
        for mode in modes:
            print(f'------------------ start Answering with mode {mode} -----------------------')
            for index, input in enumerate(tqdm(dataset, desc=f"Processing {mode}")):
                #因为要考虑到situation和query_position
                #这里还要考虑situation,现在我只考虑base和instruct
                if 'Instruct' in model_name:
                    situation='instruct'
                else:
                    situation='base'
                #list[tensor],每一个都是每步结果
                results=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,position)
                #这个是解码策略
                if mode =='original':
                    #保存然后绘图
                    results_clone = [t.clone() for t in results]
                    results_clone=np.array(results_clone)
                    save_heatmap_params(results_clone,task,position,'single',index)
                    if position not in confidence_result:
                        #字段的值是list[tensor]
                        confidence_result[position]=results
                    else:
                        for i in range(len(results)):
                            confidence_result[position][i]+=results[i]    
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented.")
    print('--------------------finish Answering----------------')
    #我想先画十张图然后再画平均的图,同时我想关注一下解码结果,有没有解码出正确答案

    for position in range(nshot+1):
        confidence_result[position]=np.array(confidence_result[position]) / samples_num
        #根据位置和解码策略来画图:表现解码顺序(取平均值)
        save_heatmap_params(confidence_result[position],task,position,'avg')

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='sudoku')
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--gen_length',type=int,default=128)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--block_length',type=int,default=128)
    parser.add_argument('--temperature',type=float,default=0.0)
    parser.add_argument('--data_path',type=str,default='./data/sudoku.csv')
    parser.add_argument('--samples_num',type=int,default=10)
    parser.add_argument('--nshot',type=int,default=5)
    #没用到gpqa,所以不需要SEED
    args=parser.parse_args()
    main(args)
    

