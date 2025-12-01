import re
import os
import sys
import time
import pickle
import random
import logging
import argparse
from transformers.integrations.deepspeed import importlib_metadata
import yaml
from typing import List
from typing import Dict
import easydict
from hashlib import md5
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.load_groundtruth import load_groundtruth
from utils.eval_utils import eval

from src.dataset import PromptCorpus

logger =logging.getLogger(__name__)
#调用自己的模型
def init_model(model_name,device):
    #加载模型
    model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16).to(device)
    if torch.cuda.is_available():
        model.cuda()
    return model

#在这里处理多个位置的情况
def generate(restricted_token,data,model,tokenizer,device,steps:int=3,gen_length:int=3,block_length:int=3,temperature:float=0.,mode:str="original",mask_id:int=126336)->List[Dict[int,str]]:
    #我想在这里处理List tensor的情况,因为list(tensor)是不同长度的tensor,所以我可以通过for 循环一次处理,相当于每一次的batch都是1
    assert len(data["query_position"]) == len(data["input_sequence"]), "query_position and input_sequence must have the same length"
    #收集各个位置的答案
    answers=[]
    #记录每一个位置和每一个输入的prompt,前者是list[int],后者是list[tensor]
    for (position,prompt) in zip(data["query_position"],data["input_sequence"]):
        #每一个prompt都是可变长度的tensor,我现在需要针对每一个tensor进行处理
        #添加一个维度,变成(1,length)
        answer={}
        #绑定输入必须要是二维的
        mask_positions = (prompt == mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")
        # 找到第一个和最后一个mask token的位置
        first_mask_pos = mask_positions[1][0].item()
        #都是一个,也不需要去找最后一个
        # last_mask_pos = mask_positions[1][-1].item()

        # 打印输入给模型的prompt（解码）输入给模型前,看看prompt是什么
        # try:
        #     if prompt.dim() == 1:
        #         prompt_text = tokenizer.decode(prompt, skip_special_tokens=False)
        #     else:
        #         prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=False)
        #     print(f"\n===== PROMPT (position={position}) =====", flush=True)
        #     print(prompt_text, flush=True)
        #     print("===== END PROMPT =====\n", flush=True)
        # except Exception as e:
        #     print(f"[WARN] Failed to decode prompt for position={position}: {e}", flush=True)

        prompt=prompt.to(device)

        if mode=='original':
            from src.generate import forward
            #这里需要的是答案,但是可能会有魔改,目前先返回一下答案吧
            out=forward(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence')
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        #输出的是tensor[1,gen_length],输出一下
        # print(tokenizer.batch_decode(out[:,:],skip_special_tokens=False)[0])
        #解码看看输出
        answer["prompt"]=tokenizer.batch_decode(out[:,first_mask_pos:first_mask_pos+gen_length],skip_special_tokens=False)[0]
        # answer["analysis"]=analysis
        pos=int(position.item()) if torch.is_tensor(position) else int(position)
        answer["position"]=pos

        # print(answer["prompt"])
        # print(answer["position"])
        #明白了,每一次都是不同的position,所以answers只会不断的追加这些答案,最后会变成list[dict]
        answers.append(answer)
    #List[Dict]
    return answers

#生成 list[dict[int,str]]
def inference_mode(args,tokenizer,model,dataset,restricted_token):
    result=[]
    #这里就是每一次都取出所有的数据,然后再用这几个数据构造不同位置的数据,并进行记录
    for data in tqdm(dataset):
        model.eval()
        # 调试输出 data 的关键维度/信息
        #后续要删除
        # try:
        #     seq_list = data.get("input_sequence", [])
        #     qpos_list = data.get("query_position", [])
        #     shapes = [tuple(t.shape) for t in seq_list]
        #     print("[DEBUG][data] num_sequences=", len(seq_list), 
        #           " seq_shapes=", shapes, 
        #           " query_positions_len=", len(qpos_list),
        #           " query_positions=", qpos_list,
        #           " label=", data.get("label", None),
        #           " test_index=", data.get("test_index", None),
        #           flush=True)
        # except Exception as e:
        #     print(f"[WARN] Failed to inspect data: {e}", flush=True)
        #在取出前已经提前unsqeeze好了
        #这里是中转路口,这里的数据都是list[dict[int,str]]
        answers=generate(restricted_token,data,model=model,tokenizer=tokenizer,device=args.device,steps=args.steps,gen_length=args.gen_length,block_length=args.block_length,temperature=args.temperature,mode=args.mode,mask_id=args.mask_id)
        #回答总量:list[List[Dict]],第一个list是所有位置的回答,第二个list是每一个位置的回答,dict是各个字段的信息
        result.append(answers)
    return result


# def get_config_hash(cfg):
#     label_mapping_hash = md5(str.encode('-'.join(tuple(cfg.label_mapping.values())))).hexdigest()[:3]
#     template_hash = md5(str.encode(cfg.template)).hexdigest()[:3]
#     hash_str = label_mapping_hash + template_hash
#     return hash_str

# def inference_mode(model,dataset: DataLoader,restricted_token):
#     result=[]
#     for data in tqdm(dataset):

def main(corpus_config,args):
    cfg=easydict.EasyDict(corpus_config)
    print(cfg)


    # Debug: 打印 gen_length 并覆盖 mask_length
    try:
        old_mask_len = corpus_config.get('mask_length', None)
        print(f"[DEBUG] llada_main: args.gen_length={args.gen_length}, old mask_length in config={old_mask_len}")
    except Exception:
        pass
    corpus_config["mask_length"] = args.gen_length
    # 显式传递 mask_length，确保生效
    corpus=PromptCorpus(mask_length=args.gen_length, **cfg)



    #设置指令和数据集
    
    corpus_config["model"]=args.model
    corpus_config["temperature"]=args.temperature
    corpus_config["do_sample"]=args.do_sample
    corpus_config["topk"]=args.topk
    # 覆盖 mask_length 为命令行的 gen_length
    corpus_config["mask_length"] = args.gen_length
    
    dataset=DataLoader(corpus,batch_size=1,shuffle=False)

    model=init_model(args.model,args.device)

    # cfg_fname=os.path.split(args.config)[-1].replace(".yaml","")
    # cfg_hash_str=get_config_hash(cfg)

    print("start answering...")
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    if args.generate:
        # result=inference_mode(model,dataset,restricted_token_text)
        print("generate mode is still not implemented")
    else:
        #包括每一个数据集的结果,要把他们都输出出来
        result=inference_mode(args=args,tokenizer=tokenizer, model=model, dataset=dataset, restricted_token=corpus.restricted_token)
        # print(result)
    #     dump_fname=f"{cfg_fname}_{cfg.n_shot}_shot_{args.model}_seed{args.seed}_{cfg.sample_mode}_temperature({args.temperature}_top{args.topk}_hash({cfg_hash_str}).pkl"

    # output_ckpt={"result":result, "config":corpus_config}
    # pickle.dump(output_ckpt,
    #             open(os.path.join(args.output,dump_fname),'wb'))

    print("finish answering...")
    #这里主要是计算结果,然后再利用结果进行eval
    print("start evaluating...")
    #result=list[List[Dict]],每一个List是所有位置的回答,每一个Dict是每一个位置的回答
    gts=load_groundtruth(corpus_config['test_data_path'],args.config)
    eval(args.task,result,gts,args.output,args)


    print("finish evaluating...")

    


if __name__=='__main__':

    parser=argparse.ArgumentParser()
    #第一个应该是测试数据集的路径
    #这要是corpus的路径
    parser.add_argument("--config","-c",type=str,required=True)
    parser.add_argument("--model",type=str,default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--seed",type=int,default=1234)
    #默认4shot,但是由于这里标签很多,所以不一定只是4shot,但是尽量每个样本都拿出来
    parser.add_argument("--nshot","-n",type=int,default=1)
    parser.add_argument("--test_data_path",type=str,default="")
    parser.add_argument("--train_sample_mode",type=str,default="balance")

    parser.add_argument("--output","-o",type=str,default="default_output")
    parser.add_argument("--task",type=str,default="sst2")
    parser.add_argument("--gen_length",type=int,default=128)
    parser.add_argument("--steps",type=int,default=128)
    parser.add_argument("--block_length",type=int,default=128)
    #-----------------这些都是后期加的,不过可以加上去试试看---------------------
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--temperature",type=float,default=0.0)
    parser.add_argument("--mode",type=str,default="original")
    parser.add_argument("--mask_id",type=int,default=126336)
    parser.add_argument("--do_sample",action="store_true")
    parser.add_argument("--topk",type=int,default=-1)

    args = parser.parse_args()
    #刚开始的参数值,后面再进行修改
    print(args)

    random.seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #加载训练集和测试集的数据文件
    #从对应yaml中加载配置,返回一个dict
    corpus_config=yaml.safe_load(open(args.config))

    #这些可以从参数重收到进行转化
    #运行时改变数据路径
    if args.test_data_path:
        print(f"override test data path from {corpus_config['test_data_path']} to {args.test_data_path}")
        corpus_config['test_data_path'] = args.test_data_path

    #运行时改变
    if args.nshot > 0:
        print(f"override n-shot from {corpus_config['n_shot']} to {args.nshot}")
        corpus_config['n_shot'] = args.nshot

    #运行时转化
    if args.train_sample_mode is not None:
        print(f"override train data sample mode from {corpus_config['sample_mode']} to {args.train_sample_mode}")
        corpus_config["sample_mode"] = args.train_sample_mode
    
    #运行时转化
    # if args.gen_length >3:
    #     print(f"override mask length from {corpus_config['mask_length']} to {args.gen_length}")
    #     corpus_config["mask_length"] = args.gen_length
        
    main(corpus_config=corpus_config, args=args)

    


