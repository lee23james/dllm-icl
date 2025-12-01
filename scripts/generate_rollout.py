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
from utils.attention_utils import get_token_position,extract_token_attention_or_rollout
from model.modeling_llada import LLaDAModelLM
#重新进行推理来获得所有的rollout
#计算rollout,同时之前的也必须要用到带attentions的输出才行
#根据要求的间隔,记录等间隔step的所有层的attentiion score和最终rollout,方便后面进行可视化
def generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,steps_dis=3,mask_id=126336):
    #目前硬编码了sudoku的代码逻辑,等后续找到规律了再适配其他数据集(比如countdown)
    if task=='sudoku':
        query=query_extract(input,task,query_position,gen_length)
        if situation=='base':
            user_input=query
        elif situation=='instruct':
            m=[{"role":"user","content":query}]
            user_input=tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
        prompt=tokenizer(user_input)['input_ids']
        #对prompt进行升维处理,添加B这一维度
        #生成的是list[int]
        prompt_clone=prompt.copy()

        # print(f"\n[DEBUG] Decoded prompt:")
        # print(prompt)
        # print(tokenizer.decode(prompt, skip_special_tokens=False))
        # print(f"\n[DEBUG] Looking for mask_id={mask_id} in prompt...")

        prompt=torch.tensor(prompt).to(model.device).unsqueeze(0)
        #开始取顺序,针对位置进行特殊化,并且开始手机每一个attention_outpiut
        mask_positions=(prompt==mask_id).nonzero(as_tuple=True)
        # print(mask_positions)
        if len(mask_positions[0])==0:
            raise ValueError("No mask tokens found in prompt")
        #找到开始的位置
        first_mask_pos=mask_positions[1][0].item()
        last_mask_pos=mask_positions[1][-1].item()
        #开始进行取策略处理
        token_position=get_token_position(prompt_clone,tokenizer,first_mask_pos,query_position)
        if mode=='original':
            from src.generate import generate_with_attentions
            #返回的是list[list[tensor]],list[tensor]
            #第一个参数一个list是step数,另一个list是32层,每个tensor的形状是[S,S],第二个参数的list是步数
            _,attentions,rollouts,orders=generate_with_attentions(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence',output_attentions=True,return_order=True)
            # 开始进行 rollout 的计算
            #先进行计算
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        #开始进行魔改,这里记录的是没一步哪些得到转移了(累积结果,包括未转移的)并且这个对应位置是1代表转移了,0代表未转移
        orders_result=[]
        #我想的就是list[list[int]],第一个list是step数,第二个list是最近的3个step解码的token位置(小于等于),对应的是相对坐标
        determined=[]#记录的是总共有哪些token被转移(就是之前的),但是这个是相对位移
        top_3_result=[]#记录的是前面3个位置的注意力关注
        top_3_record=[]#记录的是前面3个位置的注意力关注的历史累积值,位置是list[list[int],第一个list是步数,第二个list是前3步关注的解码位置数
        # top_3_record_abs=[]#记录的是绝对位置
        #每一个元素代表的是每一步转移的的元素(也是累积结果,只不过不包括未转移的)
        for block_index in orders.keys():
            #每一个block块字段可能都包含多个步骤,需要step by step分析
            for step in range(len(orders[block_index])):
                top_3_sum=[]
                #包含
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
                #这个determined一次就会更新所有token的conf的值,记录历史累计值
                determined.extend([j for j in orders[block_index][step]])
                top_3_result.append(orders[block_index][step])
                if len(top_3_result)<3:
                    #好的这里面的每一步都是包含不大于3项的
                    # 扁平化：遍历每个子列表，提取所有整数
                    for sublist in top_3_result:
                        top_3_sum.extend(sublist)
                    top_3_record.append(top_3_sum)
                else:
                    # 扁平化：只取最近3步，并提取所有整数
                    for sublist in top_3_result[-3:]:
                        top_3_sum.extend(sublist)
                    top_3_record.append(top_3_sum)
        
        # 将 top_3_record 中的相对位置转换为绝对位置（加上 first_mask_pos）
        # top_3_record_abs = [[pos + first_mask_pos for pos in step_tokens] 
        #                     for step_tokens in top_3_record]
        
        attn_step=0
        rollout_list=[]
        #因为我这里要进行五个位置的计算,所以会有选择的对步数进行间隔输出(有些相距不会太大)
        #要记录特定步骤的layer分析结果
        # attention_ratio_step=[]
        #这里是对每一步进行有选择性的转化,第一个参数是list[tensor],其中list是32层,第二个参数是tensor
        for attention,rollout in zip(attentions,rollouts):
            #每隔几层进行转化,因为在相近step中变化不大,这里我可以在32步中取1
            attn_step+=1
            #筛选步骤
            if attn_step % steps_dis==0:
                #还是计算各层指标吧,方便后期处理
                #第i步
                layer_anaylse=[]
                layer_anaylse_sum=[]
                #这里是对每一层进行计算,并且每一个tensor都是[S,S]的形状,已经处理好了batch和head
                #精确到token级别的计算
                for layer in attention:
                    #dict[tensor,float],dict[str,tensor]
                    attention_dict,attention_sum_dict=extract_token_attention_or_rollout(
                        layer,#tensor[S,S]
                        (first_mask_pos,last_mask_pos),#tuple
                        token_position['question_token_positions'],#tuple
                        token_position['example1_token_positions'],#tuple
                        token_position['example2_token_positions'],#tuple
                        token_position['example3_token_positions'],#tuple
                        token_position['example4_token_positions'],#tuple
                        token_position['example5_token_positions'],#tuple
                        top_3_record[attn_step-1],#list[int],第一个list是步数,第二个list是前3步关注的解码位置数
                    )
                    #这个是每一层,方便进行层分析,所以每一层都要考虑纳入
                    layer_anaylse.append(attention_dict)#list[dict],每个dict是{token:float,attention_ratio:float}
                    layer_anaylse_sum.append(attention_sum_dict)#list[dict],每个dict是{token:float,attention_ratio:float}



                rollout_list.append({
                    "step": attn_step,#int
                    "rollout": rollout.tolist(),  # list,
                    # "attention": [layer.tolist() for layer in attention],  # list[list[list[float]]]
                    "first_mask_pos": first_mask_pos,#int
                    "last_mask_pos": last_mask_pos,#int
                    "position": query_position,#int
                    "token_position": token_position,  # dict
                    "layer_anaylse": layer_anaylse,  # list[dict]
                    # "layer_anaylse_sum": layer_anaylse_sum,  # list[dict]这个是粗粒度,不太方便进行更细致的观看
                })
                # if layer_analyse:
                #     for layer in attentions[i]:
                #         _,sum=extract_token_attention_or_rollout(layer,)
    else:
        raise NotImplementedError(f"other task is not implemented.")
    return rollout_list#list[dict],每个dict是{step:int,rollout:numpy array,attention:list[tensor]}
            
def save_rollout_params(rollout_list,task,position,index:int=None):
    """
    将 rollout 结果保存为 JSON 文件，便于后续可视化与分析。
    文件命名：rollout_params/{task}_{position}_{mode}[_{index}].json
    内容：[{step:int, rollout:List[List[float]], attention:List[List[List[float]]]}]
    """
    #将参数记录在相关文件夹里面,方便之后进行可视化,并且进行高度解耦
    # rollout_list: list[dict{step:int, rollout:numpy array or tensor, attention:list[tensor]}]
    import os,json
    assert isinstance(rollout_list,list),"rollout_list must be a list"
    #基本结构断言
    for i ,item in enumerate(rollout_list):
        assert isinstance(item,dict),f"第 {i} 项必须是 dict"
        assert "step" in item, f"第 {i} 项缺少 key: step"
        assert "rollout" in item, f"第 {i} 项缺少 key: rollout"
        if "attention" in item:
            attn = item["attention"]
            assert (attn is None) or isinstance(attn, list), f"第 {i} 项 attention 必须是 None 或 list"
        try:
            #将标准python对象转化为json对象
            json.dumps(item, ensure_ascii=False)
        except Exception as e:
            raise AssertionError(f"第 {i} 项不可 JSON 序列化: {e}")
    data = {"rollout_list": rollout_list}
    #index是数据集的序号,便于之后可以看清是什么样子的
    # filename=f"rollout_params/{task}_{position}_{index}.json"
    #project_root是项目根目录,params是params文件夹,rollout_params是rollout_params文件夹,用于拼接目标文件夹
    filename=os.path.join(project_root,"params","rollout_params",f"{task}_{position}_{index}.json")
    #创建目录,方便把文件存进去
    out_dir=os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    
    # 检查文件是否存在，如果存在则删除（避免覆盖时的潜在问题）
    import time
    if os.path.exists(filename):
        old_size = os.path.getsize(filename)
        old_mtime = os.path.getmtime(filename)
        print(f"警告: 文件 {filename} 已存在 (大小: {old_size/1024/1024:.2f}MB, 修改时间: {time.ctime(old_mtime)})，将被覆盖")
        os.remove(filename)
    
    # 添加时间戳到数据中，方便确认文件是新生成的
    data['metadata'] = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp': time.time()
    }
    
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    
    new_size = os.path.getsize(filename)
    print(f"已保存: {filename} (大小: {new_size/1024/1024:.2f}MB)")

    return filename

def main(args):
    task=args.task
    model_name=args.model_name
    device=args.device
    gen_length=args.gen_length
    steps=args.steps
    block_length=args.block_length
    temperature=args.temperature
    data_path=args.data_path
    #想要关注的数据集数量
    samples_num=args.samples_num
    #换一个角度(我现在想关注的是位置的变化)
    nshot=args.nshot
    #这个是收集的结果间隔是多少
    step_dis=args.step_dis
    # confidence_result={}#变成和位置相关的dict,key是位置,value是解码策略
    #目前只关注一个mode:orignal
    modes=['original']
    dataset=load_dataset(data_path,task)
    #取出前10个sample
    samples_num=min(samples_num,len(dataset))
    dataset=dataset[:samples_num]

    print('------------------ load model -----------------------')
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    # model_kwargs = {
    #         "trust_remote_code": self.trust_remote_code,
    #         "torch_dtype": self.torch_dtype,
    #         "local_files_only": self.local_files_only,
    #     }
    #选用自己的模型然后开始进行推理
    model=LLaDAModelLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True)
    model.to(device)

    print('--------------------start Answering----------------')

    #考虑各种位置
    for position in range(nshot+1):
        for mode in modes:
            print(f'------------------ start Answering with mode {mode} -----------------------')
            for index, input in enumerate(tqdm(dataset)):
                if 'Instruct' in model_name:
                    situation='instruct'
                else:
                    situation='base'
                #results就是rollout_list
                results=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,position,step_dis)
                if mode=='original':
                    #经典的每一步都在base的基础上生成
                    #复制一遍结果并进行保存,防止出现其他乱起八糟的东西
                    #结果是list[dict],每个dict是{step:int,rollout:numpy array,attention:list[tensor]等各种东西}
                    # results_clone=[t.clone() for t in results]
                    #保存参数到对应文件,方便之后进行图形化操作,这里的step_dis主要是压缩间隔,想完整也可以设置为1
                    save_rollout_params(results,task,position,index)
                    #在这里我还需要去取多个样本进行平均吗,其实我觉得可以
                    #算了,还是先不加了,先一个个取出来,找找规律,如果有结果,我再进行相加看看平均值即可
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented.")
    print('--------------------finish Answering----------------')

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
    parser.add_argument('--step_dis',type=int,default=5)
    #目前针对本服务器,只需要记录original即可,记录五个位置,总共32步骤(间隔5步)进行分析即可
    
    args=parser.parse_args()
    main(args)

