import random, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import argparse
from tqdm import tqdm
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import query_extract, load_dataset, eval, group_by_label, corpus_sampling

# random.seed(1234)

#科研规则:有一个主的设置,其他都放在别的地方,调用就是分层的调用
#我打算在上面进行修改,加上位置这个参数
#分层的调用(以后就是一个scripts作为主要调用的脚本,并添加一系列的参数)
#记得封装generate脚本,方便调用,因为对于DLLM,generate有一个unmask再remask的过程,所以目前
#分base和instuct进行构造
#输入进来的指令是当作l,后面加了一维,变成b,l
#gpqa和其他的不一样,gpqa输出答案组合,而不是一个答案
def generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot=None,iscot=False,thread=None,mask_id=126336):
    #我需要去修改内部逻辑,因为原始generate是针对query放在最后的情况,所以并不需要去考虑加mdm_mask,但是目前需要加进去,因为我把query放在中间了
    #修改这个的逻辑,如果是gpqa那么就统一放在一个list依次计算答案即可
    #在gpqa这里,position多少不重要因为都要进行计算(后续如果真的需要再改)
    if task!='gpqa': 
            # for query_position in nshot:
            #prompts是一个list,包含所有位置的prompt
        #这里要根据nshot和question_query去构造更好的prompt,这里需要定期进行测量和评估
        query = query_extract(input,task,query_position,gen_length,nshot,iscot)

        if situation=='base':
            user_input = query
            # print("prompt: \n",user_input)
        elif situation=='instruct':
            m=[{"role":"user","content":query}]
            #这里之后还要修改逻辑,因为我不清楚用了apply之后会在哪个位置加上特殊的提示词
            user_input = tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
            # print("prompt: \n",user_input)
        prompt=tokenizer(user_input)['input_ids']
        #要让模型接收提示,一定要先将prompt用tokenizer转化为tokenID(input_ids),然后再转化为tensor并移动到模型所在设备才能推理
        #先转化为tensor方便使用
        prompt=torch.tensor(prompt).to(model.device).unsqueeze(0)
        # mask_token_id=tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        # prompt=[mask_id if token == mask_token_id else token for token in prompt]
        #这里要记了一下生成开始和结束范围
        # 找到mask token的位置
        mask_positions = (prompt == mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")
        # 找到第一个和最后一个mask token的位置
        first_mask_pos = mask_positions[1][0].item()
        last_mask_pos = mask_positions[1][-1].item()

        # prompt=prompt.to(model.device).unsqueeze(0)
        if mode=='original':
            from src.generate import generate
            out=generate(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence')
        elif mode=='fast_dllm':
            from src.generate import generate_with_fast_dllm
            out=generate_with_fast_dllm(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence',threshold=thread)[0]#现在需要的是第一个参数
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        #输出所有指令看一下,prompt+answer
        # print(tokenizer.batch_decode(out[:,:],skip_special_tokens=False)[0])
        #注意这个是放在最后才输出的,我们要稍微修改一下这里,变成换不同位置输出出来
        #默认解码的是第一维
        # answer = tokenizer.batch_decode(out[:,prompt.shape[1]:],skip_special_tokens=True)[0]
        answer=tokenizer.batch_decode(out[:,first_mask_pos:last_mask_pos+1],skip_special_tokens=False)[0]
    else:
        #task==gpqa,暂时不考虑用thread的情况
        answers=[]
        if nshot is not None:
            #list[str],str,并且都没有维度,后期要看清自己加维度
            prompts,correct_letter = query_extract(input,task,query_position,gen_length,nshot)
            for query in prompts:
                if situation=='base':
                    user_input = query
                elif situation=='instruct':
                    m=[{"role":"user","content":query}]
                    user_input=tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
                prompt=tokenizer(user_input)['input_ids']
                #对于每一个str加一个维度
                prompt=torch.tensor(prompt).to(model.device).unsqueeze(0)
                mask_positions = (prompt == mask_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) == 0:
                    raise ValueError("No mask tokens found in prompt")
                # 找到第一个和最后一个mask token的位置
                first_mask_pos = mask_positions[1][0].item()
                last_mask_pos = mask_positions[1][-1].item()
                if mode=='original':
                    from src.generate import generate
                    out=generate(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence')
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented.")
                #解码出为str
                # print(tokenizer.batch_decode(out[:,:],skip_special_tokens=False)[0])
                answer=tokenizer.batch_decode(out[:,first_mask_pos:last_mask_pos+1],skip_special_tokens=False)[0]
                answers.append(answer)
        else:
            raise NotImplementedError(f"task is gpqa but nshot is None")
    if task=='gpqa':
        return answers,correct_letter
    else:
        return answer

def main(args):
    random.seed(args.seed)
    task=args.task
    model_name=args.model_name
    device=args.device
    gen_length=args.gen_length
    steps=args.steps
    block_length=args.block_length
    temperature=args.temperature
    mode=args.mode
    data_path=args.data_path
    result_path=args.result_path
    #同时我这里也把位置作为一个参数,进行快速实验
    # query_position=args.query_position
    max_samples=args.max_samples
    nshot=args.nshot
    iscot=args.iscot
    thread = args.thread
    #开始全流程进行评估,先倒入数据集再导入模型,然后生成指令,最后进行推理得出答案进行评估
    #变成一个dataset,开始进行取值,根据query_position的位置来进行
    dataset=load_dataset(data_path,task,max_samples)
    #从标准数据集格式转化成python能处理的形式
    print('------------------Load model------------------')

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,local_files_only=True)
    #模糊匹配
    if 'LLaDA' in model_name:
        model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    else:
        model=AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    #进入评测环节
    model.eval()

    print('------------------Start Answering------------------')


    #这里要考虑的就是
    for query_position in range(nshot+1):
        results=[]
        correct_letters=[]
        for input in tqdm(dataset):
            #开始进行推理
            #让模型进行回答
            #这里开始要考虑到query_position的位置,因为不同的位置的准确率是有可能不一样的
            #这里我要做的就是都设置为相同的配置从而正确的进行评估
            if 'Instruct' in model_name:
                situation='instruct'
            else:
                situation='base'
            if task=='gpqa':
                #我想设置为各个位置的prompt,变成list,每一个list取出一个值来进行转化,这次我想根据query_position的位置来取值,按这个位置放置list,这样子就很方便记录
                #这个input是什么,能不能根据同样的input设置不同的prompt,只用在nshot的for循环下进行就可以了
                answer,correct_letter=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot)
                correct_letters.append(correct_letter)
            else:
                #暂时先不考虑其他的,我想先使用thread看看效果
                answer=generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot,iscot,thread)
            #最后的result应该是一个list,包含所有的回答
            #results可能是list[str]或者list[list[str]],如果是list[list[str]],第一个list是个数,第二个list是位置
            results.append(answer)
        #根据任务进行评测
        if task=='gpqa':
            eval(task,results,dataset,result_path,args,correct_letters)
        else:
            eval(task,results,dataset,result_path,args,position=query_position)

    print('-------------------Finish----------------')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='sudoku',choices=['sudoku','countdown','math500','gpqa','gsm8k','mbpp'])
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--gen_length',type=int,default=128)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--block_length',type=int,default=128)
    parser.add_argument('--temperature',type=float,default=0.0)
    parser.add_argument('--mode',type=str,default='original')
    parser.add_argument('--data_path',type=str,default='./data/sudoku.csv')
    parser.add_argument('--result_path',type=str,default='../results/sudoku_results')
    # parser.add_argument('--query_position',type=int,default=0)
    parser.add_argument('--max_samples',type=int,default=None)
    parser.add_argument('--nshot',type=int,default=None)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--iscot',type=bool,default=False)
    parser.add_argument('--thread', type=float, default=0.9)
    args=parser.parse_args()
    main(args)