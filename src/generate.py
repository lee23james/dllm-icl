import torch
import numpy as np
import torch.nn.functional as F
import os, sys

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

BASE_LINE= None

def entropy_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    epsilon = 1e-12
    probs_safe = probabilities.clone() + epsilon
    entropy = -torch.sum(probabilities.clone() * torch.log(probs_safe), dim=-1)
    return entropy

#取出b,l,v,然后取出第一和第二个值进行相减去看差值,然后进而计算置信度差值
def margin_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    confidence = top1_probs - top2_probs
    return confidence

#制作交叉熵,方便后续计算困惑度
#可能用get_likelihood更加合适
def cross_entropy_function(probabilities, labels):
    
    # 1. 检查输入维度 (与您的风格一致)
    if probabilities.dim() != 3:
        raise ValueError("Input 'probabilities' must be 3D [batch_size, sequence_len, vocab_size]")
    if labels.dim() != 2:
        raise ValueError("Input 'labels' must be 2D [batch_size, sequence_len]")
    
    labels_expanded = labels.unsqueeze(-1)

    correct_token_probs = torch.gather(probabilities, dim=2, index=labels_expanded).squeeze(-1)


    epsilon = 1e-12
    probs_safe = correct_token_probs.clone() + epsilon


    entropy = -torch.log(probs_safe)
    
    return entropy

def load_baseline(model, baseline_name):
    global BASE_LINE
    if BASE_LINE is None:
        from utils.load_json_or_jsonl import load_json_or_jsonl
        p_baseline_dict = load_json_or_jsonl(baseline_name)
        token_num_ = p_baseline_dict['num_token']
        p_baseline_dict = p_baseline_dict['p_baseline_dict']
        del_keys = []
        for key in p_baseline_dict.keys():
            del_keys.append(key)
        for key in del_keys:
            p_baseline_dict[int(key)] = p_baseline_dict[key]
        for key in del_keys:
            del p_baseline_dict[key]
        for key in p_baseline_dict.keys():
            p_baseline_dict[key] = p_baseline_dict[key] / token_num_
        BASE_LINE = torch.full((126464,), 1/token_num_, device=model.device, dtype=torch.float32)
        keys = torch.tensor(list(p_baseline_dict.keys()), device=model.device, dtype=torch.long)
        values = torch.tensor(list(p_baseline_dict.values()), device=model.device, dtype=torch.float32)
        BASE_LINE.scatter_(0, keys, values)
    else:
        BASE_LINE = BASE_LINE.to(model.device)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    # print(f"[DEBUG] get_num_transfer_tokens: mask_index.shape={mask_index.shape}, steps={steps}", flush=True)
    mask_num = mask_index.sum(dim=1, keepdim=True)
    # 也可以再看一下中间量
    # print(f"[DEBUG] mask_num.shape={mask_num.shape}", flush=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)


    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        threshs[0]=-1
        sorted_confidence= torch.sort(confidence[j][mask_index[j]],diim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i==0 or top_i == len(threshs)-1:
            top_i+=1
        
        _,select_index = torch.topk(confidence[j],k=top_i)
        transfer_index[j,select_index]=True
    
    return x0, transfer_index


#这里其实是加装mask_id方便进行解码和重掩码的,如果要改位置,这里也需要改
#这里需要详细了解prompt是在哪个地方生成,在哪个地方结束

#这里输入进去的貌似就是tokenid了,不再是prompt了
#我先在需要改进一下我这个底层逻辑,我需要加一下限制词汇表,如果有这个说明我的生成区域很小,如果没有的话那就说明生成区域很大
#如果是None的话那就按之前的想法继续,如果不是None的话先提取答案然后再看看是否在限制词语中,如果不在还得从logits中找到概率最大的词语去返回答案
@torch.no_grad()
def generate(model, prompt, gen_start,steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):

    
    x = prompt.clone().to(model.device)


    # x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    if return_order:
        orders = {}
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, gen_start + num_block * block_length:gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append((select_index-gen_start).tolist())
            x[transfer_index] = x0[transfer_index]
    #orders这里是一个dict虽然包含的是每一块,每一步转移的位置,但是并没有进行全排列
    # 就是把一个大步转化为一个小步然后进行分析         
    if return_order:
        return x, orders        
    return x


@torch.no_grad()
def forward(model, prompt, gen_start,steps=3, gen_length=3, block_length=3, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
   #复制prompt送到device再计算logits
    x = prompt.clone().to(model.device)
    # x[:, :prompt.shape[1]] = prompt.clone()
    # 形状校正：允许 [B, L] 或 [B, L]，其余直接报错
    assert x.dim() == 2, f"x must be 2D , got {x.shape}"  # [B, L]
    
    # 维度断言
    B, L = x.shape
    assert 0 <= gen_start < L, f"gen_start out of range: gen_start={gen_start}, L={L}"
    assert block_length > 0, f"block_length must be > 0, got {block_length}"
    assert gen_start + block_length <= L, f"slice out of range: gen_start={gen_start}, block_length={block_length}, L={L}"

    prompt_index = (x != mask_id)
    if return_order:
        orders = {}
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    #num_blocs一般是1(在我们的计算过程中,所以可以忽略)
    
    # step_probability_records=[]
    # analysis_entry=None

    for num_block in range(num_blocks):
        block_mask_index = (x[:, gen_start + num_block * block_length:gen_start + (num_block + 1) * block_length] == mask_id)
        
        assert block_mask_index.dim() == 2, f"block_mask_index must be 2D, got {block_mask_index.shape}"
        assert block_mask_index.size(1) == block_length, f"unexpected block length: {block_mask_index.shape}, expected {block_length}"

        #调试打印
        # print(f"[DEBUG] x.shape={x.shape}, gen_start={gen_start}, block_length={block_length}, num_block={num_block}", flush=True)
        # print(f"[DEBUG] block_mask_index.shape={block_mask_index.shape}, steps={steps}", flush=True)

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        #一般steps=3,每一步解码不同的位置,但是一般我们只要最前面的,所以一个简单的方法就是记录位置就可以了,设置一个dict,取最前面的dict相当于最靠近的答案取出来
        
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                #p现在集中了所有词汇的概率(包括受限词汇和非受限词汇),目前它的格式是[batch_size,sequence_len,vocab_size]
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append((select_index-gen_start).tolist())
            x[transfer_index] = x0[transfer_index]       
    if return_order: 
        return x, orders        
    return x

#先不考虑添加顺序,只考虑输出注意力分数矩阵
#我认为我还是希望在底层就返回的是list列表,这样子防止内存过大
#这里还是需要返回一下解码顺序(top3)和conf值,进行一些系统的分析(top2)
@torch.no_grad()
def generate_with_attentions(model, prompt, gen_start,steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, output_attentions:bool=False,return_order:bool=False):
    """
    根据attention_outpout是否为true来返回attention score方便后面进行注意力分数和注意力矩阵的计算
    其他和原始的基本没变,后期考虑合并到原来的函数中进行attention功能的实现
    """        
    #注意因为prompt的原因,天然输入进去的就必须是ids,并且包含B这个维度
    x = prompt.clone().to(model.device)


    # x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    if return_order:
        orders = {}
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    #针对每一个block开始进行计算
    #这里的steps已经进行除块处理了,相当于是原来的1/num_block
    #收集所有的步骤中的attention
    if output_attentions:
        #记录的是每一步所包含的量
        attention_step_list=[]
        rollout_step_list=[]
    for num_block in range(num_blocks):
        block_mask_index = (x[:, gen_start + num_block * block_length:gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        #针对每一步开始进行计算(每一块内的一步)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                #cfg>0不适配传递atten(可能是batch维度不匹配,比较困难)
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                
            else:
                outputs=model(x,output_attentions=output_attentions)#在这里让模型输出权重
                logits = outputs.logits
                #ouput.attentions是一个tuple,每个元素是一个tensor,tensor的形状是[batch_size,num_heads,seq_len,seq_len],tuple中的每一个元素代表一层注意力权重
                if output_attentions and outputs.attentions is not None:
                    #先不保存先,我想先存在一个列表里面总共是32层,后面再进行直接计算
                    #针对的是每一步
                    #这个事记录一步中的每一层
                    attn_list=[]
                    S=x.shape[1]
                    rollout=torch.eye(S,device=x.device,dtype=torch.float32)
                    #处理attentions的每一层元素
                    #我想在这里顺便就把rollout计算出来,因为这个事tensor,实在是太大了
                    for attn in outputs.attentions:
                        #处理不同的格式
                        #如果attn是tuple,则取第一个元素
                        if isinstance(attn,tuple):
                            attn=attn[0]
                            #[B,H,S,S]
                        if attn.dim()==4:
                            attn=attn[0]#降维,取第一个batch
                        #[H,S,S],目前的结构是list[tensor[H,S,S]],每个tensor.dim=3
                        if attn is not None:
                            #在这里直接处理attn变成[S,S]的tensor,这是每一层的注意力分数
                            attn=attn.mean(dim=0)
                        else:
                            raise ValueError("attn is None")
                        A=attn.to(torch.float32)
                        A=A+torch.eye(S,device=A.device,dtype=A.dtype)
                        #归一化
                        A=A/(A.sum(dim=-1,keepdim=True)+1e-9)
                        rollout=torch.matmul(rollout,A)
                        #append的时候最好还是numpy,等到转化成json再变成list
                        #list[numpy array],每个numpy array的形状是[S,S],总共32层
                        attn_list.append(attn.detach().cpu())
                        #这里已经对于attention进行了处理,把tuple[tensor(B,H,S,S)]变成了list[tensor(H,S,S)]
                    rollout_step_list.append(rollout.detach().cpu())
                    # attn_list=process_attention_weights(attention_list)
                    #这里是一步,记录下当前的一步的attention,总共32层,list[list[numpy array]]
                    attention_step_list.append(attn_list)
                elif output_attentions:
                    rollout_step_list.append(None)
                    attention_step_list.append(None)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                #取出steps,返回的是dict[int,list[int]]      
                if return_order:
                    if num_block+1 not in orders:
                        #orders是一个dict,key是block,记录的是转移的位置,value是list[int],记录的是转移的位置
                        orders[num_block+1] = []
                    #这里记录的是转移的位置(但是我必须要考虑prompt的影响)
                    #select_index有可能一次生成多个,所以也必须要考虑进去
                    #这里我进行了魔改:记录的是每一块,每一步转移的position,然后之后我转化成每一个每一步是哪些posiition进行了解码
                    #这里记录的是相对坐标
                    orders[num_block+1].append((select_index-gen_start).tolist())
            x[transfer_index] = x0[transfer_index]
    #这里我也要考虑conf
    #第一个list是步数,第二个list是32层,方便进行层分析
    if output_attentions and not return_order:
        return x,attention_step_list,rollout_step_list#list[tensor]],list[tensor]
    #进行token级别的细粒度分析
    elif output_attentions and return_order:
        return x,attention_step_list,rollout_step_list,orders
    else:
        return x


#这里是计算logits差值和熵的,想探究的是全局的变化和解码的顺序
#我打算整理出总体和局部,一点一点进行分析
#我想把答案也记录方便我出表,比对准确率和熵值的关系
@torch.no_grad()
def generate_with_conf(model, prompt, gen_start,steps=128, gen_length=128, block_length=128, temperature=0.,
            cfg_scale=0., remasking='low_confidence', mask_id=126336, 
            return_order=False,return_conf_diff=False,return_entropy=False,return_token_change=False,return_conf=False):    
    x = prompt.clone().to(model.device)
    # x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    if return_order:
        orders = {}
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    # 收集每一步的置信度差值和熵值、
    #这里的值应该是list[list[float]],第一个list是表示的是mask区域各个位置,第二个list展示的是每一步的差值,如果后期变为0了就说明已经被解码了
    #这样子同时反映了解码顺序和conf差值还有熵值的变化(针对blocklength=genlength)
    confidence_margin_list = []  # 每一步的margin值
    entropy_list = []             # 每一步的熵值
    prev_top1_tokens=None
    token_change_list=[]#这里考虑的是true/false
    conf_list=[]#这里考虑的是每一步的conf值
    for num_block in range(num_blocks):
        #先是对当前Block块内进行mask处理,后面的block不关心
        block_mask_index = (x[:, gen_start + num_block * block_length:gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            #结束完新的一轮转换,现在的x又是上一轮所有mask解码一部分剩余的(一部分在这里是1)
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            #对概率加噪
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)#b,l,记录解码后的最大概率的token索引
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)#b,l,vocab_size,记录的是每个位置所有可能token的概率
                #x0_p是b,l,记录的是所选token的置信度(先升维再降维)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)


                # 计算所有可预测位置的margin和entropy
                # p的形状是[batch, seq_len, vocab_size]
                if return_conf_diff or return_entropy:
                    # 计算margin（置信度差值）
                    if return_conf_diff:
                        #返回b,l的结构,并且值是对应的置信度差值,表示的是每一步的置信置信度差值
                        #目前这一步还没有经过mask,也就是说生成的是包含部分的mask
                        step_margin = margin_function(p.unsqueeze(0) if p.dim() == 2 else p)  # [batch, seq_len]
                        # 只保留mask位置的值（可预测的位置），其他位置设为nan
                        #这里先对之后的block的置信度差值应该保留,因为这里并不是选值,而是对所有位置探究影响
                        step_margin_masked = torch.where(mask_index, step_margin, -np.inf)
                        #是不是保存为numpy后面再说
                        #每次手机进入表哥前先要切片,,来计算答案部分的置信度差值
                        step_margin_trimmed=step_margin_masked[:,gen_start:gen_start+gen_length]
                        confidence_margin_list.append(step_margin_trimmed[0].detach().cpu().to(torch.float32).numpy())
                    
                    # 计算entropy（熵值）
                    if return_entropy:
                        step_entropy = entropy_function(p.unsqueeze(0) if p.dim() == 2 else p)  # [batch, seq_len]
                        # 只保留mask位置的值（可预测的位置），其他位置设为nan
                        step_entropy_masked = torch.where(mask_index, step_entropy, -np.inf)
                        step_entropy_trimmed=step_entropy_masked[:,gen_start:gen_start+gen_length]
                        entropy_list.append(step_entropy_trimmed[0].detach().cpu().to(torch.float32).numpy())
                #贪婪解码才能这样,否则不好记录
                if return_token_change and temperature==0.:#贪婪解码
                    #获取当前步,这里需要是贪婪的情况,因为当会变的时候不太好处理,并且是在这次mask的位置和上一次比
                    #如果上一次没有,就先记录所有mask的logits,然后当前bool选择false,等第二步开始再进行记录
                    current_top1_tokens=torch.argmax(logits,dim=-1)#b,l
                    #因为mask_id是越来越少,主要考虑的是mask的部分和上一个部分是否改变,解码的记为false即可
                    #每一次都是加噪的x0进行传递,不过倒也不在乎,因为最后它只会变成1个已经传递并且解码好的
                    #所以第一步是先对上一步一景解码好的进行false处理,然后对mask部分再比对logits看看是否和上一部分是否有差别

                    if prev_top1_tokens is not None:
                        #那就先记录变成当前的情况
                        #先记录当前和过去相不相等的情况
                        #这里是整段记录,还包括prompt和后面部分
                        token_changed=(current_top1_tokens!=prev_top1_tokens)
                        #这里记录的是只在当前mask的token才记录变化的情况,其他位置一律false处理,也不在乎是否已经被unmask,因为解码有别的记录
                        token_change_masked=torch.where(mask_index,token_changed,torch.tensor(False,device=token_changed.device))
                        #先进行answer切片
                        token_change_trimmed=token_change_masked[:,gen_start:gen_start+gen_length]#目前是答案区域的变化部分,
                        token_change_list.append(token_change_trimmed[0].detach().cpu().to(torch.float32).numpy())
                    else:
                        #第一步,没有上一步可以比较，全部设为False,每一步都是所有位置的bool
                        token_change_masked = torch.where(mask_index, torch.tensor(False, device=x.device), torch.tensor(False, device=x.device))
                        #第一步也要先进行切片
                        token_change_trimmed=token_change_masked[:,gen_start:gen_start+gen_length]
                        token_change_list.append(token_change_trimmed[0].detach().cpu().to(torch.float32).numpy())
                    
                    #比较完之后再进行记录这次成果,因为下一轮的mask是会变的
                    prev_top1_tokens=current_top1_tokens.clone()
                


            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                # random模式不计算真实的margin和entropy
                if return_conf_diff:
                    confidence_margin_list.append(None)
                if return_entropy:
                    entropy_list.append(None)
                if return_token_change:
                    #暂时不记录
                    token_change_list.append(None)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            #这个是关键比对的就是平均熵值和平均置信度的关系
            if return_conf:
                #记录的是每一步mask地方的置信度
                conf=confidence.clone()
                conf=conf[:,gen_start:gen_start+gen_length]
                conf_list.append(conf[0].detach().cpu().to(torch.float32).numpy())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append((select_index-gen_start).tolist())
            x[transfer_index] = x0[transfer_index]
    
    #orders这里是一个dict虽然包含的是每一块,每一步转移的位置,但是并没有进行全排列
    # 就是把一个大步转化为一个小步然后进行分析
    # 根据需要返回不同的值
    result = [x]
    
    if return_order:
        result.append(orders)
    
    if return_conf_diff:
        result.append(confidence_margin_list)
    
    if return_entropy:
        result.append(entropy_list)
    
    if return_token_change:
        result.append(token_change_list)
    
    if return_conf:
        result.append(conf_list)
    
    # 如果只有x，直接返回x；否则返回元组
    if len(result) == 1:
        return x
    else:
        return tuple(result)

#=============dllm====================
@ torch.no_grad()
def generate_with_fast_dllm(model,prompt,gen_start,steps=128,gen_length=128,block_length=128, temperature=0.,
remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = prompt.clone().to(model.device)

    # x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe=0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, gen_start + num_block * block_length:gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe+=1
            mask_index=(x==mask_id)
            logits=model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x,nfe

#这个是之前利用当前置信度,取出最高的conf进行综合分数的计算
@torch.no_grad()
def generate_with_pc_sampler(model, prompt, gen_start,steps=128, gen_length=128, block_length=128, lambd=1, alpha=1, baseline_name='P_baseline.json', temperature=0.,
                  cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
    
    global BASE_LINE
    if BASE_LINE is None:
        load_baseline(model, baseline_name)
    if return_order:
        orders = {}
    
    x=prompt.clone().to(model.device)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        
        block_mask_index = (x[:, gen_start + num_block * block_length: gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            x0_p = pc_sampler_function(
                probabilities=x0_p[:, gen_start:gen_start+gen_length],
                token_ids=x0[:, gen_start:gen_start+gen_length],
                lambda_val=lambd,
                alpha=alpha,
                bg_freq_tensor=BASE_LINE
            )
            
            confidence = torch.where(mask_index[:, gen_start:gen_start+gen_length], x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index+gen_start] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x


#这里记录一下pcsampler,同时加上自己的部分,他是根据当前token的置信度进行计算,我现在像的就是,我能不能换一种方式
def pc_sampler_function(
    probabilities: torch.Tensor,#每个位置的概率(就是置信度)
    token_ids: torch.Tensor,
    lambda_val: float,
    alpha: float,
    bg_freq_tensor: torch.Tensor
) -> torch.Tensor:
    
    if probabilities.shape != token_ids.shape:
        raise f"probabilities.shape: {probabilities.shape}, token_ids.shape: {token_ids.shape} must be equal"

    device = probabilities.device
    sequence_len = probabilities.shape[1]
    f_bg_tensor = bg_freq_tensor[token_ids]
    epsilon = 1e-9
    cross_entropy_scores = -probabilities * torch.log(f_bg_tensor + epsilon)
    cross_entropy_scores = torch.clamp(cross_entropy_scores, max=alpha)
    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = torch.exp(-lambda_val * positions)
    final_scores = positional_bias * cross_entropy_scores

    return final_scores

#看看效果是不是更加出色,这里对PCsampler进行改进,,计算的是之前的平均置信度的值的高低,进而去判断哪边更好
#这里我真正想要替换的是conf(在哪里是准确率,但是我想要利用的是平均概率进行计算,而且计算的是之前的平均值)
#第一个list是步骤,第二个list是所有层数的平均值,所以我现在的想法就是计算看看能不能提点
#这里处理了维度信息,本质还是取出第一个维度进行计算,因为并没有填充
def sampler_with_conf(
    conf: list[torch.Tensor],#这个原来是概率,但是我需要去记录的是之前的平均置信度,这样子比较好改造,这里我想要利用的是整一块的平均置信度进行计算,相当于对他们的结构进行修改
    token_ids: torch.Tensor,#token id(方便进行学习)
    lambda_val: float,#调整解码顺序,方便之后进行解码
    alpha: float,#防止给分到离谱
    bg_freq_tensor: torch.Tensor#背景频率,方便进行打分
)->torch.Tensor:#这里得出的应该是综合分数,借此在进行计算
    """
    使用历史步骤的置信度累计平均值来计算采样分数
    
    Args:
        conf: 所有历史步骤的置信度列表，每个元素是 [sequence_len] 或 [batch_size, sequence_len] 的 tensor
        token_ids: 当前步骤的 token id，形状与 conf 中每个元素相同
        lambda_val: 位置偏置参数
        alpha: 交叉熵分数的上限
        bg_freq_tensor: 背景频率 tensor，用于计算交叉熵
    
    Returns:
        final_scores: 综合分数 tensor，形状与 token_ids 相同
    """
    if len(conf) == 0:
        raise ValueError("conf list cannot be empty")
    
    # 获取设备信息（从第一个 conf tensor 或 token_ids）
    device = conf[0].device if isinstance(conf[0], torch.Tensor) else token_ids.device
    
    # 确保所有 conf tensor 的形状一致
    reference_shape = conf[0].shape
    for i, conf_tensor in enumerate(conf):
        if conf_tensor.shape != reference_shape:
            raise ValueError(f"All conf tensors must have the same shape. "
                           f"conf[0].shape: {reference_shape}, conf[{i}].shape: {conf_tensor.shape}")
    
    # 处理 batch 维度
    # conf_list 中每个元素可能是 [sequence_len] 或 [batch_size, sequence_len]
    # token_ids 可能是 [sequence_len] 或 [batch_size, sequence_len]
    has_batch_dim = len(reference_shape) == 2
    
    if has_batch_dim:
        # 如果 conf 有 batch 维度，取第一个样本
        conf = [c[0] for c in conf]
        reference_shape = conf[0].shape  # 更新为 [sequence_len]
    
    # 处理 token_ids 的 batch 维度
    if len(token_ids.shape) == 2:
        # token_ids 有 batch 维度，取第一个样本用于计算
        token_ids_for_calc = token_ids[0]
        batch_size = token_ids.shape[0]
    else:
        # token_ids 没有 batch 维度
        token_ids_for_calc = token_ids
        batch_size = None
    
    # 确保 token_ids_for_calc 的形状与处理后的 conf tensor 一致
    if token_ids_for_calc.shape != reference_shape:
        raise ValueError(f"token_ids.shape: {token_ids_for_calc.shape} must match conf tensor shape: {reference_shape}")
    
    # 将所有历史步骤的 conf 堆叠成一个 tensor
    # conf[i] 是 [sequence_len]，堆叠后是 [num_steps, sequence_len]
    conf_stack = torch.stack(conf, dim=0)  # [num_steps, sequence_len]
    
    # 计算每个位置在所有历史步骤中的累计平均值
    # 沿着步骤维度计算平均值
    avg_conf = torch.mean(conf_stack, dim=0)  # [sequence_len]
    
    # 使用累计平均值替换 pc_sampler_function 中的 probabilities
    # 后续计算逻辑与 pc_sampler_function 相同
    sequence_len = avg_conf.shape[-1]  # 获取序列长度
    f_bg_tensor = bg_freq_tensor[token_ids_for_calc]  # [sequence_len]
    epsilon = 1e-9
    cross_entropy_scores = -avg_conf * torch.log(f_bg_tensor + epsilon)
    cross_entropy_scores = torch.clamp(cross_entropy_scores, max=alpha)
    
    # 计算位置偏置
    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = torch.exp(-lambda_val * positions)
    
    final_scores = positional_bias * cross_entropy_scores  # [sequence_len]
    
    # 如果原始 token_ids 有 batch 维度，需要将结果扩展到 batch 维度
    if batch_size is not None:
        # 将 [sequence_len] 扩展为 [batch_size, sequence_len]
        # 所有 batch 使用相同的分数（因为 conf_list 只记录了第一个样本）
        final_scores = final_scores.unsqueeze(0).expand(batch_size, -1)  # [batch_size, sequence_len]
    
    return final_scores

#利用我修改的代码结构来计算生成的置信度,进而看看结果是否可靠
#记住prompt刚输入进去就是token id了(因为需要探究的是位置)
#目前默认gen_length=block_length,之后再进行调整
@torch.no_grad()
def generate_with_conf_sampler(model, prompt, gen_start,steps=256, gen_length=256, block_length=256, lambd=1, alpha=1, baseline_name='P_baseline.json', temperature=0.,
                  cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False, num=None):#默认Num=None
    global BASE_LINE
    if BASE_LINE is None:
        load_baseline(model,baseline_name)
    if return_order:
        orders={}
    #这里的x和promot都是token id,不需要进行转换
    x = prompt.clone().to(model.device)
    prompt_index=(x!=mask_id)
    if return_order:
        orders={}
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    for num_block in range(num_blocks):
        # 每个 block 开始时重置 conf_list，只记录当前 block 的历史步骤
        conf_list = []
        
        # 修复：使用 gen_start 而不是 prompt.shape[1]
        block_mask_index = (x[:, gen_start + num_block * block_length: gen_start + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens=get_num_transfer_tokens(block_mask_index,steps)
        for i in range(steps):
            mask_index=(x==mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0=torch.argmax(logits_with_noise,dim=-1)#b,l

            if remasking=='low_confidence':
                p=F.softmax(logits,dim=-1)
                x0_p=torch.squeeze(
                    torch.gather(p,dim=-1,index=torch.unsqueeze(x0, -1)), -1)#b,l
            elif remasking=='random':
                # 修复：拼写错误 shapr -> shape
                x0_p=torch.rand((x0.shape[0],x0.shape[1]),device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, gen_start + (num_block + 1) * block_length:] = -np.inf
            #该mask的地方进行转化
            x0=torch.where(mask_index,x0,x)
            # 记录当前步骤的置信度（包含所有位置，包括非 mask 位置）
            # 虽然记录了所有位置，但最后会通过 mask_index 过滤，只保留 mask 位置的分数
            conf=x0_p.clone()
            conf=conf[:,gen_start:gen_start+gen_length]
            #这里记录每一个步骤的conf[0]
            conf_list.append(conf[0])
            #这里已经进行切片处理
            #根据 num 参数决定使用多少个历史步骤
            #如果 num=None，使用全部步骤；如果 num 不为 None，使用最近 num 个步骤（不足则取全部）,可以通过参数进行调教
            if num is None:
                conf_list_for_sampler = conf_list
            else:
                # 取最近 num 个步骤，如果不足 num 个则取全部
                conf_list_for_sampler = conf_list[-num:] if len(conf_list) > num else conf_list
            x0_p= sampler_with_conf(
                conf=conf_list_for_sampler,
                token_ids=x0[:,gen_start:gen_start+gen_length],
                lambda_val=lambd,
                alpha=alpha,
                bg_freq_tensor=BASE_LINE
            )
            confidence=torch.where(mask_index[:,gen_start:gen_start+gen_length],x0_p,-np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                #当成整片处理,暂时不考虑半自回归形态
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index+gen_start] = True #这里进行切片处理
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x