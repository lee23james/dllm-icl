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
from datetime import datetime
from utils.eval_utils import query_extract, load_dataset
from utils.attention_utils import get_token_position,find_answer_token_positions,find_token_sequence_position,find_last_token_sequence_position,find_zero_position,find_countdown_answer_token_positions,find_gsm8k_answer_token_positions,find_math_answer_token_positions,find_mbpp_answer_token_positions
from utils.conf_utils import cal_token_change_positive_total,cal_accmulate_conf,paint_conf,cal_accmulate_conf_countdown,cal_accmulate_conf_gsm8k,cal_accmulate_conf_math,cal_accmulate_conf_mbpp
from utils.eval_utils import eval
#现在的首要任务就是扩展到countdown
#我要在这里进行扩展,去查看nshot中各种顺序和熵的关系,并且扩展到4个数据集上,进行应用,最后就是取出50个数据集做少量应用来去分配更多的icl(这就是探究的性质之一)
def generate(model,tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,query_position,nshot,mask_id=126336,return_current_conf:bool=False):
    #对于任务的类型进行详细的分类
    if task!='gpqa':
        #提前先进行答案区域的提取,方便后续进行计算,这个是针对sudoku需要提前处理数据集
        # if task=='sudoku':
        #     zero_position_list=find_zero_position(input['Puzzle'])
        #这里根据nshot和question去寻找最合适的组合(默认固定prompt的位置,因为这会造成其他问题)
        query=query_extract(input,task,query_position,gen_length,nshot)
        if situation=='base':
            user_input=query
        elif situation=='instruct':
            m=[{"role":"user","content":query}]
            user_input=tokenizer.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
        prompt=tokenizer(user_input)['input_ids']
        #这里先加上一个维度
        # prompt_clone=prompt.copy()
        prompt=torch.tensor(prompt).to(model.device).unsqueeze(0)
        mask_positions=(prompt==mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0])==0:
            raise ValueError("No mask tokens found in prompt")
        #找到开始的位置
        first_mask_pos=mask_positions[1][0].item()
        last_mask_pos=mask_positions[1][-1].item()
        #开始进行取策略处理
        # token_position=get_token_position(prompt_clone,tokenizer,first_mask_pos,query_position)
        if mode=='original':#其他的我成熟了再考虑
            #我感觉只用看答案回答区域即可,因为其他的不怎么去看,然后关注的是标的是哪几个0对应的部分即可
            from src.generate import generate_with_conf
            #现在最重要的是同一时刻进行测评
            x,orders,token_change,conf=generate_with_conf(model,prompt,first_mask_pos,steps,gen_length,block_length,temperature,cfg_scale=0.,remasking='low_confidence',
            return_order=True,return_conf_diff=False,return_entropy=False,return_token_change=True,return_conf=True)
            #我想一下就传递这几个数据即可,然后还要把处理一下坐标格式这些都传递回去
            #现在我想变成y轴是token,x轴是step然后依次展开
            #返回的已经切片完成了,下面直接处理即可
            
            orders_result=[]
            determined=[]
            for block_index in orders.keys():
                for step in range(len(orders[block_index])):
                    this_step = torch.full((1,gen_length),0,dtype=torch.float16)
                    #先记录当前步哪些被转移了
                    this_step[0,orders[block_index][step]]=1
                    if determined!=[]:
                        this_step[0,determined]=1
                    orders_result.append(this_step)
                    #添加转移的元素(记录的是之前转移的元素,并且针对的是所有的token)
                    determined.extend([j for j in orders[block_index][step]])
            #这里orders_result记录的是list[tensor([1, gen_length])],描绘的是每一步里面有都是0的mask部分,哪些是1(已解码的)
            #先要处理的是答案区域和回答对应0的区域都要标注出来,目前返回的dict,根据这个可以画图
            #这里返回的是tensor,全部是token id
            x_answer=x[:,first_mask_pos:last_mask_pos+1]
            # print(f"x_answer: {x_answer[0]}")
            # x_answer_decode=tokenizer.decode(x_answer[0],skip_special_tokens=False)
            # print(f"x_answer_decode: {x_answer_decode}")
            #返回的数据集的字段是不一样的
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
                #这里就是取出[begin][done]之间的坐标看看重合度
                answer_token_positions=find_mbpp_answer_token_positions(x_answer[0],tokenizer)
            else:
                raise NotImplementedError(f"Task {task} not implemented.")
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        answer=tokenizer.batch_decode(x[:,first_mask_pos:last_mask_pos+1],skip_special_tokens=False)[0]
    else:
        raise NotImplementedError(f"gpqa not implemented.")
    #现在要返回答案进行测评,为例后续画图,比对正确率和这些的关系
    #返回的是解码之后的结果,不返回熵值了,保持一致性
    # return answer,orders_result,conf_diff,entropy,token_change,conf,answer_token_positions
    if return_current_conf:
        current_conf=get_decoded_conf_list(conf,orders,gen_length)
        # return answer,orders_result,token_change,conf,answer_token_positions,current_conf
    else:
        current_conf=None
    return answer,orders_result,token_change,conf,answer_token_positions,current_conf
    # return answer,orders_result,token_change,conf,answer_token_positions

#计算当前解码一步的conf累积值
def get_decoded_conf_list(conf,orders,gen_length):
    """
    conf: list of numpy arrays,每个元素是[gen_length]的numpy数组
    orders: dict,每个元素是[block_idx]的list,每个list是[step_indices]的list
    gen_length: int,生成序列的长度
    """
    #初始化
    token_decoded_conf=np.zeros(gen_length,dtype=np.float32)
    global_step=0
    for block_idx in sorted(orders.keys()):
        block_steps=orders[block_idx]
        for step_indices in block_steps:
            if global_step>=len(conf):
                break
            current_step_map=conf[global_step]
            for pos in step_indices:
                if 0 <= pos < gen_length:
                    token_decoded_conf[pos] = current_step_map[pos]
            global_step+=1
    #返回numpy array数组
    return token_decoded_conf
        


#防止出现各种numpy或者tensor类型
def convert_to_json_serializable(obj):
    """
    递归地将对象转换为JSON可序列化的格式
    处理numpy类型、nan、inf等特殊值
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None  # JSON不支持nan，转换为None
        elif np.isinf(obj):
            return None  # JSON不支持inf，转换为None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        # 处理Python原生类型，但需要检查float是否为nan或inf
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        return obj
    else:
        # 其他类型尝试转换为字符串
        return str(obj)

#这里我后期要改,因为现在想要保存的是另外的东西,或者说绘图的时候还是想分清楚一点
#这个是所有样本的的总体结果,其中我需要添加的是位置和nshot数量,方便进行观察
def save_results_to_json(all_samples_result, sample_data, task, position,  nshot,gen_length,steps,block_length, accuracy=None, output_dir=None):
    """
    保存all_samples_result和sample_data到JSON文件
    
    Args:
        all_samples_result: 所有样本的统计结果
        sample_data: 每个样本的详细数据
        task: 任务名称
        position: 位置索引
        mode: 模式
        output_dir: 输出目录，如果为None则使用默认目录
    """
    #保存的是总体数据和各个样本的数据
    if output_dir is None:
        output_dir = os.path.join(project_root, 'params', 'conf_params', task, f'nshot_{nshot}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为JSON可序列化格式
    # 把总体计算结果进行转换(求总体的平均值)
    all_samples_result_serializable = convert_to_json_serializable(all_samples_result)
    # 把个体结果进行转化,其中个体结果包含各个样本的总体计算平均值,还有每个样本的部分token在全部的比例(暂时先不存储单个情况了,因为已经观察过了)
    # sample_data_serializable = convert_to_json_serializable(sample_data)
    
    # 构建保存的数据结构
    if accuracy is not None:
        save_data = {
            'all_samples_result': all_samples_result_serializable,
            # 'sample_data': sample_data_serializable,
            'metadata': {
                'accuracy': accuracy,
                'task': task,
                'position': position,
                'nshot': nshot,
                'num_samples': len(sample_data.get('result', [])),#记录总共的样本数量
                'gen_length': gen_length,
                'steps': steps,
                'block_length': block_length,
            }
        }
    else:
        save_data = {
            'all_samples_result': all_samples_result_serializable,
            # 'sample_data': sample_data_serializable,
            'metadata': {
                'task': task,
                'position': position,
                'nshot': nshot,
                'num_samples': len(sample_data.get('result', [])),#记录总共的样本数量
                'gen_length': gen_length,
                'steps': steps,
                'block_length': block_length,
            }
        }
    # 保存到JSON文件,标注出是什么位置的,并附加时间戳方便追踪
    # 在 output_dir 下创建 position_{position} 子文件夹
    position_dir = os.path.join(output_dir, f'position_{position}')
    os.makedirs(position_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'step_{steps}_gen_{gen_length}_{timestamp}.json'
    output_file = os.path.join(position_dir, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f'已保存结果到: {output_file}')
    return output_file


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
    paint_num=args.paint_num
    ispaint=args.ispaint
    return_current_conf=args.return_current_conf
    # mode=args.mode
    #换一个角度(我现在想关注的是位置的变化),这里添加nshot的参数,方便进行观察
    nshot=args.nshot
    result_path=args.result_path
    modes=['original']
    dataset=load_dataset(data_path,task)
    #取出前10个sample
    #还是计算平均值比较好
    samples_num=min(samples_num,len(dataset))
    dataset=dataset[:samples_num]

    print('------------------ load model -----------------------')
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model=AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    model.eval()
    print('--------------------start Answering----------------')
    #我只想画10张图,其他的我就进行累加即可
    #记录所有位置的答案
    for position in range(nshot+1):
        for mode in modes:
            print(f'------------------ start Answering with mode {mode} -----------------------')
            num=0
            # 初始化累积结构，用于存储所有样本的统计信息,这里额外统计的是token_change的更多细节
            # 这个是跨样本累加器,用于计算所有样本的累计信息
            # 根据任务类型初始化不同的键
            #适配了5个任务
            #如果之后还要考虑哪些范围可以在这里加
            if task == 'sudoku':
                range_names = ['answer_positions', 'all']
            elif task == 'countdown':
                range_names = ['answer_positions',  'all']
            elif task == 'gsm8k':
                range_names = ['answer_positions', 'all']
            elif task == 'math500':
                range_names = ['answer_positions', 'all']
            elif task == 'mbpp':
                #answer_positions表示的是答案的位置
                range_names = ['answer_positions', 'all']
            else:
                raise NotImplementedError(f"Task {task} not implemented.")
            
            accumulated_stats = {
                'conf': {name: {'overall_means': [], 'total_valid_counts': []} for name in range_names},
                # 'conf_diff': {name: {'overall_means': [], 'total_valid_counts': []} for name in range_names},
                # 'entropy': {name: {'overall_means': [], 'total_valid_counts': []} for name in range_names},
                'token_change': {name: {'overall_means': [], 'total_valid_counts': []} for name in range_names},
                'current_conf': {name: {'value_sums': []} for name in range_names},  # current_conf的累积统计（存储每个样本的 value_sum/total_valid_count）
                'change_bbox': {
                    'step_span': [],  # 跨样本累积 step_span
                    'token_span': [],  # 跨样本累积 token_span
                    'area': [],  # 跨样本累积 area
                    'token_change_total': []  # 跨样本累积 token_change 总数量
                }
            }
            
            sample_data={'result':[],'history_token_cover':[]   }
            #记录当前位置下的所有result,所以应该有nshot个result
            results=[]
            for index,input in enumerate(tqdm(dataset)):
                if 'Instruct' in model_name:
                    situation='instruct'
                else:
                    situation='base'
                num+=1
                answer,orders_result,token_change,conf,answer_token_positions,current_conf=generate(model,
                tokenizer,input,task,steps,gen_length,block_length,temperature,mode,situation,position,nshot,return_current_conf=return_current_conf)
                #这里进行健壮性补全
                results.append(answer)
                if answer_token_positions is None or answer is None:
                    continue

                #开始进行画图
                #当前这个问题的回答,已经计算好了,之后要存到参数里面
                #先计算当前平均值(先计算一个样本的情况,包括各种步数和累计计算的值)
                if task=='sudoku':
                    result=cal_accmulate_conf(conf,answer_token_positions,token_change=token_change)
                elif task=='countdown':
                    result=cal_accmulate_conf_countdown(conf,answer_token_positions,token_change=token_change,current_conf=current_conf)
                elif task=='gsm8k':
                    result=cal_accmulate_conf_gsm8k(conf,answer_token_positions,token_change=token_change,current_conf=current_conf)
                elif task=='math500':
                    result=cal_accmulate_conf_math(conf,answer_token_positions,token_change=token_change,current_conf=current_conf)
                #看看current_conf的效果怎么样
                elif task=='mbpp':
                    #mbpp的计算方式和math500是一样的
                    result=cal_accmulate_conf_mbpp(conf=conf,answer_token_positions=answer_token_positions,token_change=token_change,current_conf=current_conf)
                else:
                    raise NotImplementedError(f"Task {task} not implemented.")
                #这里就可以进行文件的保存了
                #先取出跨样本累加计算器的目录进行累加计算
                # 累积当前样本的统计信息,保存的应该是总的平均指标,其他大部分情况我都要case by case的观察
                #这里我要把覆盖值也算上(针对每一个样本进行计算即可)
                history_token_cover = {}
                # 基于 token_change 构建 step × gen_length 的二值矩阵，并统计覆盖变化的最小外接矩形
                # token_change: List[np.ndarray]，每个元素长度为 gen_length
                # 这里是计算每一步的token_change,和总体的token_change(缺点是只局限于某一个样本)
                if isinstance(token_change, list) and len(token_change) > 0:
                    # 规范化为 steps × gen_length 的 numpy 矩阵，变化>0 计为1，其余为0，-inf视为0
                    step_arrays = []
                    for step_arr in token_change:
                        if isinstance(step_arr, torch.Tensor):
                            arr = step_arr.detach().float().cpu().numpy()
                        else:
                            arr = np.asarray(step_arr)
                        if arr.ndim > 1:
                            arr = arr.flatten()
                        # 屏蔽 -inf，并将正值视为1
                        arr = np.where(np.isfinite(arr) & (arr > 0), 1, 0).astype(np.int32)
                        step_arrays.append(arr)
                    change_matrix = np.stack(step_arrays, axis=0)  # shape: steps × gen_length
                    token_change_total = cal_token_change_positive_total(token_change)
                    if np.any(change_matrix == 1):
                        step_idxs, token_idxs = np.where(change_matrix == 1)
                        step_span = int(step_idxs.max() - step_idxs.min() + 1)
                        token_span = int(token_idxs.max() - token_idxs.min() + 1)
                        history_token_cover['change_bbox'] = {
                            'step_span': step_span,     # 高（跨多少步）
                            'token_span': token_span,   # 宽（跨多少 token）
                            'area': int(step_span * token_span),
                            'token_change_total': token_change_total  # token_change 的总数量
                        }
                    else:
                        history_token_cover['change_bbox'] = {
                            'step_span': 0,
                            'token_span': 0,
                            'area': 0,
                            'token_change_total': token_change_total
                        }
                else:
                    token_change_total = cal_token_change_positive_total(token_change)
                    history_token_cover['change_bbox'] = {
                        'step_span': 0,
                        'token_span': 0,
                        'area': 0,
                        'token_change_total': token_change_total
                    }
                history_token_cover['token_change_total'] = token_change_total
                
                sample_data['result'].append(result)
                sample_data['history_token_cover'].append(history_token_cover)

                # 累积 change_bbox 的统计信息
                change_bbox = history_token_cover.get('change_bbox', {})
                accumulated_stats['change_bbox']['step_span'].append(change_bbox.get('step_span', 0))
                accumulated_stats['change_bbox']['token_span'].append(change_bbox.get('token_span', 0))
                accumulated_stats['change_bbox']['area'].append(change_bbox.get('area', 0))
                accumulated_stats['change_bbox']['token_change_total'].append(change_bbox.get('token_change_total', 0))
                #其他的范围目前确实不想加太多,计算非常麻烦
                for var_name in ['conf']:
                    # 使用之前定义的 range_names，这样会自动包含 'all' 范围
                    for range_name in range_names:
                        # 检查 result 中是否存在该范围（可能某些范围在某些情况下不存在）
                        if range_name in result[var_name]:
                            #传入的是总的平均值和总的有用值数量
                            overall_mean = result[var_name][range_name]['overall_mean']
                            total_valid_count = result[var_name][range_name]['total_valid_count']
                            # 只累积有效值（不是nan）
                            if not np.isnan(overall_mean):
                                #添加每一个样本的result的平均结果到跨样本累计值中
                                accumulated_stats[var_name][range_name]['overall_means'].append(overall_mean)
                                accumulated_stats[var_name][range_name]['total_valid_counts'].append(total_valid_count)
                
                # 累积 current_conf 的统计信息
                if 'current_conf' in result:
                    for range_name in range_names:
                        if range_name in result['current_conf']:
                            value_sum = result['current_conf'][range_name]['value_sum']
                            total_valid_count = result['current_conf'][range_name]['total_valid_count']
                            # 计算单个样本的平均值：value_sum / total_valid_count
                            if total_valid_count > 0:
                                mean_value = value_sum / total_valid_count
                            else:
                                mean_value = 0.0
                            # 累积每个样本的平均值
                            accumulated_stats['current_conf'][range_name]['value_sums'].append(mean_value)

                #确认要画图
                if num<=paint_num and ispaint==True:
                    #进行画图
                    #这里需要改,但是我目前确实不太需要计算
                    paint_conf(conf,conf_diff,entropy,answer_token_positions,token_change,orders_result,task,position,index)

            print('--------------------finish Answering----------------')
            #这里我想计算一下总的平均值
            all_samples_result = {}
            for var_name in ['conf']:
                all_samples_result[var_name] = {}
                # 根据任务类型使用相应的 range_names
                for range_name in range_names:
                    overall_means = accumulated_stats[var_name][range_name]['overall_means']
                    total_valid_counts = accumulated_stats[var_name][range_name]['total_valid_counts']
                    
                    if len(overall_means) > 0:
                        # 计算所有样本的平均值（简单平均）,对每一个平均了步数的样本再进行一次平均
                        mean_of_means_val = np.mean(overall_means)
                        # 转换为Python float，处理nan
                        if np.isnan(mean_of_means_val):
                            mean_of_means = np.nan
                        else:
                            mean_of_means = float(mean_of_means_val)
                        # 计算总的有效值数量
                        total_count = int(sum(total_valid_counts))
                    else:
                        mean_of_means = np.nan
                        total_count = 0
                    
                    all_samples_result[var_name][range_name] = {
                        'mean_of_means': mean_of_means,  # 所有样本平均值的简单平均
                        'total_valid_count': total_count,  # 所有样本的总有效值数量
                        'num_valid_samples': int(len(overall_means))  # 有效样本数量
                    }
            
            # 计算 change_bbox 的跨样本平均值,这里额外进行计算,是实际上比对的就是这些量
            all_samples_result['change_bbox'] = {}
            for metric_name in ['step_span', 'token_span', 'area', 'token_change_total']:
                values = accumulated_stats['change_bbox'][metric_name]
                if len(values) > 0:
                    mean_value = float(np.mean(values))
                else:
                    mean_value = 0.0
                all_samples_result['change_bbox'][metric_name] = {
                    'mean': mean_value,  # 所有样本的平均值
                    'num_samples': int(len(values))  # 样本数量
                }
            
            # 计算 current_conf 的跨样本统计,进行统计
            all_samples_result['current_conf'] = {}
            for range_name in range_names:
                mean_values = accumulated_stats['current_conf'][range_name]['value_sums']  # 存储的是每个样本的 value_sum/total_valid_count
                
                if len(mean_values) > 0:
                    # 计算所有样本比例的平均值（跨样本平均值）
                    mean_of_means = float(np.mean(mean_values))
                    # 有效样本数量
                    num_valid_samples = int(len(mean_values))
                else:
                    mean_of_means = 0.0
                    num_valid_samples = 0
                
                all_samples_result['current_conf'][range_name] = {
                    'mean_of_means': mean_of_means,  # 所有样本比例的平均值（value_sum/total_valid_count 的平均值）
                    'num_valid_samples': num_valid_samples  # 有效样本数量
                }
            # 保存整体参数
            #保存的是每一个位置的所有测试结果(是按照所有数据一步一步来,和位置分隔开)
            #其实是顺便也评测了一遍,但是因为大头在模型输出,其实也可以看作是相对独立的进程
            accuracy=eval(task,results,dataset,result_path,args,position=position)
            #这里方便我画图
            save_results_to_json(all_samples_result, sample_data, task, position,nshot,gen_length,steps,block_length,accuracy)




#现在主要关注的就是all_samples_result的conf(digit_symbol或者是digit_symbol)的平均值,还有change_box的step_span\token_span\area的平均值(关注的是覆盖了多少)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='sudoku')
    parser.add_argument('--model_name',type=str,default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--device',type=str,default='cuda:1')
    parser.add_argument('--gen_length',type=int,default=128)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--block_length',type=int,default=128)
    parser.add_argument('--temperature',type=float,default=0.0)
    parser.add_argument('--data_path',type=str,default='./data/sudoku.csv')
    parser.add_argument('--samples_num',type=int,default=288)
    parser.add_argument('--nshot',type=int,default=5)
    parser.add_argument('--paint_num',type=int,default=5)
    parser.add_argument('--ispaint', action='store_true', default=False)
    parser.add_argument('--result_path',type=str,default='./results/conf_results')
    parser.add_argument('--return_current_conf',action='store_true',default=False)
    args=parser.parse_args()
    main(args)