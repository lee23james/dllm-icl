import os, json, re, csv, sys
from pathlib import Path
from typing import Dict, List, Optional
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.template import *
from utils.load_json_or_jsonl import load_json_or_jsonl
from src.eval_template import sudoku_prompt_eval,countdown_prompt_eval
import random
import torch
import torch.nn.functional as F
from collections import defaultdict
from string import printable
import builtins
from datetime import datetime

#工具类一般会是各种函数组合在一起,想用的时候都可以用到
#根据query_position的位置,来构建prompt
#目前考虑的是countdown和sudoku都需要考虑nshot和position的情况,利用nshot来构造相关的模版,进而进行计算分析
def query_extract(input, task, query_position, gen_length,nshot=None,iscot=False):
    if task == 'mbpp':
        return mbpp_prompt(input['prompt'], query_position,nshot, gen_length,input['test_list'][0],input['test_list'][1],input['test_list'][2])
    elif task == 'math500':
        return math_500_prompt(input['problem'],query_position, nshot, gen_length)
    elif task =='countdown':
        return countdown_prompt(input['input'],query_position, nshot,gen_length)
    elif task =='sudoku':
        # sudoku 任务不支持 iscot，始终传递 False
        return sudoku_prompt(input['Puzzle'],query_position, gen_length,nshot)
    #修改逻辑,我先在需要根据nshot统一放在一个list,然后依次取出进行计算
    elif task =='gsm8k':
        return gsm8k_prompt(input['question'],query_position, nshot, gen_length)
    #这个数据集暂时用不到,主要是太麻烦了,分类我自有方法
    elif task =='gpqa'and nshot is not None:
        return gpqa_prompt(input['question'],  [input['option_A'], input['option_B'], input['option_C']],input['correct_answer'],nshot, gen_length)
    else:
        raise NotImplementedError(f"Mode {task} not implemented or nshot is None")


#这里的评估目前还要实打实的考虑评估,看看是不是也和conf也有关系
def eval_prompt(input,task,query_position):
    if task=='sudoku':
        return sudoku_prompt_eval(input['Puzzle'],input['Solution'],query_position)
    elif task=='countdown':
        return countdown_prompt_eval(input['input'],query_position,input['output'])
    else:
        raise NotImplementedError(f"Mode {task} not implemented")


#加载数据集,转化为list[Dict[str,str]]格式(相当于吧json和jsonl序列化)
def load_dataset(data_path, task,max_samples=None):
    if task == 'sudoku':
        dataset = load_sudoku_dataset(data_path)
        if not dataset:
            raise ValueError(f"Error: Dataset file '{data_path}' not found.")
        # return dataset
    #根据任务去转化返回对应数据集
    else:
        data_json = load_json_or_jsonl(data_path)
        dataset = []
        for key in data_json.keys():
            dataset.append(data_json[key])

    #指定添加的数量
    if max_samples is not None and max_samples >0:
        dataset=dataset[:max_samples]
    return dataset

#我先准备gsm8k和数独还有countdown,代码这个任务我稍晚再加进去
#先准备sudoku数据集
#因为sudoku数据集是csv,这里考虑把csv数据集转化成List[Dict[str,str]]格式
def load_sudoku_dataset(file_path:str)->List[Dict[str,str]]:
    dataset=[]
    try:
        with open(file_path,mode='r',encoding='utf-8')as csvfile:
            reader =csv.DictReader(csvfile)
            for row in reader:
                dataset.append(row)
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
    return dataset

#比较sudoku的答案
def check_solution(prediction:str,ground_truth:str)->bool:
    match=re.search(r'<answer>(.*?)</answer>',prediction,re.DOTALL)
    if not match:
        return ground_truth in prediction.replace(" ","").replace("\n","")
    solution_part=match.group(1).strip().replace("\n", "").replace(" ", "")
    return solution_part == ground_truth

#评估sudoku的正确率
def eval_sudoku(results,dataset,result_path,args,position,iswrite=True):
    true_num = 0
    #results相当于是一个list,每一项都是模型回答预测的答案
    result_statuses = []
    for index,answer in enumerate(results):
        puzzle_data = dataset[index]
        is_correct = check_solution(answer,puzzle_data['Solution'])
        result_statuses.append((index, answer, is_correct))
        #进行比较
        if is_correct:
            true_num+=1

    print('-------------------Finish Answering-----------------------')

    accuracy = true_num/len(dataset)
    print(f"Final Accuracy:{accuracy:.4f}({true_num}/{len(dataset)})")

    # 只有当 iswrite=True 时才写入文件
    if iswrite:
        # 保持原有功能：写入原来的 result_path
        # 如果 result_path 是目录，在该目录下创建 result.txt
        result_path_obj = Path(result_path)
        if result_path_obj.is_dir() or (not result_path_obj.exists() and not result_path_obj.suffix):
            # 如果是目录或没有扩展名，在目录下创建 result.txt
            result_file_path = result_path_obj / "result.txt"
            result_path_obj.mkdir(parents=True, exist_ok=True)
        else:
            # 如果是文件路径，直接使用
            result_file_path = result_path_obj
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file_path, 'a', encoding='utf-8') as file:
            file.write("----------------Args Configguration-----------------\n")
            for arg in vars(args):
                file.write(f"{arg}: {getattr(args, arg)}\n")
            file.write("\n")
            file.write(f"Total Accuracy: {accuracy}\n")
            file.write("\n\n")
        
        # 新增功能：创建新的目录结构并保存详细结果
        # base_dir 是 result_path 的父目录（如果是文件）或 result_path 本身（如果是目录）
        base_dir = result_file_path.parent if result_file_path.is_file() else result_file_path
        
        # 从 args 中读取实验配置
        nshot = getattr(args, "nshot", None)
        steps = getattr(args, "steps", None)
        gen_length = getattr(args, "gen_length", None)
        
        # 构建多级目录：shot_step_gen / position
        shot_part = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
        position_part = f"position_{position}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = base_dir / shot_part / position_part
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳命名的 txt 文件
        txt_output_path = output_dir / f"{timestamp}.txt"
        
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("=== Execution Results Summary ===\n")
            f.write(f"Position: {position}\n")
            f.write(f"nshot: {nshot}\n")
            f.write(f"steps: {steps}\n")
            f.write(f"gen_length: {gen_length}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("=== Args Configuration ===\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n")
            
            f.write(f"Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})\n\n")
            
            f.write("=== Detailed Results ===\n")
            for index, answer, is_correct in result_statuses:
                f.write(f"=======idx:{index}=========\n")
                f.write(f"Result: {answer}\n")
                f.write(f"Status: {'true' if is_correct else 'false'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Detailed results saved to: {txt_output_path}")
    
    #这里返回准确率,方便后续画点
    return accuracy

#接受一群输入来计算各个位置的准确率,这个不进行记录,直接计算即可
#这里的result结构式list[list[str]],第一个list是问题的列表,第二个list是答案的列表,总共position+1个候选答案,然后根据这些候选答案来看看有没有一个是正确答案,如果有一个是正确答案,那么truenum+1
def  eval_sudoku_position(results,dataset):
    """
    评估sudoku数据集在各个位置的准确率
    Args:
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集,每个元素包含'Solution'字段作为正确答案
    Returns:
        accuracy: float, 准确率 (至少有一个候选答案正确的样本数 / 总样本数)
    """
    true_num = 0
    
    # 遍历每个问题及其候选答案
    for index, candidate_answers in enumerate(results):
        puzzle_data = dataset[index]
        ground_truth = puzzle_data['Solution']
        
        # 检查这position+1个候选答案中是否有任何一个正确
        is_correct = False
        for answer in candidate_answers:
            if check_solution(answer, ground_truth):
                is_correct = True
                break
        
        # 如果有一个是正确答案,那么truenum+1
        if is_correct:
            true_num += 1
    
    # 计算准确率
    accuracy = true_num / len(dataset) if len(dataset) > 0 else 0.0
    
    print('----------------- Finish Evaluating Position Accuracy -------------------')
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")
    
    # 返回准确率,方便后续使用
    return accuracy



#轮到countdown了
def countdown_check(model_answer,ground_truth):
    if ground_truth in model_answer:
        return True
    else:
        return False

#评估countdown数据集并计算正确率
def eval_countdown(results, dataset, result_path, args, position, iswrite=True):
    true_num = 0
    # 存储每个结果的检查状态
    result_statuses = []
    for index, answer in enumerate(results):
        result = dataset[index]
        is_correct = countdown_check(answer, result['output'])
        result_statuses.append((index, answer, is_correct))
        if is_correct:
            true_num += 1

    print('----------------- Finish Answering -------------------')
    accuracy = true_num/len(dataset)
    print(f"Final Accuracy:{accuracy:.4f}({true_num}/{len(dataset)})")

    # 只有当 iswrite=True 时才写入文件
    if iswrite:
        # 保持原有功能：写入原来的 result_path
        # 如果 result_path 是目录，在该目录下创建 result.txt
        result_path_obj = Path(result_path)
        if result_path_obj.is_dir() or (not result_path_obj.exists() and not result_path_obj.suffix):
            # 如果是目录或没有扩展名，在目录下创建 result.txt
            result_file_path = result_path_obj / "result.txt"
            result_path_obj.mkdir(parents=True, exist_ok=True)
        else:
            # 如果是文件路径，直接使用
            result_file_path = result_path_obj
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file_path, 'a', encoding='utf-8') as file:
            file.write("----------------- Args Configuration -------------------\n")
            for arg in vars(args):
                file.write(f"{arg}: {getattr(args, arg)}\n")
            file.write("\n\n")
            file.write(f"Total Accuracy: {true_num / len(dataset)}\n")
            file.write("\n\n")
        
        # 新增功能：创建新的目录结构并保存详细结果
        # base_dir 是 result_path 的父目录（如果是文件）或 result_path 本身（如果是目录）
        base_dir = result_file_path.parent if result_file_path.is_file() else result_file_path
        
        # 从 args 中读取实验配置
        nshot = getattr(args, "nshot", None)
        steps = getattr(args, "steps", None)
        gen_length = getattr(args, "gen_length", None)
        
        # 构建多级目录：shot_step_gen / position
        shot_part = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
        position_part = f"position_{position}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = base_dir / shot_part / position_part
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳命名的 txt 文件
        txt_output_path = output_dir / f"{timestamp}.txt"
        
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("=== Execution Results Summary ===\n")
            f.write(f"Position: {position}\n")
            f.write(f"nshot: {nshot}\n")
            f.write(f"steps: {steps}\n")
            f.write(f"gen_length: {gen_length}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("=== Args Configuration ===\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n")
            
            f.write(f"Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})\n\n")
            
            f.write("=== Detailed Results ===\n")
            for index, answer, is_correct in result_statuses:
                f.write(f"=======idx:{index}=========\n")
                f.write(f"Result: {answer}\n")
                f.write(f"Status: {'true' if is_correct else 'false'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Detailed results saved to: {txt_output_path}")
    
    #这里返回准确率,方便后续画点
    return accuracy

#接受一群输入来计算各个位置的准确率,这个不进行记录,直接计算即可
#这里的result结构式list[list[str]],第一个list是问题的列表,第二个list是答案的列表,总共position+1个候选答案,然后根据这些候选答案来看看有没有一个是正确答案,如果有一个是正确答案,那么truenum+1
def  eval_countdown_position(results,dataset):
    """
    评估countdown数据集在各个位置的准确率
    Args:
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集,每个元素包含'output'字段作为正确答案
    Returns:
        accuracy: float, 准确率 (至少有一个候选答案正确的样本数 / 总样本数)
    """
    true_num = 0
    
    # 遍历每个问题及其候选答案
    for index, candidate_answers in enumerate(results):
        puzzle_data = dataset[index]
        ground_truth = puzzle_data['output']
        
        # 检查这position+1个候选答案中是否有任何一个正确
        is_correct = False
        for answer in candidate_answers:
            if countdown_check(answer, ground_truth):
                is_correct = True
                break
        
        # 如果有一个是正确答案,那么truenum+1
        if is_correct:
            true_num += 1
    
    # 计算准确率
    accuracy = true_num / len(dataset) if len(dataset) > 0 else 0.0
    
    print('----------------- Finish Evaluating Position Accuracy -------------------')
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")
    
    # 返回准确率,方便后续使用
    return accuracy



# 评估math500并收集准确率
def collect_answer_from_response(response):
    _res = ""

    # ------------------ 第一阶段：匹配 <answer> 标签 ------------------
    try:
        regex_list_tag = [
            r"<answer>.*?boxed{(.*?)}.*?</answer>",
            r"<answer>.*?framebox{(.*?)}.*?</answer>"
        ]

        for regex in regex_list_tag:
            matches = re.findall(regex, response, flags=re.DOTALL)
            _res = matches[0] if matches else ""
            if _res != "":
                break

    except Exception:
        pass

    if _res != "":
        return _res.strip('.').strip()

    # ------------------ 第二阶段：匹配普通 boxed/framebox ------------------
    try:
        regex_list_box = [
            r"boxed{(.*?)}",
            r"framebox{(.*?)}"
        ]

        for regex in regex_list_box:
            matches = re.findall(regex, response, flags=re.MULTILINE)
            _res = matches[0] if matches else ""
            if _res != "":
                break

    except Exception:
        pass

    return _res.strip('.').strip()

#一般都是用in进行匹配
def eval_math500(results,dataset,result_path,args,position,iswrite=True):
    true_num=0
    result_statuses = []
    for index,answer in enumerate(results):
        is_correct=dataset[index]['answer'] in collect_answer_from_response(answer)
        result_statuses.append((index, answer, is_correct))
        if is_correct:
            true_num+=1

    print('-------------------Finish Answering-----------------------')

    accuracy = true_num/len(dataset)
    print(f"Final Accuracy:{accuracy:.4f}({true_num}/{len(dataset)})")

    # 只有当 iswrite=True 时才写入文件
    if iswrite:
        # 保持原有功能：写入原来的 result_path
        # 如果 result_path 是目录，在该目录下创建 result.txt
        result_path_obj = Path(result_path)
        if result_path_obj.is_dir() or (not result_path_obj.exists() and not result_path_obj.suffix):
            # 如果是目录或没有扩展名，在目录下创建 result.txt
            result_file_path = result_path_obj / "result.txt"
            result_path_obj.mkdir(parents=True, exist_ok=True)
        else:
            # 如果是文件路径，直接使用
            result_file_path = result_path_obj
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file_path, 'a', encoding='utf-8') as file:
            file.write("-------------------Args Configuration---------------------\n")
            for arg in vars(args):
                file.write(f"{arg}:{getattr(args,arg)}\n")
            file.write("\n\n")
            file.write(f"Total Accuracy:{accuracy}\n")
            file.write("\n\n")
        
        # 新增功能：创建新的目录结构并保存详细结果
        # base_dir 是 result_path 的父目录（如果是文件）或 result_path 本身（如果是目录）
        base_dir = result_file_path.parent if result_file_path.is_file() else result_file_path
        
        # 从 args 中读取实验配置
        nshot = getattr(args, "nshot", None)
        steps = getattr(args, "steps", None)
        gen_length = getattr(args, "gen_length", None)
        
        # 构建多级目录：shot_step_gen / position
        shot_part = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
        position_part = f"position_{position}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = base_dir / shot_part / position_part
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳命名的 txt 文件
        txt_output_path = output_dir / f"{timestamp}.txt"
        
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("=== Execution Results Summary ===\n")
            f.write(f"Position: {position}\n")
            f.write(f"nshot: {nshot}\n")
            f.write(f"steps: {steps}\n")
            f.write(f"gen_length: {gen_length}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("=== Args Configuration ===\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n")
            
            f.write(f"Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})\n\n")
            
            f.write("=== Detailed Results ===\n")
            for index, answer, is_correct in result_statuses:
                f.write(f"=======idx:{index}=========\n")
                f.write(f"Result: {answer}\n")
                f.write(f"Status: {'true' if is_correct else 'false'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Detailed results saved to: {txt_output_path}")
    
    #这里返回准确率,方便后续画点
    return accuracy

def eval_math500_position(results,dataset):
    """
    评估math500数据集在各个位置的准确率
    Args:
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集,每个元素包含'answer'字段作为正确答案
    Returns:
        accuracy: float, 准确率 (至少有一个候选答案正确的样本数 / 总样本数)
    """
    true_num = 0
    
    # 遍历每个问题及其候选答案
    for index, candidate_answers in enumerate(results):
        puzzle_data = dataset[index]
        ground_truth = puzzle_data['answer']
        # 检查这position+1个候选答案中是否有任何一个正确
        is_correct = False
        for answer in candidate_answers:    
            if ground_truth in collect_answer_from_response(answer):
                is_correct = True
                break
        
        # 如果有一个是正确答案,那么truenum+1
        if is_correct:
            true_num += 1
    
    # 计算准确率
    accuracy = true_num / len(dataset) if len(dataset) > 0 else 0.0
    
    print('----------------- Finish Evaluating Position Accuracy -------------------')
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")
    
    # 返回准确率,方便后续使用
    return accuracy

#开始评估mbpp
#这个是针对mbpp-full的,现在我需要扩展到sanitized子集
#我现在要稍微改进一下,我需要取前3个进行测试,来完成代码评估
#有可能会在bestposition下求出最好的位置
def generate_mbpp_test_files(
    samples:List[Dict],
    model_outputs:List[str],
    output_dir:Path,#生成的py放置的位置
    template_path:Optional[Path] =None,
    prefix: str="test_index_"#自动添加后缀
)->List[Path]:

    if len(samples)!=len(model_outputs):
        raise ValueError("The number of samples and model outputs must be the same")

    default_template="""\"\"\"
Test file for task_id: {task_id}
Problem description: {text}
\"\"\"

{setup_code}

{model_code}

{test_code}
"""
    template=default_template
    if template_path and template_path.exists():
        template=template_path.read_text()
    output_paths=[]
    #dataset就是相当于sample
    for i,(sample,model_code) in enumerate(zip(samples,model_outputs)):
        if isinstance(sample,str):
            sample=json.loads(sample)
        required_fields=["prompt","task_id","test_list"]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample is missing required field:{field}")
        task_id=sample["task_id"]
        
        # 改进的代码提取逻辑
        import re
        # 提取[BEGIN]和[DONE]之间的代码块
        done_match = re.search(r'\[BEGIN\]([\s\S]*?)\[DONE\]',model_code,re.MULTILINE)
        if done_match:
            model_code=done_match.group(1).strip()
        try:
            extracted_func=model_code.split('```python')[1].split('```')[0]
            extracted_func=extracted_func.strip()
        except:
            # 如果没有markdown标记，直接使用原始代码
            extracted_func = model_code.strip()
        
        test_code="\n\n".join(sample["test_list"][:3])
        if "challenge_test_list" in sample:
            test_code+="\n\n"+"\n\n".join(sample['challenge_test_list'])
        if "test_setup_code" in sample:
            setup_code_content=sample["test_setup_code"]
        else:
            import_list=sample.get("test_imports",[])
            if import_list:
                setup_code_content="\n".join(import_list)
            else:
                setup_code_content=""


        test_file_content = template.format(
            task_id=task_id,
            text=sample["prompt"],
            setup_code=setup_code_content,
            #对模型生成代码进行清理
            model_code=extracted_func,
            test_code=test_code
        )
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path=output_dir/f"{prefix}{task_id}.py"
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(test_file_content)
        output_paths.append(output_path)
    return output_paths

#这个主要是生成后续代码评估模型需要的问题、还有模型回答的文件,position是一个参数,可以记录str还是int
#创建所需文件夹
def eval_mbpp(results, dataset, result_dir, args,position):
    """
    为 MBPP 任务生成用于代码执行评测的测试文件。
    目录结构：
        result_dir/
            mbpp_results/ (通常由 result_dir 指定)
                shot_{nshot}_step_{steps}_gen_{gen_length}/
                    position_{query_position}/
                        test_index_{task_id}_timestamp.py
    """
    base_dir = Path(result_dir)
    
    # 从 args 中读取实验配置，若不存在则使用安全默认值
    nshot = getattr(args, "nshot", None)
    steps = getattr(args, "steps", None)
    gen_length = getattr(args, "gen_length", None)
    if position is not None:
        #对于query_position的判断,如果是字符就选择auto
        # if isinstance(position,int):
        query_position = position
        # else:
        #     query_position = "best_position"#有一个最好的位置
    else:
        query_position = "auto-icl"
    # 构建多级目录：shot_step_gen / position / timestamp
    shot_part = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
    position_part = f"position_{query_position}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / shot_part / position_part / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成具体的测试文件到该目录下
    generate_mbpp_test_files(dataset, results, output_dir)

#results:list[list[str]],第一个list是问题的列表,第二个list是答案的列表,总共position+1个候选答案
#resultdir生成的是文件放置的位置
def eval_mbpp_position(results, dataset, result_dir=None, args=None, nshot=None):
    """
    评估MBPP数据集在各个位置的准确率（upperbound）
    
    Args:
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集
        result_dir: 结果目录（必须提供或通过args.result_path获取）
        args: 参数对象（必须提供，包含nshot, steps, gen_length等）
        nshot: shot数量（必须提供或通过args.nshot获取）
    Returns:
        accuracy: float, upperbound准确率（通过evaluate_python_files_positions计算）
    """
    from utils.judge_python_code import evaluate_python_files_positions
    
    # 参数检查
    assert args is not None, "args must be provided"
    assert len(results) > 0, "results cannot be empty"
    assert len(results) == len(dataset), f"results length ({len(results)}) must match dataset length ({len(dataset)})"
    
    # 获取必要参数
    if nshot is None:
        nshot = getattr(args, 'nshot', None)
    assert nshot is not None and nshot >= 0, "nshot must be provided via args.nshot or nshot parameter"
    
    if result_dir is None:
        result_dir = getattr(args, 'result_path', None)
    assert result_dir is not None, "result_dir must be provided via result_dir parameter or args.result_path"
    
    steps = getattr(args, 'steps', None)
    assert steps is not None, "args.steps must be provided"
    
    gen_length = getattr(args, 'gen_length', None)
    assert gen_length is not None, "args.gen_length must be provided"
    
    # 检查 results 结构：results[question_index] = [answer_pos0, answer_pos1, ..., answer_pos_nshot]
    assert isinstance(results[0], list), "results must be list[list[str]]"
    assert len(results[0]) == nshot + 1, f"Each question should have {nshot + 1} answers (one for each position), but got {len(results[0])}"
    
    # 将按问题组织的结果转换为按位置组织：results_by_position[position] = [answer1, answer2, ...]
    results_by_position = [[] for _ in range(nshot + 1)]
    for question_index, candidate_answers in enumerate(results):
        assert len(candidate_answers) == nshot + 1, f"Question {question_index} should have {nshot + 1} answers"
        for position in range(nshot + 1):
            results_by_position[position].append(candidate_answers[position])
    
    # 为每个位置生成测试文件
    for position in range(nshot + 1):
        eval_mbpp(results_by_position[position], dataset, result_dir, args, position)
    
    # 调用 evaluate_python_files_positions 计算 upperbound accuracy
    accuracy = evaluate_python_files_positions(result_dir, nshot, steps, gen_length)
    
    print('----------------- Finish Evaluating MBPP Position Accuracy -------------------')
    print(f"Upperbound Accuracy: {accuracy:.4f}")
    
    return accuracy


def normalize_number(s: str) -> str:
    """规范化数字字符串"""
    s = s.strip()
    s = re.sub(r'[\$,]', '', s)  # 去掉 $ 和 ,
    s = s.rstrip('.')           # 去掉末尾的点
    return s

#从标准答案,和模型的预测回答中按照1.严格匹配 2.灵活匹配的方式去查看是否正确
#这里面ground_truth是直接用列表里面的answer字段,然后提取出数字来
#这里考虑到了度哟中情况,既可以处理###又可以处理###的情况
def gsm8k_check(model_answer:str,ground_truth:str)->bool:

    #现在要从ground_truth中提取数字
    #groundtruth一定可以找到
    strict_truth=re.search("#### (\\-?[0-9\\.\\,]+)",ground_truth)
    truth=strict_truth.group(1)
    
    strict_match=re.search(r"The answer is (\-?[0-9\.\,]+)",model_answer)
    #先找出模型推理得出的正确答案
    if strict_match:
        answer= strict_match.group(1)
    else:
        #然后进行灵活匹配
        flexible_match=re.search("#### (\\-?[0-9\\.\\,]+)",model_answer)
        if flexible_match:
            answer=flexible_match.group(1)
        else:
            return truth in model_answer

    if normalize_number(answer)==normalize_number(truth):
        return True
    else:
        return False
    

#自己配的评估gsm8k的代码,主要还是测试的是准确率,因为改变的是位置,无法运用到任何一个评估框架得自己写
#首先得先理解results列表里面主要是什么
def eval_gsm8k(results,dataset,result_path,args,position,iswrite=True):
    true_num=0
    # 存储每个结果的检查状态
    result_statuses = []
    for index,answer in enumerate(results):
        # result = dataset[index]
        ground_truth=dataset[index]['answer']
        is_correct = gsm8k_check(answer,ground_truth)
        result_statuses.append((index, answer, is_correct))
        if is_correct:
            true_num+=1
    
    print('-------------------Finish Answering-----------------------')

    accuracy=true_num/len(dataset)
    print(f"Final Accuracy:{accuracy:.4f}({true_num}/{len(dataset)})")

    # 只有当 iswrite=True 时才写入文件
    if iswrite:
    # 保持原有功能：写入原来的 result_path
    # 如果 result_path 是目录，在该目录下创建 result.txt
        result_path_obj = Path(result_path)
        if result_path_obj.is_dir() or (not result_path_obj.exists() and not result_path_obj.suffix):
            # 如果是目录或没有扩展名，在目录下创建 result.txt
            result_file_path = result_path_obj / "result.txt"
            result_path_obj.mkdir(parents=True, exist_ok=True)
        else:
            # 如果是文件路径，直接使用
            result_file_path = result_path_obj
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file_path, 'a', encoding='utf-8') as file:
            file.write("-------------------Args Configuration---------------------\n")
            for arg in vars(args):
                file.write(f"{arg}:{getattr(args,arg)}\n")
            file.write("\n")
            file.write(f"Total Accuracy:{accuracy}\n")
            file.write("\n\n")
        
        # 新增功能：创建新的目录结构并保存详细结果
        # base_dir 是 result_path 的父目录（如果是文件）或 result_path 本身（如果是目录）
        base_dir = result_file_path.parent if result_file_path.is_file() else result_file_path
        
        # 从 args 中读取实验配置
        nshot = getattr(args, "nshot", None)
        steps = getattr(args, "steps", None)
        gen_length = getattr(args, "gen_length", None)
        
        # 构建多级目录：shot_step_gen / position
        shot_part = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
        position_part = f"position_{position}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = base_dir / shot_part / position_part
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳命名的 txt 文件
        txt_output_path = output_dir / f"{timestamp}.txt"
        
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("=== Execution Results Summary ===\n")
            f.write(f"Position: {position}\n")
            f.write(f"nshot: {nshot}\n")
            f.write(f"steps: {steps}\n")
            f.write(f"gen_length: {gen_length}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("=== Args Configuration ===\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n")
            
            f.write(f"Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})\n\n")
            
            f.write("=== Detailed Results ===\n")
            for index, answer, is_correct in result_statuses:
                f.write(f"=======idx:{index}=========\n")
                f.write(f"Result: {answer}\n")
                f.write(f"Status: {'true' if is_correct else 'false'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Detailed results saved to: {txt_output_path}")
    
    #这里返回准确率,方便后续画点
    return accuracy


#接受一群输入来计算各个位置的准确率,这个不进行记录,直接计算即可
#这里的result结构式list[list[str]],第一个list是问题的列表,第二个list是答案的列表,总共position+1个候选答案,然后根据这些候选答案来看看有没有一个是正确答案,如果有一个是正确答案,那么truenum+1
def eval_gsm8k_position(results,dataset):
    """
    评估gsm8k数据集在各个位置的准确率
    Args:
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集,每个元素包含'answer'字段作为正确答案
    Returns:
        accuracy: float, 准确率 (至少有一个候选答案正确的样本数 / 总样本数)
    """
    true_num = 0
    
    # 遍历每个问题及其候选答案
    for index, candidate_answers in enumerate(results):
        puzzle_data = dataset[index]
        ground_truth = puzzle_data['answer']
        
        # 检查这position+1个候选答案中是否有任何一个正确
        is_correct = False
        for answer in candidate_answers:
            if gsm8k_check(answer, ground_truth):
                is_correct = True
                break
        
        # 如果有一个是正确答案,那么truenum+1
        if is_correct:
            true_num += 1
    
    # 计算准确率
    accuracy = true_num / len(dataset) if len(dataset) > 0 else 0.0
    
    print('----------------- Finish Evaluating Position Accuracy -------------------')
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")
    
    # 返回准确率,方便后续使用
    return accuracy


def gpqa_check(model_answer:str,ground_truth:str)->bool:
    pattern=r"The answer is\s*\(([A-D])\)"
    match=re.search(pattern,model_answer)
    if match:
        answer=match.group(1)
    else:
        return ground_truth in model_answer
    return answer==ground_truth



#现在开始评估多选题gpqa的代码
#results是list[list[str]],第一个list是一个数据集,每一个list是所有位置的回答,每一个str是答案
def eval_gpqa(results,dataset,result_path,args,correct_letters):
    true_num={}
    for index,answers in enumerate(results):
        ground_truth=correct_letters[index]
        for position,answer in enumerate(answers):
        #添加相应字段
            if str(position) not in true_num:
                true_num[str(position)]=0
            if gpqa_check(answer,ground_truth):
                true_num[str(position)]+=1
            # if gpqa_check(answer,ground_truth):
            #     true_num+=1

    print('-------------------Finish Answering-----------------------')

    # accuracy=true_num/len(dataset)


    with open(result_path,'a',encoding='utf-8') as file:
        file.write("-------------------Args Configuration---------------------\n")
        for arg in vars(args):
            file.write(f"{arg}:{getattr(args,arg)}\n")
        file.write("\n")

        # file.write(f"Total Accuracy:{accuracy}\n")

        for (position,num) in true_num.items():
            file.write(f"Position {position} Accuracy: {num / len(dataset)} \n")
        file.write("\n\n")


#--------------------------------下面是ag_news、SST2、SUBJ、TREC的评估工具类------------------------------------、
#这个函数主要是将数据集按照标签进行分组
def group_by_label(corpus:List[Dict],label_str="label")->Dict[str,List[Dict]]:
    grouped_corpus= defaultdict(list)
    for elem in corpus:
        #可以明确的是产生了字段,对于不同的字段采样相同的label_str
        label = elem[label_str]
        grouped_corpus[label].append(elem)
        #对于相应的标签,选取了特定的分组
    return grouped_corpus


#   情感分析采集标签策略:选择平衡模式的时候是每一个标签都采样kshot个样本,选择随机模式的时候是随机选择kshot个样本
def corpus_sampling(train_corpus:List[Dict],kshot,mode='balance',label_str="label")->List[Dict]:
    #根据标签获取了一群数据集样本
    grouped_corpus = group_by_label(train_corpus,label_str=label_str)
    selected=[]
    if mode =="balance":
        for label in grouped_corpus:
            #对于每一个标签,随机采样kshot个样本(貌似不太需要固定seed,因为我发现他一次产生所有位置)
            selected+=random.sample(grouped_corpus[label],kshot)
    elif mode=="random":
        print("random sample train corpus,k shot = k examples")
        selected +=random.sample(train_corpus,kshot)
        
    else:
        raise NotImplementedError("Please choose mode between balance and random")
    #返回的是一个list,list的每个元素是一个字典,字典的key是sentence_1_str,sentence_2_str,label_str,index
    return selected


#动态构造prompt,输入的是模版,输出的是替换模版的指令,相当于模版替换过程,这里构造掩码长度,方便后续评估
def create_prompt(mask_length,template,sentence:tuple,label_text,test=False,sentence_pair=False):
    """
    :param template: "f'Review: {sentence_1}\nSentiment: {label_text}\n\n'"
    :param sentence: tuple, e.g., (sent1, ) or (sent1, sent2)
    :param label_text: string, e.g., good or bad
    :param test: Boolean
    :param sentence_pair: Boolean
    :return:
    """
    #sentence_pair只有true和false之分
    if sentence_pair:
        assert len(sentence)==2,"you should input sentence pair"
        assert "sentence_1" in template,"sentence_1 not found in template"
        assert "sentence_2" in template,"sentence_2 not found in template"
        sentence_1,sentence_2 =sentence
        #清理格式,规范prompt输入
        sentence_1 =' '.join(sentence_1.split())
        sentence_2=' '.join(sentence_2.split())

    else:
        assert len(sentence) ==1 ,"you should input single sentence as string"
        assert type(sentence) == tuple
        assert "sentence_1" in template,"sentence_1 not in template"
        assert "sentence_2" not in template,"sentence_2 found in template"

        sentence_1 = sentence[0]
        sentence_1=' '.join(sentence_1.split())
    #如果是测试,就加入一些掩码标记,标记着这个是需要回答的,因为需要把它嵌入到各个位置,所以还是稍微有一些不一样的
    if test:
        #去掉答案,加上掩码(这里稍微修改一下,因为可能会自动生成空格)
        # template = template[:template.index("{label_text}")]+"<|mdm_mask|>"*mask_length+"'"
        prefix = template[:template.index("{label_text}")].rstrip()
        template = prefix + "<|mdm_mask|>" * mask_length + "'"
        assert f"{label_text}" not in template,"should remove label text for test data"
        template_text = builtins.eval(template)
    
    else:
        template_text=builtins.eval(template)
    return template_text
#---------------------------------情感分析------------------------------------
#情感分析评估(我想先用准确率进行测试)
#先进行sst2的评估
#现在开始评估多选题gpqa的代码
#其中results是list[List[Dict]],每一个List是所有位置的回答,每一个Dict是每一个位置的回答
#dataset是list[str],每一个str就是答案
def eval_emotion(results,dataset,result_path,args):
    true_num={}
    # gts={}#我想设置为一个字段,就是同一字段的进行记录对比来看看各个位置的效果
    for index,answers in enumerate(results):
        ground_truth=dataset[index]
        #answer有两个字段,一个字段是position,一个是prompt,这里的position相当于index
        for answer in answers:
            #添加相应的字段
            if str(answer['position']) not in true_num:
                true_num[str(answer['position'])]=0
            if ground_truth in answer['prompt']:
                true_num[str(answer['position'])]+=1
    
    print('-------------------Finish Answering-----------------------')

    #健壮性检查,添加一个检查看看是目录还是文件
    if os.path.isdir(result_path):
        os.makedirs(result_path,exist_ok=True)
        result_path=os.path.join(result_path,'result.txt')
    else:
        os.makedirs(os.path.dirname(result_path) or ".",exist_ok=True)

    #写入文件配置以查看结果
    with open(result_path,'a',encoding='utf-8') as file:
        file.write("-------------------Args Configuration---------------------\n")
        for arg in vars(args):
            file.write(f"{arg}:{getattr(args,arg)}\n")
        file.write("\n\n")

        for (position,num) in true_num.items():
            file.write(f"Position {position} Accuracy: {num / len(dataset)} \n")
        file.write("\n\n")

#仿照sst2进行评测
# def eval_subj(results,dataset,result_path,args):
#     true_num={}
#     for index,answers in enumerate(results):
#         ground_truth=dataset[index]
#         for answer in answers:
#             if str(answer['position']) not in true_num:
#                 true_num[str(answer['position'])]=0
#             if ground_truth in answer['prompt']:
#                 true_num[str(answer['position'])]+=1
#     print('-------------------Finish Answering-----------------------')
    
#     #健壮性检查,添加一个检查看看是目录还是文件
#     if os.path.isdir(result_path):
#         os.makedirs(result_path,exist_ok=True)
#         result_path=os.path.join(result_path,'result.txt')
#     else:
#         os.makedirs(os.path.dirname(result_path) or ".",exist_ok=True)
    
#     with open(result_path,'a',encoding='utf-8') as file:
#         file.write("-------------------Args Configuration---------------------\n")
#         for arg in vars(args):
#             file.write(f"{arg}:{getattr(args,arg)}\n")
#         file.write("\n\n")
#         for (position,num) in true_num.items():
#             file.write(f"Position {position} Accuracy: {num / len(dataset)} \n")
#         file.write("\n\n")

# def eval_trec(results,dataset,result_path,args):
#     true_num={}
#     for index,answers in enumerate(results):
#         ground_truth=dataset[index]
#         for answer in answers:
#             if str(answer['position']) not in true_num:
#                 true_num[str(answer['position'])]=0
#             if ground_truth in answer['prompt']:
#                 true_num[str(answer['position'])]+=1
#     print('-------------------Finish Answering-----------------------')
    
#     #健壮性检查,添加一个检查看看是目录还是文件
#     if os.path.isdir(result_path):
#         os.makedirs(result_path,exist_ok=True)
#         result_path=os.path.join(result_path,'result.txt')
#     else:
#         os.makedirs(os.path.dirname(result_path) or ".",exist_ok=True)

#     with open(result_path,'a',encoding='utf-8') as file:
#         file.write("-------------------Args Configuration---------------------\n")
#         for arg in vars(args):
#             file.write(f"{arg}:{getattr(args,arg)}\n")
#         file.write("\n\n")
#         for (position,num) in true_num.items():
#             file.write(f"Position {position} Accuracy: {num / len(dataset)} \n")
#         file.write("\n\n")

#因为涉及到位置,我一定要加入position这个参数吗,方便分析
def eval(
    task,
    results,
    dataset,
    result_path,
    args,
    correct_letters=None,
    position:int|str=None
    ):
    if task=='sudoku':
        return eval_sudoku(results,dataset,result_path,args,position)
    elif task=='countdown':
        return eval_countdown(results,dataset,result_path,args,position)
    elif task=='math500':
        return eval_math500(results,dataset,result_path,args,position)
    #因为这里已经记录上去了,这里按照传统给了一个文件夹路径,方便记录
    elif task=='mbpp':
        eval_mbpp(results,dataset,result_path,args,position)
        return None
    elif task=='gpqa'and correct_letters is not None:
        eval_gpqa(results,dataset,result_path,args,correct_letters)
        return None
    elif task=='gsm8k':
        return eval_gsm8k(results,dataset,result_path,args,position)
    elif task=='sst2':
        eval_emotion(results,dataset,result_path,args)
        return None
    elif task=='subj':
        eval_emotion(results,dataset,result_path,args)
        return None
    elif task=='trec':
        eval_emotion(results,dataset,result_path,args)
        return None
    elif task=='agnews':
        eval_emotion(results,dataset,result_path,args)
        return None
    else:
        raise NotImplementedError(f"Mode {task} not implemented.")

#results指的是list[list[str]],第一个list是问题的列表,第二个list是答案的列表,总共position+1个候选答案
def eval_position(task,results,dataset,result_dir=None,args=None,nshot=None):
    """
    统一的position评估接口
    
    Args:
        task: 任务名称
        results: list[list[str]], 外层list是问题的列表,内层list是答案的列表,总共position+1个候选答案
        dataset: 数据集
        result_dir: 结果目录（mbpp任务需要）
        args: 参数对象（mbpp任务需要）
        nshot: shot数量（mbpp任务需要）
    Returns:
        accuracy: float, upperbound准确率
    """
    if task=='sudoku':
        return eval_sudoku_position(results,dataset)
    elif task=='countdown':
        return eval_countdown_position(results,dataset)
    elif task=='math500':
        return eval_math500_position(results,dataset)
    #mbpp比较特殊,需要额外参数来生成测试文件,统一接口
    elif task=='mbpp':
        assert result_dir is not None, "result_dir must be provided for mbpp task"
        assert args is not None, "args must be provided for mbpp task"
        assert nshot is not None, "nshot must be provided for mbpp task"
        return eval_mbpp_position(results,dataset,result_dir,args,nshot)
    elif task=='gsm8k':
        return eval_gsm8k_position(results,dataset)
    else:
        raise NotImplementedError(f"Task {task} not supported in eval_position")
