"""
Dream-7B 模型评估脚本 - 修复版
适配 Dream 的扩散生成特性，通过调整 shots 顺序实现位置变化
"""
import random, os, sys
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.eval_utils import query_extract, load_dataset, eval
from utils.judge_python_code import evaluate_python_files


def generate_dream_fixed(model, tokenizer, input_data, task, steps, gen_length, temperature, query_position, nshot=None):
    """
    为 Dream 构造适合的 prompt
    策略：根据 position 选择不同的 shots 组合，但 query 始终在末尾
    """
    # 获取基础 prompt（使用 position=0 获取完整结构）
    query_full = query_extract(input_data, task, 0, gen_length, nshot, False)
    
    # 移除 mask tokens
    query_clean = query_full.replace("<|mdm_mask|>", "")
    
    # 根据 position 调整：选择前 (nshot - position) 个 shots
    # 这样模拟"query 在不同位置"的效果
    # position=0: 使用所有 shots
    # position=4: 使用 0 个 shots（纯 zero-shot）
    effective_nshot = nshot - query_position
    if effective_nshot < 0:
        effective_nshot = 0
    
    # 分割 prompt，只保留前 effective_nshot 个 shots + query
    # 找到 front_prompt 结束位置
    if task == 'gsm8k':
        front_end = query_clean.find("question: There are 15 trees")
        front_prompt = query_clean[:front_end].strip()
        
        # 找到各个 shot 的位置
        shot_markers = [
            "question: There are 15 trees",
            "question: If there are 3 cars", 
            "question: Olivia has",
            "question: Michael had",
            "question: Jason had",
            "question: Shawn has",
            "question: There were nine computers",
            "question: Leah had"
        ]
        
        # 收集有效的 shots
        shots = []
        for i, marker in enumerate(shot_markers[:effective_nshot]):
            start = query_clean.find(marker)
            if start == -1:
                continue
            # 找到这个 shot 的结束位置（下一个 shot 或 query）
            if i + 1 < len(shot_markers):
                next_marker = shot_markers[i + 1]
                end = query_clean.find(next_marker)
            else:
                # 最后一个 shot，找到 query
                end = query_clean.find("question: Janet")
            if end == -1:
                end = len(query_clean)
            shot_content = query_clean[start:end].strip()
            shots.append(shot_content)
        
        # 找到 query
        query_start = query_clean.find("question: Janet")
        query_end = query_clean.find("target:", query_start)
        if query_end == -1:
            query_end = len(query_clean)
        query_content = query_clean[query_start:query_end].strip()
        
        # 构造最终 prompt
        final_prompt = front_prompt + "\n\n" + "\n\n".join(shots) + "\n\n" + query_content + "\nAnswer: "
        
    elif task == 'math500':
        # Math500 类似处理
        query_clean = query_clean.replace("target:", "Answer:")
        final_prompt = query_clean
    elif task == 'mbpp':
        query_clean = query_clean.replace("target:", "Answer:")
        final_prompt = query_clean
    elif task == 'sudoku':
        query_clean = query_clean.replace("target:", "Answer:")
        final_prompt = query_clean
    elif task == 'countdown':
        query_clean = query_clean.replace("target:", "Answer:")
        final_prompt = query_clean
    else:
        query_clean = query_clean.replace("target:", "Answer:")
        final_prompt = query_clean
    
    # 编码 prompt
    inputs = tokenizer(final_prompt, return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # 截断过长的 prompt
    max_length = 2048
    max_prompt_len = max_length - gen_length
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
        attention_mask = attention_mask[:, -max_prompt_len:]
    
    # 生成
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
            return_dict_in_generate=True,
        )
    
    # 解码
    generated_ids = output.sequences[0]
    input_length = input_ids.shape[1]
    generated_text = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
    generated_text = generated_text.split(tokenizer.eos_token)[0]
    
    # 截断后续内容
    if "Question:" in generated_text:
        generated_text = generated_text.split("Question:")[0]
    elif "question:" in generated_text.lower():
        idx = generated_text.lower().find("question:")
        generated_text = generated_text[:idx]
    
    return generated_text.strip()


def main(args):
    random.seed(args.seed)
    
    task = args.task
    model_name = args.model_name
    device = args.device
    gen_length = args.gen_length
    steps = args.steps
    temperature = args.temperature
    data_path = args.data_path
    result_path = args.result_path
    max_samples = args.max_samples
    nshot = args.nshot
    
    dataset = load_dataset(data_path, task, max_samples)
    print(f'------------------Load dataset: {len(dataset)} samples------------------')
    
    print('------------------Load Dream model------------------')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model = model.to(device).eval()
    print('------------------Start Answering------------------')
    
    if args.query_position is not None:
        position_list = [args.query_position]
    else:
        position_list = list(range(nshot + 1))
    
    for query_position in position_list:
        results = []
        
        dataset_iter = tqdm(dataset, desc=f"Position {query_position}")
        
        for input_data in dataset_iter:
            answer = generate_dream_fixed(model, tokenizer, input_data, task, steps,
                                      gen_length, temperature, query_position, nshot)
            results.append(answer)
        
        # 评估
        if task != 'mbpp':
            eval(task, results, dataset, result_path, args, position=query_position)
        else:
            from utils.eval_utils import eval_mbpp
            eval_mbpp(results, dataset, result_path, args, position=query_position)
    
    if task == 'mbpp':
        evaluate_python_files(
            folder_path=result_path,
            nshot=nshot,
            steps=steps,
            gen_length=gen_length,
            find_not_position=False,
            find_random_position=False,
            iswrite=True,
            output_path=None,
        )
    
    print('-------------------Finish----------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--model_name', type=str, default='/hy-tmp/dllm-icl/model/Dream-7B-Base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./data/gsm8k.jsonl')
    parser.add_argument('--result_path', type=str, default='./results/dream/gsm8k')
    parser.add_argument('--query_position', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--nshot', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    
    main(args)
