"""
Dream-7B 模型评估脚本
适配 Dream 的 diffusion_generate API 以支持不同位置的评测
"""
import random, os, sys, time
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.eval_utils import query_extract, load_dataset, eval, group_by_label, corpus_sampling, eval_position
from utils.judge_python_code import evaluate_python_files

# 检查 accelerate 是否可用
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available, multi-GPU support disabled")


def generate_dream(model, tokenizer, input_data, task, steps, gen_length, temperature, situation, query_position, nshot=None, prebuilt_prompt=None, max_length=2048):
    """
    使用 Dream 模型生成答案
    """
    if prebuilt_prompt is not None:
        query = prebuilt_prompt
    else:
        # 构造 query（根据位置）
        query_full = query_extract(input_data, task, query_position, gen_length, nshot, False)
        
        # Dream Base 模型适配：将 LLaDA 格式转换为 Dream 友好的格式
        # 1. 移除所有 mask tokens
        query_clean = query_full.replace("<|mdm_mask|>", "")
        # 2. 将 "target:" 改为 "Answer:" 并添加换行
        query_clean = query_clean.replace("target:", "Answer:")
        # 3. 清理多余的空白
        import re
        query_clean = re.sub(r'\n\s*\n', '\n\n', query_clean)
        query = query_clean.strip()

    if situation == 'base':
        user_input = query
    elif situation == 'instruct':
        messages = [{"role": "user", "content": query}]
        user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        raise ValueError(f"Unknown situation: {situation}")

    # 编码 prompt
    inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # 截断过长的 prompt
    max_prompt_len = max_length - gen_length
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
        attention_mask = attention_mask[:, -max_prompt_len:]
    
    # 使用 Dream 的 diffusion_generate
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
    
    # 解码生成的序列
    generated_ids = output.sequences[0]
    input_length = input_ids.shape[1]
    generated_text = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
    generated_text = generated_text.split(tokenizer.eos_token)[0]
    
    # 额外截断：Dream 不会自动生成 EOS，需要按任务特定的下一个样例前缀截断
    stop_markers = []
    if task == "sudoku":
        stop_markers = ["\n\nPuzzle:", "\nPuzzle:"]
    elif task == "countdown":
        stop_markers = ["\n\nQuestion:", "\nQuestion:"]
    elif task == "math500":
        stop_markers = ["\n\nQ:", "\nQ:"]
    elif task == "gsm8k":
        stop_markers = ["\n\nquestion:", "\nquestion:"]
    elif task == "mbpp":
        stop_markers = ["\n\nYou are an expert Python programmer", "\nYou are an expert Python programmer"]

    for marker in stop_markers:
        if marker in generated_text:
            generated_text = generated_text.split(marker)[0]
            break

    if "Question:" in generated_text:
        generated_text = generated_text.split("Question:")[0]
    elif "question:" in generated_text.lower():
        idx = generated_text.lower().find("question:")
        generated_text = generated_text[:idx]
    
    generated_text = generated_text.strip()

    return generated_text


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
    
    accelerator = None
    if ACCELERATE_AVAILABLE:
        if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
            accelerator = accelerate.Accelerator()
            print(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] Accelerate initialized")
    
    full_dataset = load_dataset(data_path, task, max_samples)
    
    if accelerator is not None:
        with accelerator.split_between_processes(full_dataset) as subset_dataset:
            dataset = subset_dataset
            print(f'[Rank {accelerator.process_index}] Load dataset: {len(dataset)} samples (total: {len(full_dataset)})')
    else:
        dataset = full_dataset
        print('------------------Load dataset------------------')
    
    print('------------------Load Dream model------------------')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    if accelerator is not None:
        device = accelerator.device
        print(f"[Rank {accelerator.process_index}] Loading model on {device}")
        model = model.to(device)
    else:
        model = model.to(device)
    
    model.eval()
    print('------------------Start Answering------------------')
    
    if args.query_position is not None:
        if nshot is None:
            raise ValueError("--nshot must be provided when using --query_position")
        if not (0 <= args.query_position <= nshot):
            raise ValueError(f"--query_position must be in [0, nshot], got {args.query_position} (nshot={nshot})")
        position_list = [args.query_position]
    else:
        if nshot is None:
            raise ValueError("--nshot must be provided to enumerate all positions (0..nshot)")
        position_list = list(range(nshot + 1))
    
    for query_position in position_list:
        results = []
        correct_letters = []
        total_inference_time = 0.0
        total_generation_count = 0
        
        dataset_iter = dataset
        if accelerator is not None and accelerator.num_processes > 1:
            if accelerator.is_main_process:
                dataset_iter = tqdm(dataset, desc=f"[Rank {accelerator.process_index}] Position {query_position}")
        else:
            dataset_iter = tqdm(dataset, desc=f"Position {query_position}")
        
        for input_data in dataset_iter:
            situation = 'instruct' if 'Instruct' in model_name else 'base'

            if task == 'gpqa':
                prompts, correct_letter = query_extract(input_data, task, query_position, gen_length, nshot)
                answers = []
                for query in prompts:
                    _t0 = time.time()
                    answer = generate_dream(model, tokenizer, {'question': query, 'answer': input_data.get('answer', '')},
                                          task, steps, gen_length, temperature, situation, query_position, nshot)
                    total_inference_time += time.time() - _t0
                    total_generation_count += 1
                    answers.append(answer)
                correct_letters.append(correct_letter)
                results.append(answers)
            else:
                _t0 = time.time()
                answer = generate_dream(model, tokenizer, input_data, task, steps,
                                      gen_length, temperature, situation, query_position, nshot)
                total_inference_time += time.time() - _t0
                total_generation_count += 1
                results.append(answer)
        
        if accelerator is not None and accelerator.num_processes > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                all_results_list = [None] * accelerator.num_processes
                dist.all_gather_object(all_results_list, results)
                
                if accelerator.is_main_process:
                    merged_results = []
                    for proc_results in all_results_list:
                        if proc_results is not None:
                            merged_results.extend(proc_results)
                    results = merged_results
                    
                    if task == 'gpqa':
                        all_correct_letters_list = [None] * accelerator.num_processes
                        dist.all_gather_object(all_correct_letters_list, correct_letters)
                        merged_correct_letters = []
                        for proc_correct_letters in all_correct_letters_list:
                            if proc_correct_letters is not None:
                                merged_correct_letters.extend(proc_correct_letters)
                        correct_letters = merged_correct_letters
                else:
                    continue
        
        if accelerator is None or accelerator.is_main_process:
            avg_latency = total_inference_time / total_generation_count if total_generation_count > 0 else 0.0
            print(f"\n{'='*50}")
            print(f"Time Statistics (position={query_position}):")
            print(f"  Total Inference Time   : {total_inference_time:.2f} s")
            print(f"  Total Generation Count : {total_generation_count}")
            print(f"  Avg Latency per Query  : {avg_latency:.4f} s/query")
            print(f"{'='*50}\n")

        if task != 'mbpp':
            if accelerator is None or accelerator.is_main_process:
                if task == 'gpqa':
                    eval(task, results, full_dataset, result_path, args, correct_letters, position=query_position)
                else:
                    eval(task, results, full_dataset, result_path, args, position=query_position)
        else:
            if accelerator is None or accelerator.is_main_process:
                from utils.eval_utils import eval_mbpp
                result_path = getattr(args, 'result_path', None)
                if result_path is None:
                    result_path = f'./results/{task}_results'
                eval_mbpp(results, full_dataset, result_path, args, position=query_position)
    
    if task == 'mbpp':
        if accelerator is None or accelerator.is_main_process:
            result_path = getattr(args, 'result_path', None)
            if result_path is None:
                result_path = f'./results/{task}_results'
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
    
    if accelerator is None or accelerator.is_main_process:
        print('-------------------Finish----------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sudoku', choices=['sudoku', 'countdown', 'math500', 'gpqa', 'gsm8k', 'mbpp'])
    parser.add_argument('--model_name', type=str, default='/hy-tmp/dllm-icl/model/Dream-7B-Base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./data/sudoku.csv')
    parser.add_argument('--result_path', type=str, default='./results/dream/sudoku_results')
    parser.add_argument('--query_position', type=int, default=None,
                        help='If set, only evaluate this single position (0..nshot). Default: evaluate all positions.')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--nshot', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    
    try:
        main(args)
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
