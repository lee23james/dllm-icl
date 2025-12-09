from collections import defaultdict
import os
import subprocess
import concurrent.futures
import sys
import re
from pathlib import Path
import argparse
import json
from datetime import datetime

from sympy import O

def run_python_file(file_path):
    try:
        compile(open(file_path,'rb').read(),file_path,'exec')
        result=subprocess.run(
            [sys.executable,file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stderr:
            return file_path,"RuntimeError",result.stderr.strip()
        else:
            return file_path,"Success",result.stdout.strip()
    except SyntaxError as e:
        return file_path,"SyntaxError",str(e)
    except subprocess.TimeoutExpired:
        return file_path,"Timeout","Execution timed out"
    except Exception as e:
        return file_path,"Error",str(e)

#专门针对nshot所有位置和autoicl进行测评
#对文件进行评测
#对于之前的eval.py运行文件进行扫描并且评测导入相对应的文件夹中(所以在sh脚本中这个是第二行,eval.py是第一行进行运行)
def evaluate_python_files(folder_path, nshot, steps, gen_length, dev_samples_num=None,find_not_position=False, find_best_position=False,iswrite=True, return_json=False, output_path=None):
    """
    评估 Python 文件并返回结果
    
    Args:
        folder_path: 包含 Python 文件的文件夹路径
        nshot: shot 数量
        steps: 步数
        gen_length: 生成长度
        find_not_position: 是否只查找 position_auto-icl 文件夹
        iswrite: 是否写入文件（默认 True）
        return_json: 是否返回 JSON 格式（默认 False）
        output_path: 输出路径（可选）
    
    Returns:
        如果 iswrite=False，返回 {'Accuracy': [accuracy_list]}
        否则返回 None
    """

    # 构建目标文件夹路径：shot_{nshot}_step_{steps}_gen_{gen_length}
    target_folder_name = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
    target_folder = Path(folder_path) / target_folder_name
    
    if not target_folder.exists():
        print(f"Target folder not found: {target_folder}")
        if not iswrite:
            return {'Accuracy': []}
        return
    
    print(f"Searching in folder: {target_folder}")
    
    # 按 position 分组收集 py 文件和执行结果
    position_files = {}  # {position: [list of py files]}
    position_results = {}  # {position: [list of (file_path, status, message)]}
    position_accuracies = []  # 存储每个 position 的准确率
    
    if find_best_position:
        #找不到 best_position
        positions_to_check = [f'best_position_{dev_samples_num}']
    # 根据 find_not_position 参数决定查找哪些 position
    elif find_not_position:
        # 如果指定了 find_not_position，只查找 position_auto-icl 文件夹
        positions_to_check = ['auto-icl']
    else:
        # 否则遍历所有 position (0 到 nshot)
        positions_to_check = list(range(nshot + 1))
    
    # 遍历需要检查的 position
    for position in positions_to_check:
        position_folder = target_folder / f"position_{position}"
        if not position_folder.exists():
            print(f"Warning: position_{position} folder not found, skipping...")
            position_accuracies.append({
                'position': position,
                'accuracy': 0.0,
                'total_files': 0,
                'successful_files': 0
            })
            position_results[position] = []
            continue
        
        # 找到该 position 下时间戳最新的文件夹
        timestamp_folders = []
        for item in position_folder.iterdir():
            if item.is_dir():
                # 检查文件夹名是否符合时间戳格式：YYYYMMDD_HHMMSS
                if re.match(r'^\d{8}_\d{6}$', item.name):
                    timestamp_folders.append(item)
        
        if not timestamp_folders:
            print(f"No timestamp folders found in position_{position}")
            position_accuracies.append({
                'position': position,
                'accuracy': 0.0,
                'total_files': 0,
                'successful_files': 0
            })
            position_results[position] = []
            continue
        
        # 找到最新的时间戳文件夹（按名称排序，最新的在最后）
        latest_timestamp_folder = max(timestamp_folders, key=lambda x: x.name)
        print(f"Using latest timestamp folder: {latest_timestamp_folder.name} in position_{position}")
        
        # 在该文件夹下查找所有 py 文件
        py_files = [str(p) for p in latest_timestamp_folder.rglob("*.py")]
        position_files[position] = py_files
        
        if not py_files:
            print(f"No Python files found in position_{position}")
            position_accuracies.append({
                'position': position,
                'accuracy': 0.0,
                'total_files': 0,
                'successful_files': 0
            })
            position_results[position] = []
            continue
        
        print(f"Found {len(py_files)} Python files in position_{position}, starting execution...")
        
        # 执行该 position 下的所有 py 文件
        success_count = 0
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
            future_to_file = {executor.submit(run_python_file, f): f for f in py_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, status, message = future.result()
                results.append((file_path, status, message))
                
                if status == "Success":
                    success_count += 1
                
                # print(f"{file_path}: {status}")
        
        # 保存该 position 的结果
        position_results[position] = results
        
        # 计算该 position 的准确率
        accuracy = (success_count / len(py_files)) * 100 if len(py_files) > 0 else 0.0
        position_accuracies.append({
            'position': position,
            'accuracy': round(accuracy, 2),
            'total_files': len(py_files),
            'successful_files': success_count
        })
        
        print(f"Position {position} accuracy: {accuracy:.2f}% ({success_count}/{len(py_files)})")
    
    # 如果 iswrite=False，只返回 Accuracy 列表，不写入文件
    if not iswrite:
        accuracy_list = [acc['accuracy'] for acc in position_accuracies]
        return {'Accuracy': accuracy_list}
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 生成 JSON 文件（如果指定了 return_json）
    if return_json:
        if output_path:
            # 构建 JSON 文件名：step_{steps}_gen_{gen_length}_nshot_{nshot}_accuracy_{timestamp}.json
            json_filename = f"step_{steps}_gen_{gen_length}_nshot_{nshot}_accuracy_{timestamp}.json"
            json_output_path = Path(output_path) / json_filename
            json_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # 如果没有指定 output_path，保存到目标文件夹
            json_output_path = target_folder / f"step_{steps}_gen_{gen_length}_nshot_{nshot}_accuracy_{timestamp}.json"
        
        # 构建 JSON 数据
        json_data = {
            'nshot': nshot,
            'steps': steps,
            'gen_length': gen_length,
            'timestamp': datetime.now().isoformat(),
            'Accuracy': [acc['accuracy'] for acc in position_accuracies],  # 每个 position 的准确率列表
            'position_details': position_accuracies,  # 详细信息
            'summary': {
                'total_positions': len(position_accuracies),
                'overall_accuracy': round(sum(acc['accuracy'] for acc in position_accuracies) / len(position_accuracies), 2) if position_accuracies else 0.0
            }
        }
        
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON results saved to: {json_output_path}")
        print(f"Overall accuracy across all positions: {json_data['summary']['overall_accuracy']:.2f}%")
    
    # 无论是否保存 JSON，都要保存文本文件
    # 在 target_folder/accuracy 下创建文件夹（即 shot_{nshot}_step_{steps}_gen_{gen_length}/accuracy）
    accuracy_base_dir = target_folder / "accuracy"
    accuracy_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个 position 创建子文件夹并保存结果
    for position in sorted(position_results.keys()):
        position_dir = accuracy_base_dir / f"position_{position}"
        position_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存该 position 的详细结果到 accuracy_{timestamp}.txt
        txt_output_path = position_dir / f"accuracy_{timestamp}.txt"
        
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("=== Execution Results Summary ===\n")
            f.write(f"Position: {position}\n")
            f.write(f"nshot: {nshot}\n")
            f.write(f"steps: {steps}\n")
            f.write(f"gen_length: {gen_length}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
            
            # 找到该 position 的准确率信息
            position_acc = next((acc for acc in position_accuracies if acc['position'] == position), None)
            if position_acc:
                f.write(f"Accuracy: {position_acc['accuracy']:.2f}% ({position_acc['successful_files']}/{position_acc['total_files']})\n\n")
            
            f.write("=== Detailed Results ===\n")
            for file_path, status, message in position_results[position]:
                f.write(f"File: {file_path}\n")
                f.write(f"Status: {status}\n")
                if message:
                    f.write(f"Message: {message}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Position {position} results saved to: {txt_output_path}")
    
    # 打印总体统计
    overall_acc = sum(acc['accuracy'] for acc in position_accuracies) / len(position_accuracies) if position_accuracies else 0.0
    print(f"\nExecution completed!")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print("\nPosition-wise accuracy:")
    for acc in position_accuracies:
        print(f"  Position {acc['position']}: {acc['accuracy']:.2f}% ({acc['successful_files']}/{acc['total_files']})")
    
    return None



def evaluate_python_files_positions(folder_path, nshot, steps, gen_length):
       # 构建目标文件夹路径：shot_{nshot}_step_{steps}_gen_{gen_length}
       #这里计算一下应该加进去的东西
    target_folder_name = f"shot_{nshot}_step_{steps}_gen_{gen_length}"
    target_folder = Path(folder_path) / target_folder_name
    
    if not target_folder.exists():
        print(f"Target folder not found: {target_folder}")
        return 0
    
    print(f"Searching in folder: {target_folder}") 

    # 按 position 分组收集 py 文件和执行结果
    #这里计算的是upperbound,在这道题上每一个问题的正确率
    files_by_index = defaultdict(list)
    # upperbound_acc=[]

    positions_to_check = list(range(nshot + 1))
    for position in positions_to_check:
        position_folder = target_folder / f"position_{position}"
        if not position_folder.exists():
            print(f"Warning: position_{position} folder not found, skipping...")
            continue
    
        # 找到该 position 下时间戳最新的文件夹,寻找最新的时间戳文件夹
        timestamp_folders = []
        for item in position_folder.iterdir():
            if item.is_dir():
                # 检查文件夹名是否符合时间戳格式：YYYYMMDD_HHMMSS
                if re.match(r'^\d{8}_\d{6}$', item.name):
                    timestamp_folders.append(item)

        #这里进行标记
        if not timestamp_folders:
            print(f"No timestamp folders found in position_{position}")
            continue

        # 找到最新的时间戳文件夹（按名称排序，最新的在最后）
        latest_timestamp_folder = max(timestamp_folders, key=lambda x: x.name)
        print(f"Using latest timestamp folder: {latest_timestamp_folder.name} in position_{position}")

        for py_file in latest_timestamp_folder.glob("test_index_*.py"):
            #提取index
            match = re.search(r'test_index_(\d+)\.py', py_file.name)
            if match:
                idx = match.group(1)
                files_by_index[idx].append(py_file)
    
    #准备并行执行列表
    #需要运行所有文件来判断状态
    all_files_to_run=[]
    for idx,file_list in files_by_index.items():
        all_files_to_run.extend(file_list)
    
    print(f"Total files to execute: {len(all_files_to_run)} across {len(files_by_index)} unique indices.")

    # 字典用于记录每个文件的运行结果: { file_path: status }
    execution_results = {}

    #并行执行所有文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        future_to_file= {executor.submit(run_python_file, f): f for f in all_files_to_run}

        try:
            from tqdm import tqdm
            iterator = tqdm(concurrent.futures.as_completed(future_to_file), total=len(all_files_to_run))
        except ImportError:
            iterator=concurrent.futures.as_completed(future_to_file)

        for future in iterator:
            file_path, status, message = future.result()
            execution_results[file_path] = status
    
    #计算upperbound acc
    solved_indices_count=0
    total_indices=len(files_by_index)

    for idx,file_list in files_by_index.items():
        is_solved=False
        for file_path in file_list:
            if execution_results.get(file_path) == "Success":
                is_solved=True
                break
        if is_solved:
            solved_indices_count+=1
    
    print(f"\n===== Result =====")
    print(f"Total Unique Indices: {total_indices}")
    print(f"Solved Indices: {solved_indices_count}")
    if total_indices>0:
        accuracy=solved_indices_count/total_indices
        print(f"Upperbound Accuracy: {accuracy:.2%}")
    else:
        accuracy=0.0
        print("No indices found.")
    return accuracy


#把main上的文件改成了可以通过一次运行实现
def main():
    parser= argparse.ArgumentParser(description='Run Python files in a folder and save results.')
    parser.add_argument('--folder_path',type=str,help='Path to the folder containing Python files')
    parser.add_argument('--output_path',type=str,help='Path to the TXT file to save results')
    parser.add_argument('--return_json',action='store_true',default=False,help='If True, save results as JSON format with Accuracy and timestamp')
    #需要对找到的文案进行一下归类
    parser.add_argument('--nshot', type=int, default=3 )
    parser.add_argument('--steps', type=int, default=128, help='Number of steps')
    parser.add_argument('--gen_length', type=int, default=128, help='Generation length')
    parser.add_argument('--find_not_position',action='store_true',default=False,help='If True, search in position_auto-icl folder; otherwise search all position folders')
    parser.add_argument('--no-write',action='store_true',default=False,help='If set, do not write results to files, only return Accuracy list')
    args=parser.parse_args()
    
    evaluate_python_files(
        folder_path=args.folder_path,
        nshot=args.nshot,
        steps=args.steps,
        gen_length=args.gen_length,
        find_not_position=args.find_not_position,
        iswrite=not args.no_write,  # 如果提供了 --no-write，则 iswrite=False
        return_json=args.return_json,
        output_path=args.output_path
    )

if __name__=="__main__":
    main()