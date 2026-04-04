"""数据集拆分工具"""
import json
import sys
import os

def split_jsonl(input_path, output_a, output_b):
    """将 jsonl 文件平均拆分为两份"""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    mid = len(lines) // 2
    
    with open(output_a, 'w') as f:
        f.writelines(lines[:mid])
    
    with open(output_b, 'w') as f:
        f.writelines(lines[mid:])
    
    print(f"Split {len(lines)} samples: Part A={mid}, Part B={len(lines)-mid}")
    return mid, len(lines)-mid

def split_csv(input_path, output_a, output_b):
    """将 csv 文件平均拆分为两份"""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0] if lines else ""
    data_lines = lines[1:] if lines else []
    
    mid = len(data_lines) // 2
    
    with open(output_a, 'w') as f:
        f.write(header)
        f.writelines(data_lines[:mid])
    
    with open(output_b, 'w') as f:
        f.write(header)
        f.writelines(data_lines[mid:])
    
    print(f"Split {len(data_lines)} samples: Part A={mid}, Part B={len(data_lines)-mid}")
    return mid, len(data_lines)-mid

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python split_dataset.py <input_file> <output_a> <output_b>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_a = sys.argv[2]
    output_b = sys.argv[3]
    
    if input_path.endswith('.csv'):
        split_csv(input_path, output_a, output_b)
    else:
        split_jsonl(input_path, output_a, output_b)
