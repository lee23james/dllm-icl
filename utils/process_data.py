import pandas as pd
import json
import os

def convert_mbpp_validation_parquet(input_file, output_file, indent=4, preview=True):
    """
    专门用于将 MBPP validation parquet 文件转换为 mbpp_dev.json 格式。
    
    参数:
        input_file (str): validation parquet 文件的路径。
        output_file (str): 输出 JSON 文件的路径（例如 data/mbpp_dev.json）。
        indent (int, optional): JSON 的缩进空格数，默认为 4。
        preview (bool, optional): 是否在转换完成后打印第一条数据以供检查。
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 -> {input_file}")
        return
    
    try:
        print(f"正在读取: {input_file} ...")
        df = pd.read_parquet(input_file)
        
        print(f"原始列名: {df.columns.tolist()}")
        print(f"数据行数: {len(df)}")
        
        # 2. 检查并映射列名（MBPP 数据集常见的列名映射）
        # 常见的 HuggingFace MBPP 数据集列名可能是：text, prompt, code, test_list, test_imports 等
        column_mapping = {}
        
        # 检查可能的列名变体
        if 'text' in df.columns and 'prompt' not in df.columns:
            column_mapping['text'] = 'prompt'
        if 'solution' in df.columns and 'code' not in df.columns:
            column_mapping['solution'] = 'code'
        
        if column_mapping:
            print(f"正在重命名列: {column_mapping}")
            df = df.rename(columns=column_mapping)
        
        # 3. 确保必要的字段存在
        required_fields = ['prompt', 'task_id']
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            print(f"警告: 缺少以下字段: {missing_fields}")
            print(f"可用字段: {df.columns.tolist()}")
        
        # 4. 转换为字典列表格式（与 mbpp.json 格式一致）
        records = []
        for idx, row in df.iterrows():
            record = {}
            
            # 复制所有字段
            for col in df.columns:
                value = row[col]
                
                # 先检查是否是数组类型（需要特殊处理）
                is_array = hasattr(value, '__len__') and hasattr(value, '__iter__') and not isinstance(value, (str, dict))
                
                # 处理 pandas 的特殊类型（如 NaN）- 只对非数组类型使用 pd.isna
                if not is_array:
                    try:
                        if pd.isna(value):
                            record[col] = None
                            continue
                    except (ValueError, TypeError):
                        # 如果 pd.isna 失败（可能是数组），继续后续处理
                        pass
                
                # 如果是列表或字典，直接使用
                if isinstance(value, (list, dict)):
                    record[col] = value
                # 如果是 numpy 数组或 pandas Series
                elif is_array:
                    try:
                        # 尝试转换为列表
                        value_list = list(value)
                        if len(value_list) == 1:
                            # 长度为 1 的数组，尝试提取标量
                            try:
                                record[col] = value.item()
                            except (ValueError, AttributeError):
                                record[col] = value_list[0]
                        else:
                            # 长度 > 1 的数组，转换为列表
                            record[col] = value_list
                    except (TypeError, ValueError):
                        # 如果转换失败，尝试直接使用
                        record[col] = value
                else:
                    # 标量值，直接使用（pandas 会自动转换为 Python 原生类型）
                    try:
                        # 尝试转换为 Python 原生类型
                        if hasattr(value, 'item'):
                            record[col] = value.item()
                        else:
                            record[col] = value
                    except (ValueError, AttributeError):
                        record[col] = value
            
            records.append(record)
        
        # 5. 保存为 JSON
        print(f"正在保存为: {output_file} ...")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=indent)
        
        print(f"✅ 转换成功！包含 {len(records)} 条数据。")
        
        # 6. (可选) 预览验证
        if preview and records:
            print("-" * 50)
            print("数据预览 (第1条):")
            print(json.dumps(records[0], ensure_ascii=False, indent=indent))
            print("-" * 50)
            print(f"字段列表: {list(records[0].keys())}")
            
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        print("\n请安装 pyarrow 或 fastparquet:")
        print("  方式1 (推荐): conda activate llada-icl && pip install pyarrow")
        print("  方式2: pip install pyarrow")
        print("  方式3: pip install fastparquet")
        print("\n如果使用 conda 环境，请先激活: conda activate llada-icl")
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

#在这里将这段文件转化为json形式
def parquet_to_json(input_file, output_file, rename_map=None, indent=4, preview=True):
    """
    将 Parquet 文件转换为指定格式的 JSON (List of Records)。

    参数:
        input_file (str): Parquet 文件的路径。
        output_file (str): 输出 JSON 文件的路径。
        rename_map (dict, optional): 列名重写字典。格式: {'旧列名': '新列名'}。
                                     如果不需要改名，保持为 None。
        indent (int, optional): JSON 的缩进空格数，默认为 4。设为 None 则不缩进（文件更小）。
        preview (bool, optional): 是否在转换完成后打印第一条数据以供检查。
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 -> {input_file}")
        return

    try:
        print(f"正在读取: {input_file} ...")
        df = pd.read_parquet(input_file)
        
        # 2. (可选) 重命名列
        if rename_map:
            print(f"正在重命名列: {rename_map}")
            df = df.rename(columns=rename_map)
            
        # 3. 转换为 JSON
        print(f"正在保存为: {output_file} ...")
        df.to_json(
            output_file, 
            orient='records', 
            force_ascii=False, 
            indent=indent
        )
        
        print(f"✅ 转换成功！包含 {len(df)} 条数据。")

        # 4. (可选) 预览验证
        if preview:
            print("-" * 30)
            print("数据预览 (第1条):")
            with open(output_file, 'r', encoding='utf-8') as f:
                # 只读取一小部分来解析，防止文件过大卡死
                # 这里为了简单，我们重新读一下生成的文件的第一项
                # 注意：如果文件巨大，建议只读前几行字符
                data = json.load(f)
                print(json.dumps(data[0], ensure_ascii=False, indent=indent))
            print("-" * 30)

    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")

# ==========================================
# 使用示例
# ==========================================

def convert_list_to_taskid_dict(input_file, output_file, task_id_key='task_id', indent=4, use_task_id_as_key=True):
    """
    将列表格式的 JSON 文件转换为字典格式。
    
    参数:
        input_file (str): 输入的 JSON 文件路径（列表格式）。
        output_file (str): 输出的 JSON 文件路径（字典格式）。
        task_id_key (str): 用作键的字段名，默认为 'task_id'。如果 use_task_id_as_key=False，则使用序号。
        indent (int): JSON 缩进，默认为 4。
        use_task_id_as_key (bool): 是否使用 task_id 作为键，True 使用 task_id，False 使用序号（1,2,3...）。
    """
    try:
        print(f"正在读取: {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"警告: 输入文件不是列表格式，当前类型: {type(data)}")
            return
        
        print(f"原始数据: {len(data)} 条记录")
        
        # 转换为字典
        result_dict = {}
        for idx, item in enumerate(data, start=1):
            if use_task_id_as_key:
                if task_id_key not in item:
                    print(f"警告: 记录缺少 '{task_id_key}' 字段，使用序号 {idx} 作为键")
                    result_dict[idx] = item
                else:
                    task_id = item[task_id_key]
                    # 确保 task_id 是整数（JSON 键会自动转换为字符串，但读取时会恢复）
                    if isinstance(task_id, (int, float)):
                        result_dict[int(task_id)] = item
                    else:
                        result_dict[task_id] = item
            else:
                # 使用序号作为键（与 load_json_or_jsonl 的行为一致）
                result_dict[idx] = item
        
        print(f"转换后: {len(result_dict)} 条记录（以 {'task_id' if use_task_id_as_key else '序号'} 为键）")
        
        # 保存为 JSON
        print(f"正在保存为: {output_file} ...")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=indent)
        
        print(f"✅ 转换成功！")
        print(f"示例键: {list(result_dict.keys())[:5]}")
        print(f"键类型: {type(list(result_dict.keys())[0]).__name__}")
        
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def convert_dict_to_list(input_file, output_file, indent=4):
    """
    将字典格式的 JSON 文件转换回列表格式（与 mbpp.json 格式一致）。
    
    参数:
        input_file (str): 输入的 JSON 文件路径（字典格式）。
        output_file (str): 输出的 JSON 文件路径（列表格式）。
        indent (int): JSON 缩进，默认为 4。
    """
    try:
        print(f"正在读取: {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print("文件已经是列表格式，无需转换")
            return
        
        if not isinstance(data, dict):
            print(f"错误: 输入文件格式不支持，当前类型: {type(data)}")
            return
        
        print(f"原始数据: {len(data)} 条记录（字典格式）")
        
        # 转换为列表（按 task_id 排序）
        result_list = []
        for key, value in data.items():
            result_list.append(value)
        
        # 按 task_id 排序
        if result_list and 'task_id' in result_list[0]:
            result_list.sort(key=lambda x: x.get('task_id', 0))
        
        print(f"转换后: {len(result_list)} 条记录（列表格式）")
        
        # 保存为 JSON
        print(f"正在保存为: {output_file} ...")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=indent)
        
        print(f"✅ 转换成功！")
        if result_list:
            print(f"示例第一条 task_id: {result_list[0].get('task_id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # 默认转换 validation parquet
    if len(sys.argv) == 1:
        # 转换 MBPP validation parquet 文件
        input_path = "data/validation-00000-of-00001.parquet"
        output_path = "data/mbpp_dev.json"
        print("=" * 60)
        print("转换 validation parquet 文件")
        print("=" * 60)
        convert_mbpp_validation_parquet(input_path, output_path)
    elif len(sys.argv) == 3:
        # 命令行参数：输入文件 输出文件
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        print("=" * 60)
        print(f"转换文件: {input_path} -> {output_path}")
        print("=" * 60)
        convert_mbpp_validation_parquet(input_path, output_path)
    else:
        print("用法:")
        print("  python utils/process_data.py                    # 转换 validation parquet")
        print("  python utils/process_data.py <input> <output>   # 转换指定文件")
        print("\n示例:")
        print("  python utils/process_data.py data/prompt-00000-of-00001.parquet data/mbpp_prompt.json")
    
    # ==========================================
    # 其他使用示例（已注释）
    # ==========================================

    # 场景 1: 直接转换（通用函数）
    # parquet_to_json("train.parquet", "train_converted.json")

    # 场景 2: 需要改名 (例如把 'question' 改成 'prompt')
    # my_rename_dict = {
    #     "question": "prompt", 
    #     "answer": "code"
    # }
    # parquet_to_json("train.parquet", "train_converted.json", rename_map=my_rename_dict)

