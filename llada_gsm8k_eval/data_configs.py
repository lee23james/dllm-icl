"""
GSM8K数据配置
定义不同数据源的配置
"""

import os
from typing import Dict, Any


class DataConfig:
    """数据配置类"""
    
    # 预定义的数据源配置
    CONFIGS = {
        "huggingface": {
            "source": "huggingface",
            "description": "从Hugging Face Hub加载GSM8K数据集",
            "requirements": ["datasets"],
            "example_usage": "evaluator = GSM8KEvaluator(data_source='huggingface')"
        },
        
        "local_json": {
            "source": "local",
            "description": "从本地JSON文件加载",
            "requirements": ["json"],
            "file_patterns": {
                "train": ["train.json", "gsm8k_train.json", "train_data.json"],
                "test": ["test.json", "gsm8k_test.json", "test_data.json"]
            },
            "example_usage": "evaluator = GSM8KEvaluator(data_source='local', data_path='/path/to/data_dir')"
        },
        
        "local_jsonl": {
            "source": "local", 
            "description": "从本地JSONL文件加载",
            "requirements": ["json"],
            "file_patterns": {
                "train": ["train.jsonl", "gsm8k_train.jsonl", "train_data.jsonl"],
                "test": ["test.jsonl", "gsm8k_test.jsonl", "test_data.jsonl"]
            },
            "example_usage": "evaluator = GSM8KEvaluator(data_source='local', data_path='/path/to/data_dir')"
        },
        
        "custom_directory": {
            "source": "directory",
            "description": "从自定义目录加载（自动检测文件格式）",
            "requirements": ["json", "os"],
            "supported_formats": ["json", "jsonl", "txt"],
            "example_usage": "evaluator = GSM8KEvaluator(data_source='directory', data_path='/path/to/gsm8k_data')"
        }
    }
    
    @classmethod
    def get_config(cls, config_name: str) -> Dict[str, Any]:
        """获取指定配置"""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"未知配置: {config_name}. 可用配置: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[config_name]
    
    @classmethod
    def list_configs(cls) -> None:
        """列出所有可用配置"""
        print("可用的数据源配置:")
        print("=" * 50)
        
        for name, config in cls.CONFIGS.items():
            print(f"配置名称: {name}")
            print(f"描述: {config['description']}")
            print(f"数据源: {config['source']}")
            
            if 'file_patterns' in config:
                print("支持的文件模式:")
                for split, patterns in config['file_patterns'].items():
                    print(f"  {split}: {', '.join(patterns)}")
            
            if 'supported_formats' in config:
                print(f"支持的格式: {', '.join(config['supported_formats'])}")
            
            print(f"使用示例: {config['example_usage']}")
            print("-" * 50)
    
    @classmethod
    def validate_data_path(cls, data_source: str, data_path: str) -> bool:
        """
        验证数据路径是否有效
        
        Args:
            data_source: 数据源类型
            data_path: 数据路径
            
        Returns:
            是否有效
        """
        if data_source == "huggingface":
            return True  # Hugging Face不需要本地路径
        
        if not os.path.exists(data_path):
            print(f"错误: 数据路径不存在: {data_path}")
            return False
        
        if data_source == "directory":
            # 检查目录中是否有train和test文件
            config = cls.get_config("custom_directory")
            found_files = []
            
            for file_format in config['supported_formats']:
                train_files = [f"train.{file_format}", f"gsm8k_train.{file_format}"]
                test_files = [f"test.{file_format}", f"gsm8k_test.{file_format}"]
                
                for train_file in train_files:
                    if os.path.exists(os.path.join(data_path, train_file)):
                        found_files.append(train_file)
                        break
                
                for test_file in test_files:
                    if os.path.exists(os.path.join(data_path, test_file)):
                        found_files.append(test_file)
                        break
                
                if len(found_files) >= 2:  # 至少找到train和test文件
                    return True
            
            print(f"错误: 在目录 {data_path} 中未找到有效的train和test文件")
            print(f"支持的文件格式: {config['supported_formats']}")
            return False
        
        return True


def create_sample_data_structure(base_dir: str = "gsm8k_sample_data"):
    """
    创建示例数据结构
    
    Args:
        base_dir: 基础目录路径
    """
    import json
    import os
    
    # 创建目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 示例训练数据
    train_data = [
        {
            "question": "Sarah has 12 apples. She gives away 3 apples to her friend. How many apples does Sarah have now?",
            "answer": "Sarah has 12 - 3 = 9 apples now.\n#### 9"
        },
        {
            "question": "A store sells 4 types of candy. Each type comes in 6 different flavors. How many different candy options are there?",
            "answer": "There are 4 * 6 = 24 different candy options.\n#### 24"
        },
        {
            "question": "Tom has 3 times as many books as Jerry. If Jerry has 2 books, how many books do they have together?",
            "answer": "Tom has 3 * 2 = 6 books. Together they have 6 + 2 = 8 books.\n#### 8"
        }
    ]
    
    # 示例测试数据
    test_data = [
        {
            "question": "A parking lot has 3 rows of cars. There are 5 cars in each row. How many cars are in the parking lot?",
            "answer": "There are 3 * 5 = 15 cars in the parking lot.\n#### 15"
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng worked 50/60 = 5/6 of an hour. She earned 12 * 5/6 = $10.\n#### 10"
        }
    ]
    
    # 保存为JSON格式
    with open(os.path.join(base_dir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(base_dir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 保存为JSONL格式
    with open(os.path.join(base_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(os.path.join(base_dir, "test.jsonl"), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"示例数据结构已创建在: {base_dir}")
    print("包含文件:")
    print("  - train.json / train.jsonl")
    print("  - test.json / test.jsonl")
    
    return base_dir


def get_data_path_examples():
    """获取数据路径示例"""
    examples = {
        "huggingface": {
            "description": "从Hugging Face Hub自动下载",
            "data_path": None,
            "command": "python eval_gsm8k.py --model_path /path/to/model --data_source huggingface"
        },
        
        "local_directory": {
            "description": "从本地目录加载（推荐）",
            "data_path": "/path/to/gsm8k_data",
            "command": "python eval_gsm8k.py --model_path /path/to/model --data_source directory --data_path /path/to/gsm8k_data"
        },
        
        "local_files": {
            "description": "从本地文件加载",
            "data_path": "/path/to/gsm8k_data",
            "command": "python eval_gsm8k.py --model_path /path/to/model --data_source local --data_path /path/to/gsm8k_data"
        }
    }
    
    print("数据路径配置示例:")
    print("=" * 60)
    
    for name, example in examples.items():
        print(f"{name.upper()}:")
        print(f"  描述: {example['description']}")
        print(f"  数据路径: {example['data_path']}")
        print(f"  命令: {example['command']}")
        print()
    
    return examples


if __name__ == "__main__":
    print("GSM8K数据配置工具")
    print("=" * 50)
    
    # 列出所有配置
    DataConfig.list_configs()
    
    print("\n")
    
    # 显示数据路径示例
    get_data_path_examples()
    
    print("\n")
    
    # 创建示例数据结构
    sample_dir = create_sample_data_structure()
    
    print(f"\n示例数据已创建，可以使用以下命令测试:")
    print(f"python eval_gsm8k.py --model_path /path/to/model --data_source directory --data_path {sample_dir}")
