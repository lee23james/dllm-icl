
"""
Sudoku Dataset Handler
处理4x4数独数据集，兼容GSM8K评估框架
"""

import os
import csv
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
# Note: no external dataset libraries are required for this handler

class SudokuHandler:
    """4x4数独数据集处理器 - 兼容GSM8K评估框架"""

    def __init__(self, csv_path: str = str(Path(__file__).parent / "4x4_sudoku_unique_solution.csv")):
        """
        初始化数独数据集处理器

        Args:
            csv_path: 数独数据集CSV文件路径
        """
        self.csv_path = Path(csv_path)
        self.test_size =256 #前256个作为测试集
        self.few_shot_size=5 #5-shot示例

        # 数据缓存
        self._raw_data = None
        self._test_dataset = None
        self._train_dataset = None

        #验证文件是否存在、
        if not self.csv_path.exists():
            raise FileNotFoundError(f"数独数据集CSV文件不存在: {self.csv_path}")

    def load_raw_dataset(self) -> List[Dict[str,str]]:
        "从csv文件加载原始数据集"
        if self._raw_data is not None:
            return self._raw_data

        print(f"loading raw dataset from {self.csv_path}")
        raw_data=[]

        with open(self.csv_path,'r',encoding='utf-8') as f:
            reader=csv.DictReader(f, delimiter='\t')
            for row in reader:
                raw_data.append({
                    "question":row['Puzzle'].strip(),
                    "answer":row['Solution'].strip()
                })
        
        self._raw_data=raw_data
        print(f"loaded {len(raw_data)} raw dataset")
        return raw_data

    def format_sudoku_puzzle(self,puzzle_str:str)->str:
        "格式化数独谜题"
        if len(puzzle_str)!=16:
            raise ValueError(f"Invalid puzzle length: {len(puzzle_str)}")

        #替换成4*4网络，并将0替换为.
        grid =[]
        for i in range(0,16,4):
            row=puzzle_str[i:i+4]
            formatted_row = ' '.join(cell if cell != '0' else '.' for cell in row)
            grid.append(formatted_row)

        formatted_grid='\n'.join(grid)
        return f'Here is a 4x4 sudoku puzzle:\n{formatted_grid}'

    def format_sudoku_solution(self, solution_str: str) -> str:
        """格式化数独解答"""
        if len(solution_str) != 16:
            raise ValueError(f"Invalid solution length: {len(solution_str)}")

        # 将解答格式化为4x4网格
        grid = []
        for i in range(0, 16, 4):
            row = solution_str[i:i+4]
            formatted_row = ' '.join(row)
            grid.append(formatted_row)

        formatted_grid = '\n'.join(grid)
        return f'Here is the solution:\n{formatted_grid}'

    def convert_to_gsm8k_format(self,puzzle_str:str,solution_str:str)->Dict[str,str]:
        """将数独数据集转换为gsm8k格式"""
        question=self.format_sudoku_puzzle(puzzle_str)
        answer=self.format_sudoku_solution(solution_str)
        return{
            "question":question,
            "answer":answer
        }


    def convert_all_data(self):
        """转换所有数据集并输出格式"""
        print("=" * 80)
        print("数独数据集格式转换")
        print("=" * 80)
        
        # 加载原始数据
        raw_data = self.load_raw_dataset()
        
        # 转换所有数据
        converted_data = []
        for i, item in enumerate(raw_data):
            converted_item = self.convert_to_gsm8k_format(item['question'], item['answer'])
            converted_data.append(converted_item)
        
        print(f"\n转换完成！总共转换了 {len(converted_data)} 个样本")
        
        # 显示前几个样本的转换结果
        print(f"\n前10个样本的转换结果:")
        for i in range(min(10, len(converted_data))):
            print(f"\n--- 样本 {i+1} ---")
            print(f"问题:\n{converted_data[i]['question']}")
            print(f"答案:\n{converted_data[i]['answer']}")
        
        # 显示数据分割信息
        print(f"\n数据分割:")
        print(f"测试集: 前 {self.test_size} 个样本")
        print(f"训练集: 剩余 {len(converted_data) - self.test_size} 个样本")
        
        return converted_data

def main():
    """主函数 - 转换数据集格式"""
    handler = SudokuHandler()
    converted_data = handler.convert_all_data()
    
    print(f"\n转换完成！数据集已准备好用于评估。")
    print(f"总共 {len(converted_data)} 个样本")


if __name__ == "__main__":
    main()

        
