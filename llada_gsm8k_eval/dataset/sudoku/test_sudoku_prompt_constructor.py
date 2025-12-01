#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudoku Prompt Constructor Test with Custom Template
使用自定义数独模板和CSV数据测试prompt构造器
"""
import sys
import os
import random
from pathlib import Path

# 将 llada_gsm8k_eval 目录加入 sys.path，便于直接按模块名导入
sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_constructor_gsm8k import GSM8KPromptConstructor
from dataset.sudoku.sudoku_handler import SudokuHandler

def format_sudoku_for_template(puzzle_str: str) -> str:
    """将数独字符串格式化为模板格式"""
    if len(puzzle_str) != 16:
        raise ValueError(f"Invalid puzzle length: {len(puzzle_str)}")
    
    # 将16位字符串转换为4x4网格
    grid = []
    for i in range(0, 16, 4):
        row = puzzle_str[i:i+4]
        grid.append(row)
    
    # 格式化为模板样式
    formatted_grid = '\n'.join(grid)
    return formatted_grid

def create_sudoku_samples_from_csv(n_samples:int =4)->list:
    """从csv数据中创建数独样本"""
    sudoku_handler = SudokuHandler()
    raw_data = sudoku_handler.load_raw_dataset()

    #随机挑选样本
    selected_indices = random.sample(range(len(raw_data)), n_samples)
    samples = []

    for idx in selected_indices:
        item = raw_data[idx]
        puzzle_formatted = format_sudoku_for_template(item['question'])
        samples.append({
            "question":f"Puzzle:\n{puzzle_formatted}",
            "answer":f"\n{item['answer']}\n</answer>"
        })
    
    return samples

def test_sudoku_custom_template():
    """使用自定义数独模版测试prompt构造器"""
    random.seed(1234)

    #创建5个训练样本
    train_samples = create_sudoku_samples_from_csv(n_samples=5)
    
    #创建1个测试样本
    test_samples = create_sudoku_samples_from_csv(n_samples=1)
    test_sample=test_samples[0]

    for i,sample in enumerate(train_samples):
        print(f"\n训练样本 {i+1}:")
        print(f"问题:\n{sample['question']}")
        print(f"答案:\n{sample['answer']}")

    print(f"\n=== 测试样本 ===")
    print(f"问题:\n{test_sample['question']}")
    print(f"答案:\n{test_sample['answer']}")    

    #测试不同位置
    for position in range(6):
        constructor = GSM8KPromptConstructor(n_shots=5, query_position=position)
        prompt = constructor.construct_prompt(train_samples, test_sample, mask_length=128)
        print(f"\n=== 位置 {position} 的prompt ===")
        print(prompt)

if __name__ == "__main__":
    test_sudoku_custom_template()
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)  


