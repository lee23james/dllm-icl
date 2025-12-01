from general_eval.prompt_constructor import BasePromptConstructor
import random
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

#指令构造
class GSM8KPromptConstructor(BasePromptConstructor):
    """基于basepromptconstructor构建的GSM8K任务的提示词构造器"""
    
    def __init__(self, n_shots: int = 4, random_seed: int = 1234, query_position: int = 1):
        """
        初始化构造器并添加缓存机制
        
        Args:
            n_shots: few-shot示例数量
            random_seed: 随机种子
            query_position: 查询问题的位置
                - 0: 放在所有示例之后（传统位置）
                - 1: 放在最后一个示例之前（倒数第二位）
                - 2: 放在倒数第二个示例之前（倒数第三位）
                - ...
                - n_shots: 放在所有示例之前（第一位）
        """
        super().__init__(n_shots, random_seed)
        self.query_position = query_position
        self._cached_examples = None  # 缓存采样的示例
        self._cached_example_prompts = None  # 缓存格式化的示例
        self._mask_token_cache = {}  # 缓存mask tokens
        
        # 验证query_position参数的有效性
        if query_position < 0 or query_position > n_shots:
            raise ValueError(f"query_position must be between 0 and {n_shots}, got {query_position}")
   
    def _get_mask_tokens(self, mask_length: int) -> str:
        """获取mask tokens，使用缓存优化"""
        if mask_length not in self._mask_token_cache:
            self._mask_token_cache[mask_length] = "<|mdm_mask|>" * mask_length
        return self._mask_token_cache[mask_length]
    
    def _format_example(self, sample: Dict, is_query: bool = False, mask_length: int = 256) -> str:
        """
        将单个样本格式化为 'Q: ... \nA: ...' 的字符串 - 优化版本
        
        Args:
            sample (Dict): 包含 'question' 和 'answer' 的字典。
            is_query (bool): 如果为True，则只包含问题和 "A:"，用于引导模型回答。
            mask_length (int): 如果is_query为True，在Answer:后插入的mask token数量。
        
        Returns:
            str: 格式化后的字符串。
        """
        # 避免重复的strip操作
        question = sample['question'].strip() if isinstance(sample['question'], str) else sample['question']
        
        if is_query:
            # 使用缓存的mask tokens
            mask_tokens = self._get_mask_tokens(mask_length)
            return f"Question: {question}\nAnswer: {mask_tokens}"
        else:
            # 避免重复的strip操作
            answer = sample['answer'].strip() if isinstance(sample['answer'], str) else sample['answer']
            return f"Question: {question}\nAnswer: {answer}"

    def _get_cached_example_prompts(self, train_samples: List[Dict]) -> List[str]:
        """获取缓存的示例prompts，避免重复格式化"""
        # 检查是否需要重新缓存（训练样本是否相同）
        if self._cached_examples is None or self._cached_examples != train_samples:
            # 缓存采样的示例
            self._cached_examples = self.sample_examples(train_samples, self.n_shots)
            # 缓存格式化的prompts
            self._cached_example_prompts = [self._format_example(ex) for ex in self._cached_examples]
        
        return self._cached_example_prompts
    
    def construct_prompt(self, train_samples: List[Dict], test_sample: Dict, mask_length: int = 256) -> str:
        """
        构建GSM8K的 few-shot prompt - 支持灵活位置控制
        
        根据query_position参数将查询问题插入到不同位置：
        - query_position=0: 查询放在所有示例之后（传统方式）
        - query_position=1: 查询放在最后一个示例之前
        - query_position=2: 查询放在倒数第二个示例之前
        - ...
        - query_position=n_shots: 查询放在所有示例之前
        """
        # 1. 获取缓存的示例prompts
        example_prompts = self._get_cached_example_prompts(train_samples)
        
        # 2. 格式化测试样本（Query）
        query_prompt = self._format_example(test_sample, is_query=True, mask_length=mask_length)

        # 3. 根据query_position决定插入位置
        if self.query_position == 0:
            # 传统位置：查询放在所有示例之后
            all_parts = example_prompts + [query_prompt]
        elif self.query_position == self.n_shots:
            # 查询放在所有示例之前
            all_parts = [query_prompt] + example_prompts
        else:
            # 查询插入到指定位置
            # query_position=1 表示从后往前数第1个位置（倒数第二位）
            insert_position = len(example_prompts) - self.query_position
            
            # 分割示例列表
            before_query = example_prompts[:insert_position]
            after_query = example_prompts[insert_position:]
            
            # 组合最终的prompt
            all_parts = before_query + [query_prompt] + after_query

        # 将所有部分用两个换行符连接
        prompt = "\n\n".join(all_parts)

        # 移除开头的换行符和空格
        prompt = prompt.lstrip('\n').lstrip()

        # 将所有部分用两个换行符连接
        return prompt



#===========================================================
#测试用例
def test_gsm8k_prompt_constructor():
    # 模拟训练样本
    train_samples = [
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
        },
        {
            "question": "A parking lot has 3 rows of cars. There are 5 cars in each row. How many cars are in the parking lot?",
            "answer": "There are 3 * 5 = 15 cars in the parking lot.\n#### 15"
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng worked 50/60 = 5/6 of an hour. She earned 12 * 5/6 = $10.\n#### 10"
        }
    ]
    
    # 模拟测试样本
    test_sample = {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"
    }
    
    # 测试不同query_position的效果
    print("=== 测试不同query_position的效果 ===")
    for position in range(5):  # 0到4的位置
        print(f"\n--- query_position = {position} ---")
        constructor = GSM8KPromptConstructor(n_shots=4, query_position=position)
        prompt = constructor.construct_prompt(train_samples, test_sample, mask_length=3)  # 使用短mask便于查看
        
        # 分析prompt结构
        parts = prompt.split("\n\n")
        print(f"总共 {len(parts)} 个部分:")
        
        for i, part in enumerate(parts):
            if "<|mdm_mask|>" in part:
                print(f"  第{i+1}部分: [查询问题] - {part.split('Question:')[1].split('Answer:')[0].strip()[:30]}...")
            else:
                # 提取问题的前30个字符
                question_start = part.split('Question:')[1].split('Answer:')[0].strip()[:30]
                print(f"  第{i+1}部分: [示例] - {question_start}...")
    
    # 详细展示一个位置的完整prompt
    print(f"\n=== 详细展示 query_position=1 的完整prompt ===")
    constructor = GSM8KPromptConstructor(n_shots=4, query_position=4)
    prompt = constructor.construct_prompt(train_samples, test_sample, mask_length=256)
    print(prompt)

if __name__ == "__main__":
    test_gsm8k_prompt_constructor()