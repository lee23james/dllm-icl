"""
GSM8K评估指标
"""
import re
from typing import Dict, List, Any
from loguru import logger

#设置成类方便进行引用,同时还是想办法
class GSM8KMetrics:
    """
    GSM8K任务的评估指标
    """
    #这个应该是提取的指标
    @staticmethod
    def extract_answer(text: str) -> str:
        """
        从生成的文本中提取答案
        参考 utils/eval_utils.py 中的 gsm8k_check 逻辑
        
        Args:
            text: 生成的文本
        
        Returns:
            提取的答案（数字字符串）
        """
        # 方法1: 严格匹配 "The answer is X" 模式
        strict_match = re.search(r"The answer is (\-?[0-9\.\,]+)", text)
        if strict_match:
            return strict_match.group(1)
        
        # 方法2: 灵活匹配 "#### X" 模式
        flexible_match = re.search(r"#### (\-?[0-9\.\,]+)", text)
        if flexible_match:
            return flexible_match.group(1)
        
        # 方法3: 从<answer>标签中提取（备用方法）
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # 提取数字
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            if numbers:
                return numbers[-1]  # 返回最后一个数字
        
        # 如果都没找到，返回空字符串
        return ""
    #把提取的部分去掉一些乱七八糟的东西方便进行比对
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        标准化答案（参考 utils/eval_utils.py 中的 normalize_number）
        
        Args:
            answer: 原始答案
        
        Returns:
            标准化后的答案
        """
        # 参考 normalize_number 的逻辑
        answer = answer.strip()
        answer = re.sub(r'[\$,]', '', answer)  # 去掉 $ 和 ,
        answer = answer.rstrip('.')  # 去掉末尾的点
        
        return answer
    
    @staticmethod
    def is_correct(predicted: str, ground_truth: str) -> bool:
        """
        判断预测答案是否正确
        参考 utils/eval_utils.py 中的 gsm8k_check 逻辑
        
        Args:
            predicted: 预测答案（可以是原始文本或提取的答案）
            ground_truth: 真实答案（可能包含 "#### X" 格式）
        
        Returns:
            是否正确
        """
        # 从 ground_truth 中提取数字（ground_truth 一定可以找到）
        strict_truth = re.search(r"#### (\-?[0-9\.\,]+)", ground_truth)
        if strict_truth:
            truth = strict_truth.group(1)
        else:
            # 如果没有 #### 格式，尝试直接提取数字
            numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
            if numbers:
                truth = numbers[-1]
            else:
                truth = ground_truth.strip()
        
        # 从 predicted 中提取答案
        # 如果 predicted 已经是提取的答案（纯数字），直接使用
        # 否则尝试从文本中提取
        if re.match(r'^\-?[0-9\.\,]+$', predicted.strip()):
            answer = predicted.strip()
        else:
            answer = GSM8KMetrics.extract_answer(predicted)
        
        # 如果提取失败，检查 truth 是否在 predicted 中（灵活匹配）
        if not answer:
            return truth in predicted
        
        # 使用 normalize_answer 比较,提取出来的答案在双方都进行处理之后正则式匹配,看看效果咋样
        if GSM8KMetrics.normalize_answer(answer) == GSM8KMetrics.normalize_answer(truth):
            return True
        else:
            return False
    
    @staticmethod
    def evaluate_single(
        generated_text: str,#这个是生成的文本,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        评估单个样本
        参考 utils/eval_utils.py 中的 gsm8k_check 逻辑
        
        Args:
            generated_text: 生成的文本
            ground_truth: 真实答案（可能包含 "#### X" 格式）
        
        Returns:
            评估结果字典
        """
        # 提取预测答案
        predicted = GSM8KMetrics.extract_answer(generated_text)
        
        # 判断是否正确（is_correct 内部会处理 ground_truth 的提取）
        is_correct = GSM8KMetrics.is_correct(generated_text, ground_truth)
        
        # 从 ground_truth 中提取标准答案（用于显示）
        strict_truth = re.search(r"#### (\-?[0-9\.\,]+)", ground_truth)
        if strict_truth:
            extracted_truth = strict_truth.group(1)
        else:
            extracted_truth = ground_truth.strip()
        
        return {
            "predicted": predicted,
            "ground_truth": extracted_truth,
            "is_correct": is_correct,
            "generated_text": generated_text,
        }
    #但是要确保的是问题和答案必须要一一对应,感觉还是比较困难
    @staticmethod
    def evaluate_batch(
        generated_texts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """
        批量评估
        
        Args:
            generated_texts: 生成的文本列表
            ground_truths: 真实答案列表
        
        Returns:
            评估结果字典
        """
        assert len(generated_texts) == len(ground_truths), \
            f"Length mismatch: {len(generated_texts)} vs {len(ground_truths)}"
        
        results = []
        correct_count = 0
        
        for gen_text, gt in zip(generated_texts, ground_truths):
            result = GSM8KMetrics.evaluate_single(gen_text, gt)
            results.append(result)
            if result["is_correct"]:
                correct_count += 1
        
        accuracy = correct_count / len(results) if len(results) > 0 else 0.0
        
        return {
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results),
        }

