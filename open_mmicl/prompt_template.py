"""
Prompt模板类：用于根据模板字符串生成prompt
"""
from typing import Dict, Any, Optional
import re
from loguru import logger


class PromptTemplate:
    """
    Prompt模板类：根据模板字符串生成prompt
    
    模板格式：
    - <Q>: 会被question替换
    - <A>: 会被answer替换（对于query，会被mask替换）
    - 支持通过column_token_map自定义字段到token的映射
    """
    
    def __init__(
        self,
        prompt_template: str,
        mask_token_str: str = "<|mdm_mask|>",
        mask_length: int = 256,
        column_token_map: Optional[Dict[str, str]] = None,
        mask_column_token_map: Optional[Dict[str, str]] = None,
    ):
        """
        初始化PromptTemplate
        
        Args:
            prompt_template: 模板字符串，如 "question: <Q>\n<answer>\n<A>\n</answer>"
            mask_token_str: mask token字符串
            mask_length: mask token的长度
            column_token_map: 字段到token的映射，如 {"question": "<Q>", "answer": "<A>", "choice_a": "<a>"}
            mask_column_token_map: 需要mask的字段到token的映射，如 {"answer": "<A>"}
                                  在query生成时，这些字段会被mask替换
        """
        self.prompt_template = prompt_template
        self.mask_token_str = mask_token_str
        self.mask_length = mask_length
        self.mask_tokens = mask_token_str * mask_length
        self.column_token_map = column_token_map or {}
        # 处理mask_column_token_map，支持字典和字符串两种格式
        if mask_column_token_map is None:
            self.mask_column_token_map = {}
        elif isinstance(mask_column_token_map, str):
            # 如果是字符串，表示要mask的字段名（如"answer"）
            # 必须在 column_token_map 中能找到对应 token，否则直接报错
            field_name = mask_column_token_map
            if field_name in (column_token_map or {}):
                # 找到对应的token，构建 {字段名: token} 的映射
                self.mask_column_token_map = {field_name: column_token_map[field_name]}
                logger.debug(f"mask_column_token_map: '{field_name}' -> token '{column_token_map[field_name]}'")
            else:
                raise ValueError(
                    f"mask_column_token_map field '{field_name}' not found in column_token_map "
                    f"(available columns: {list((column_token_map or {}).keys())})"
                )
        else:
            # 字典格式，直接使用
            self.mask_column_token_map = mask_column_token_map
    
    def _replace_tokens(self, template: str, sample: Dict[str, Any], use_mask: bool = False) -> str:
        """
        根据column_token_map替换模板中的token
        
        Args:
            template: 模板字符串
            sample: 数据样本字典（包含从数据集中加载的字段，如question, answer等）
            use_mask: 是否对mask_column_token_map中的字段使用mask（用于query）
        
        Returns:
            替换后的模板字符串
        """
        result = template
        
        # 遍历 column_token_map，进行**一一对应**的替换：
        # - 每个 column_name 必须在 sample 中出现，否则直接报错
        # - 如果 use_mask=True 且该列在 mask_column_token_map 中，则用 mask_tokens 替换
        # - 否则使用 sample[column_name] 的字符串值替换
        for column_name, token in self.column_token_map.items():
            # 检查字段是否存在于样本中
            if column_name not in sample:
                # 如果模板中用了某个占位符，但样本里没有对应字段，认为配置/数据有问题，直接报错
                raise ValueError(
                    f"Column '{column_name}' not found in sample keys {list(sample.keys())} "
                    f"when replacing token '{token}'"
                )
            
            # 是否需要对该字段做 mask（例如 answer）
            should_mask = use_mask and column_name in self.mask_column_token_map
            
            if should_mask:
                value = self.mask_tokens
            else:
                value = str(sample[column_name])
            
            result = result.replace(token, value)
        
        return result
    
    def generate_ice_item(self, sample: Dict[str, Any]) -> str:
        """
        生成单个ICD示例的prompt
        
        Args:
            sample: 样本字典，包含question和answer字段
        
        Returns:
            格式化后的ICD文本
        """
        # 提取question和answer
        if "q_a" in sample:
            # 如果已经有q_a字段，直接使用
            return sample["q_a"]
        elif "question" in sample and "answer" in sample:
            # 如果提供了column_token_map，使用通用替换方法
            if self.column_token_map:
                # 通用路径：根据 column_token_map 直接替换占位符
                prompt = self._replace_tokens(self.prompt_template, sample, use_mask=False)
                return prompt
            else:
                # 向后兼容：没有column_token_map时使用旧逻辑
                # 检查是否是 MMLU 格式（包含 choices）
                if "choices" in sample:
                    # MMLU 格式：使用模板格式化
                    question = sample["question"].strip()
                    choices = sample["choices"]
                    answer = sample["answer"]
                    
                    # 构建选项文本
                    options_text = ""
                    if len(choices) >= 4:
                        options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                    elif len(choices) > 0:
                        option_labels = ["A", "B", "C", "D", "E", "F"][:len(choices)]
                        options_text = "\n".join([f"{label}. {choice}" for label, choice in zip(option_labels, choices)])
                    
                    # 使用模板格式化（支持 {{question}} 和 {{choices[i]}} 格式）
                    prompt = self.prompt_template
                    prompt = prompt.replace("{{question.strip()}}", question)
                    prompt = prompt.replace("{{question}}", question)
                    if len(choices) >= 4:
                        prompt = prompt.replace("{{choices[0]}}", choices[0])
                        prompt = prompt.replace("{{choices[1]}}", choices[1])
                        prompt = prompt.replace("{{choices[2]}}", choices[2])
                        prompt = prompt.replace("{{choices[3]}}", choices[3])
                    # 替换 <Q> 为完整的问题+选项
                    full_question = f"{question}\n{options_text}" if options_text else question
                    prompt = prompt.replace("<Q>", full_question)
                    # 替换 <A> 为答案
                    prompt = prompt.replace("<A>", answer)
                    return prompt
                else:
                    # 普通格式：使用模板格式化
                    question = sample["question"]
                    answer = sample["answer"]
                    # 替换模板中的<Q>和<A>
                    prompt = self.prompt_template.replace("<Q>", question).replace("<A>", answer)
                    return prompt
        else:
            raise ValueError(f"Unknown ICD format: {sample.keys()}")
    
    def generate_query_item(self, sample: Dict[str, Any], use_mask: bool = True) -> str:
        """
        生成query的prompt
        
        Args:
            sample: 样本字典，包含question字段（如果use_mask=False，还需要answer字段）
            use_mask: 是否使用mask替换answer部分
                - True: 推理阶段，answer部分用mask替换
                - False: 打分阶段，使用完整的answer（用于蒙特卡洛估计）
        
        Returns:
            格式化后的query文本
        """
        if use_mask:
            # 推理阶段：answer部分用mask替换
            if "q_a" in sample:
                # 如果只有q_a，提取question部分
                qa_text = sample["q_a"]
                if "<answer>" in qa_text:
                    # 提取question部分，然后加上<answer>\n和mask
                    question_part = qa_text.split("<answer>")[0].strip()
                    # 构建query prompt
                    prompt = self.prompt_template.replace("<Q>", question_part.replace("question:", "").strip())
                    prompt = prompt.replace("<A>", self.mask_tokens)
                    return prompt
                else:
                    return qa_text
            elif "question" in sample:
                # 如果提供了column_token_map，使用通用替换方法
                if self.column_token_map:
                    # 通用路径：根据 column_token_map 直接替换占位符（需要 mask 的列会被自动 mask）
                    prompt = self._replace_tokens(self.prompt_template, sample, use_mask=True)
                    return prompt
                else:
                    # 不再支持任何额外格式，统一报错
                    raise ValueError(
                        "column_token_map is required for masked query generation when using question format "
                        f"(got sample keys: {list(sample.keys())})"
                    )
            else:
                raise ValueError(f"Unknown query format: {sample.keys()}")
        else:
            # 打分阶段：使用完整的answer（用于蒙特卡洛估计）
            if "q_a" in sample:
                # 如果已经有q_a字段，直接使用（包含完整answer）
                return sample["q_a"]
            elif "question" in sample and "answer" in sample:
                # 只能走通用模板路径：必须提供 column_token_map
                if not self.column_token_map:
                    raise ValueError(
                        "column_token_map is required for scoring mode when using question/answer format "
                        f"(got sample keys: {list(sample.keys())})"
                    )
                # 通用路径：根据 column_token_map 直接替换占位符（保留完整 answer）
                prompt = self._replace_tokens(self.prompt_template, sample, use_mask=False)
                return prompt
            else:
                # 其他格式一律视为无效
                raise ValueError(f"Unknown query format for scoring: {sample.keys()}")
    
    def generate_text_for_embedding(self, sample: Dict[str, Any], output_column: Optional[str] = None) -> str:
        """
        生成用于 embedding 的文本（通常不含答案部分）
        主要用于相似度检索，**尽量复用当前任务的 prompt 模板逻辑**。
        
        对于使用了 column_token_map / prompt_template 的任务（如 MMLU）：
        - 会先复制一份 sample
        - 将 output_column（例如 "answer"）对应字段置为 ""（空字符串）
        - 然后调用与 ICD 相同的模板替换逻辑生成一个“无答案版” prompt
        
        Args:
            sample: 样本字典
        
        Args:
            sample: 样本字典
            output_column: 输出列名称（如 "answer"），用于在生成 embedding 文本时去掉该字段
        
        Returns:
            格式化后的文本（通常不含答案）
        """
        # 情况1：优先使用当前任务已经配置好的模板 + column_token_map
        #       通过将 output_column（如 "answer"）置空，生成“无答案版” prompt
        if self.column_token_map:
            sample_no_output = dict(sample)
            if output_column is not None and output_column in sample_no_output:
                # 将输出列置为空字符串，相当于模板中 <A> → ""，
                # 例如 "Answer: <A>" 会变成 "Answer: "
                sample_no_output[output_column] = ""
            
            # 直接复用 generate_ice_item 的模板替换逻辑
            # - 对 MMLU：会自动处理 question + choice_a~d
            # - 对一般任务：使用 column_token_map 中的占位符
            return self.generate_ice_item(sample_no_output)
        
        else:
            raise ValueError(f"Unknown sample format for embedding: {sample.keys()}")