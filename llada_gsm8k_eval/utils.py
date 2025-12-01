import re
import json
import yaml
import os

def extract_gsm8k_answer(text: str):
    """
    从GSM8K的生成文本中提取最终的数值答案。
    """
    match = re.search(r"####\s*([0-9,.-]+)", text)
    if match:
        try:
            # 清理数字中的逗号并转换为浮点数
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None

def extract_sudoku_answer(text: str):
    """
    从生成文本中提取4x4数独解（16位，仅包含'1'-'4'）。
    优先：第一个 </answer> 或 <\answer> 之前的“生成区域”；
    再优先：该区域内最后一个 "Answer:" 之后的子区域。
    在子区域中：
      1) 若有连续16位 [1-4]，取最靠近闭合标签的那一处（离末尾最近的一次匹配）
      2) 否则收集 [1-4]，取子区域中的最后16位
    若子区域失败，则在“生成区域”同样策略兜底；再失败，全文兜底。
    """
    if not isinstance(text, str):
        return None

    # 1) 生成区域：第一个闭合标签前
    close_tag = re.search(r"<\s*/?\s*answer\s*>", text, flags=re.IGNORECASE)
    gen_region = text[:close_tag.start()] if close_tag else text

    # 2) 子区域：生成区域内最后一个 "Answer:" 之后
    ans_pos = gen_region.lower().rfind("answer:")
    sub_region = gen_region[ans_pos + len("answer:"):] if ans_pos != -1 else gen_region

    # 3) 子区域内：先取“离末尾最近”的严格16连串
    last_strict = None
    for m in re.finditer(r"([1-4]{16})", sub_region):
        last_strict = m.group(1)
    if last_strict:
        return last_strict

    # 子区域内：否则取最后16个数字
    digits = re.findall(r"[1-4]", sub_region)
    if len(digits) >= 16:
        return "".join(digits[-16:])

    # 4) 生成区域兜底：严格16连串（最后一次）
    last_strict_gen = None
    for m in re.finditer(r"([1-4]{16})", gen_region):
        last_strict_gen = m.group(1)
    if last_strict_gen:
        return last_strict_gen

    # 生成区域兜底：最后16个数字
    digits_gen = re.findall(r"[1-4]", gen_region)
    if len(digits_gen) >= 16:
        return "".join(digits_gen[-16:])

    # 5) 全文最后兜底
    last_strict_all = None
    for m in re.finditer(r"([1-4]{16})", text):
        last_strict_all = m.group(1)
    if last_strict_all:
        return last_strict_all

    digits_all = re.findall(r"[1-4]", text)
    if len(digits_all) >= 16:
        return "".join(digits_all[-16:])

    return None



def save_results(filepath: str, results: dict):
    """将评测结果保存到JSON文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"结果已保存至: {filepath}")

def load_config(config_path: str) -> dict:
    """从YAML文件加载配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)