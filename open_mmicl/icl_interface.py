# LLaDA ICL 推理接口
# 参考原 open_mmicl ICLInferecer 结构，迁移到 DLLM（LLaDA）
# 使用 BaseInterface（LLaDAInterface）进行 prompt 构建与生成
# pyright: reportMissingImports=false
from typing import Optional, List, Dict, Any

import os
import json
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from open_mmicl.interface.base_interface import BaseInterface


class DLLMOutputHandler:
    """DLLM 推理结果收集器，替代 VLGenInferencerOutputHandler"""

    def __init__(self, num: int):
        self.num = num
        self.results_dict: Dict[str, List[Any]] = {
            "predictions": [],
            "outputs": [],
            "prompts": [],
            "ice_idx": [],
        }

    def save_prediction_and_output(
        self,
        prediction: str,
        output_list: List[str],
        origin_prompt: str,
        index: int,
    ) -> None:
        self.results_dict["predictions"].append(prediction)
        self.results_dict["outputs"].append(output_list)
        self.results_dict["prompts"].append(origin_prompt)

    def save_origin_info(self, field: str, test_ds: Dataset) -> None:
        if field not in self.results_dict:
            self.results_dict[field] = []
        if field in test_ds.column_names:
            self.results_dict[field] = list(test_ds[field])

    def creat_index(self, test_ds: Dataset) -> None:
        pass  # 可选：建立索引

    def write_to_json(
        self,
        output_json_filepath: str,
        output_json_filename: str,
    ) -> None:
        os.makedirs(output_json_filepath, exist_ok=True)
        output_path = os.path.join(output_json_filepath, f"{output_json_filename}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")


class DLLMICLInferencer:
    """
    DLLM 的 ICL 推理接口类（迁移自原 ICLInferecer，适配 LLaDA）

    使用 BaseInterface（LLaDAInterface）进行：
    - prompt 构建（build_prompt / concat_prompt）
    - 生成（interface.generate，内部调用 LLaDA 的 diffusion 生成）
    """

    def __init__(
        self,
        interface: BaseInterface,
        train_ds: Dataset,
        test_ds: Dataset,
        generation_kwargs: Dict[str, Any],
        query_position: int = 0,
        other_save_field: Optional[List[str]] = None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        output_json_filepath: Optional[str] = "./llada_icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
    ) -> None:
        """
        Args:
            interface: BaseInterface 子类（如 LLaDAInterface），提供 build_prompt、generate
            train_ds: 训练集（用于 ICD）
            test_ds: 测试集
            generation_kwargs: 生成参数（steps, gen_length, block_length, temperature 等）
            query_position: query 在 ICD 序列中的位置（0=最后，nshot=最前）
            other_save_field: 需额外保存的字段
            batch_size: DataLoader batch size（LLaDA 逐条生成，batch_size 主要影响数据加载）
            num_workers: DataLoader workers
            output_json_filepath: 输出目录
            output_json_filename: 输出文件名
        """
        self.interface = interface
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.generation_kwargs = generation_kwargs
        self.query_position = query_position
        self.other_save_field = other_save_field or []
        self.batch_size = batch_size or 1
        self.num_workers = num_workers or 0
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename

    def _build_prompt_from_data_sample_list(
        self,
        data_sample_list: List[Dict[str, Any]],
    ) -> str:
        """
        从 [ice1, ice2, ..., query] 构建完整 prompt
        与 reference 的 transfer_prompts + concat 等价
        """
        if not data_sample_list:
            raise ValueError("data_sample_list cannot be empty")
        ice_samples = data_sample_list[:-1]
        query_sample = dict(data_sample_list[-1])
        query_sample["isquery"] = 1
        return self.interface.build_prompt(
            ice_samples=ice_samples,
            test_sample=query_sample,
            query_position=self.query_position,
        )

    @torch.inference_mode()
    def inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行 ICL 推理（与 reference ICLInferecer.inference 接口一致）

        Args:
            train_ds: 训练集
            test_ds: 测试集
            ice_idx_list: 每个测试样本的 ICD 索引列表，ice_idx_list[i] = [idx1, idx2, ...]
            output_json_filepath: 输出目录
            output_json_filename: 输出文件名

        Returns:
            results_dict: 包含 predictions, outputs, prompts, ice_idx 等
        """
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        num = len(test_ds)
        output_handler = DLLMOutputHandler(num)
        output_handler.results_dict["ice_idx"] = ice_idx_list

        for field in self.other_save_field:
            output_handler.save_origin_info(field, test_ds)

        test_ds_with_ice = test_ds.add_column("ice_idx", ice_idx_list)

        logger.info("Starting DLLM ICL inference...")
        index = 0

        for i in tqdm(range(num), ncols=100):
            e = test_ds_with_ice[i]
            ice_indices = e["ice_idx"]
            ice_sample_list = [train_ds[idx] for idx in ice_indices]
            data_sample_list = ice_sample_list + [e]
            prompt = self._build_prompt_from_data_sample_list(data_sample_list)

            prompt_tensor = self.interface.tokenize_prompt(prompt)
            prompt_tensor = prompt_tensor.to(self.interface.device)

            output_tensor = self.interface.generate(
                prompt_tensor,
                **self.generation_kwargs,
            )

            prompt_len = prompt_tensor.shape[1]
            output_ids = output_tensor[0].tolist()
            complete_output = self.interface.tokenizer.decode(
                output_ids, skip_special_tokens=False
            )
            output_without_sp = self.interface.tokenizer.decode(
                output_ids, skip_special_tokens=True
            )
            generated = self.interface.tokenizer.decode(
                output_ids[prompt_len:], skip_special_tokens=True
            )
            origin_prompt = self.interface.tokenizer.decode(
                output_ids[:prompt_len], skip_special_tokens=True
            )

            output_handler.save_prediction_and_output(
                generated,
                [complete_output, output_without_sp],
                origin_prompt,
                index,
            )
            index += 1

        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return output_handler.results_dict

    @torch.inference_mode()
    def gen_inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成推理（与 inference 相同，保持接口一致）"""
        return self.inference(
            train_ds, test_ds, ice_idx_list,
            output_json_filepath, output_json_filename,
        )

    @torch.inference_mode()
    def ppl_inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """困惑度推理（LLaDA 为生成模型，暂回退到生成推理）"""
        logger.warning("PPL inference for LLaDA is not fully implemented, falling back to generation.")
        return self.inference(
            train_ds, test_ds, ice_idx_list,
            output_json_filepath, output_json_filename,
        )


class LLaDAICLInferencer(DLLMICLInferencer):
    """兼容名：LLaDAICLInferencer 为 DLLMICLInferencer 的别名"""
    pass
