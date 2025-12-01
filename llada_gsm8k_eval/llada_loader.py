#载入LLaDA模型  
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Tuple, Optional
import accelerate
# from model_config.configuration_llada import LLaDAConfig
# from model_config.modeling_llada import LLaDAModelLM


# MODEL_PATH = "/home/share/model_weight/llada/LLaDA-8B-Base/"

# 1) 本地读取 config
# config = LLaDAConfig.from_pretrained(MODEL_PATH, local_files_only=True)

from model_config.modeling_llada import LLaDAModelLM

# def load_model(model_path: str, device: str = 'cuda')->tuple[AutoModelForCausalLM, AutoTokenizer]:
#     """
#     加载LLaDA模型和分词器。

#     Args:
#         model_path (str): 模型权重和配置文件的路径。
#         device (str): 模型加载的目标设备 ('cuda' or 'cpu')。

#     Returns:
#         tuple: (model, tokenizer)
#     """ 
#     print(f"Loading model from {model_path} on {device}...")
class LLadaLoader:
    "LLaDA模型加载器"
    def __init__(
        self,
        mask_id:int =126336,
        max_length: int=4096,
        torch_dtype: torch.dtype=torch.bfloat16,
        trust_remote_code: bool=True,
        local_files_only: bool=True,
    ):
        """
        初始化LLaDA模型加载器。

        Args:
            mask_id: 掩码ID。
            max_length: 最大长度。
            torch_dtype: 模型精度。
            trust_remote_code: 是否信任远程代码。
        """
        self.mask_id = mask_id
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None
        self.device = None
        self.accelerator=None
        self._rank = 0
        self._world_size = 1


    def load_model(
        self,
        model_path:str,
        device: str="cuda",
        use_accelerate: bool=False,
    )->Tuple[AutoModel, AutoTokenizer]:
        """"
        加载LLaDA模型和分词器。
        Args:
            model_path: 模型权重和配置文件的路径。
            device: 模型加载的目标设备 ('cuda' or 'cpu')。
            use_accelerate: 是否使用accelerate库。
        Returns:
            tuple: (model, tokenizer)
        """
         # 设置设备
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")

        #初始化Accelkerator
        if use_accelerate:
            self.accelerator = accelerate.Accelerator()
            if self.accelerator.num_processes > 1:
                self._rank =self.accelerator.local_process_index
                self._world_size =self.accelerator.num_processes
                self.device=torch.device(self.accelerator.device)
        
        #准备模型参数
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "local_files_only": self.local_files_only,
        }
        
        #如果使用Accelerator,添加device_map
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        print(f"Loading model from {model_path} on {self.device}...")
        
        #加载模型
        # self.model =AutoModel.from_pretrained(
        #     model_path,
        #     **model_kwargs,
        # )


        # 加载模型
        self.model =LLaDAModelLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        #加载分词器
        self.tokenizer =AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        
        #处理设备分配
        if self.accelerator is not None and self.accelerator.num_processes >1 :
            self.model =self.accelerator.prepare(self.model)
        else:
            self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")

        return self.model,self.tokenizer


    def get_model_config(self)->dict:
        """
        获取模型配置
        
        returns:
        dict:模型配置的字典
        """
        return {
            "mask_id":self.mask_id,
            "max_length":self.max_length,
            "torch_dtype":self.torch_dtype,
            "trust_remote_code":self.trust_remote_code,
            "local_files_only":self.local_files_only,
            "device":self.device,
            "accelerator":self.accelerator,
            "rank":self._rank,
            "world_size":self._world_size,
            "use_accelerate":self.accelerator is not None,
        }

    def get_mask_id(self)->int:
        """
        获取掩码ID
        returns:
        int:掩码ID
        """
        return self.mask_id
    
    def get_device(self)->torch.device:
        """
        获取设备
        returns:
        torch.device:设备
        """
        return self.device

    def is_distributed(self)->bool:
        """
        判断是否分布式
        returns:
        bool:是否分布式
        """
        return self.accelerator is not None and self.accelerator.num_processes > 1
    
    def wait_for_everyone(self):
        """
        等待所有进程同步（仅在分布式模式下使用）
        """
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            

def load_model(
    model_path:str,
    device:str="cuda",
    use_accelerate:bool=False,
    mask_id:int=126336,
    max_length:int=4096,
    torch_dtype:torch.dtype=torch.bfloat16
)->Tuple[AutoModel, AutoTokenizer]:
    """
    便捷加载LLaDA模型和分词器
    
    Args:
        model_path:模型权重和配置文件的路径。
        device:模型加载的目标设备 ('cuda' or 'cpu')。
        use_accelerate:是否使用accelerate库。
        mask_id:掩码ID。
        max_length:最大长度。
        torch_dtype:模型精度。
        returns:
        tuple: (model, tokenizer)
    """
    loader = LLadaLoader(
        mask_id=mask_id, 
        max_length=max_length, 
        torch_dtype=torch_dtype
    )
    return loader.load_model(model_path, device, use_accelerate)


#测试用例
if __name__ == "__main__":
    model_path = "/data/share/model_weight/llada/LLaDA-8B-Base/"
    device = "cuda:1"
    use_accelerate = False
    mask_id = 126336
    max_length = 4096
    torch_dtype = torch.bfloat16
    
    # 使用加载器类一次性完成所有操作
    loader = LLadaLoader(
        mask_id=mask_id, 
        max_length=max_length, 
        torch_dtype=torch_dtype
    )
    
    # 加载模型和分词器
    model, tokenizer = loader.load_model(model_path, device, use_accelerate)
    
    # 输出模型信息
    print("=" * 50)
    print("模型架构信息:")
    print(model)
    print("\n" + "=" * 50)
    print("分词器信息:")
    print(tokenizer)
    print("\n" + "=" * 50)
    
    # 获取并输出配置信息
    config = loader.get_model_config()
    print("模型配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("模型加载完成！")