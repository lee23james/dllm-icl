import json
import os
import random
from typing import Any

from loguru import logger
#最后返回的是
# {
#   "anchor_set": [2, 5],
#   "candidate_set": {
#       2: [7, 0, 9],
#       5: [1, 8, 2]  # 注意这里不允许包含 5，本代码会 while 重采样保证
#   }
# }
# 这样的结构,挑选出可能会用来进行测试的序号,然后附上候选池的例子序号

class BaseSampler:
    def __init__(
        self,
        candidate_num,
        sampler_name,
        cache_dir,
        anchor_sample_num,
        index_ds_len,
        overwrite,
        dataset_name,
        other_info='',
        anchor_idx_list=None
    ) -> None:
        self.candidate_num = candidate_num
        self.sampler_name = sampler_name
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.overwrite = overwrite
        self.anchor_sample_num = anchor_sample_num
        self.anchor_set_cache_fn = os.path.join(
            cache_dir, f'{dataset_name}-anchor_sample_num:{self.anchor_sample_num}.json'
        )
        #只要缓存方法改变,就会选择不同的数据集进行加载
        # cache 文件名中追加 other_info（比如 coarse_k / lambda 等），用于区分不同配置的缓存
        # 注意 Python 运算优先级：原来写成 `'-' if other_info else '' + other_info`
        # 实际在 other_info 非空时只会得到 '-'，不会拼上真正的内容，这里修正为正确的括号形式。
        suffix = f"-{other_info}" if other_info else ""
        cache_fn = (
            f"{dataset_name}-{self.sampler_name}-"
            f"anchor_sample_num: {self.anchor_sample_num}:{self.candidate_num}"
            f"{suffix}.json"
        )
        self.cache_file = os.path.join(self.cache_dir, cache_fn)
        self.index_ds_len = index_ds_len
        if anchor_idx_list is None:
            self.anchor_idx_list = self.sample_anchor_set()
        else:
            self.anchor_idx_list = anchor_idx_list

    def __call__(self, train_ds) -> Any:
        total_data = {}
        total_data['anchor_set'] = self.anchor_idx_list
        data = self.load_cache_file()
        if data is not None:
            total_data['candidate_set'] = data
            return total_data
        data = self.sample(self.anchor_idx_list, train_ds)
        self.save_cache_file(data)
        total_data['candidate_set'] = data
        return total_data

    #怎么采样的,需要某些技巧
    def sample(self, *args, **kwargs):
        raise NotImplemented

    #缓存了每一个anchor token的候选采样数据集(其实是在选定一个子候选池之后方便进行计算)
    def load_cache_file(self):
        if not os.path.exists(self.cache_file) or self.overwrite:
            logger.info(
                f'the candidate set cache {self.cache_file} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            return
        else:
            logger.info(
                f'the candidate set cache {self.cache_file} exists, reloding...'
            )
            #加载的时候会把数字转换成字符串
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            data = {int(k): v for k, v in data.items()}
            return data
    #cache_file是保存candidate data的
    def save_cache_file(self, data):
        with open(self.cache_file, 'w') as f:
            logger.info(f'save the candidate data to {self.cache_file}')
            json.dump(data, f)

    #观测anchor_set_cache_fn这个文件是否存在,然后依次进行加载
    def sample_anchor_set(self):
        logger.info(self.anchor_set_cache_fn)
        if os.path.exists(self.anchor_set_cache_fn) and not self.overwrite:
            logger.info('the anchor_set_cache_filename exists, loding...')
            anchor_idx_list = json.load(open(self.anchor_set_cache_fn, 'r'))
        else:
            logger.info(
                f'the anchor set cache {self.anchor_set_cache_fn} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            anchor_idx_list = random.sample(
                range(0, self.index_ds_len), self.anchor_sample_num
            )
            with open(self.anchor_set_cache_fn, 'w') as f:
                logger.info(f'save to {self.anchor_set_cache_fn}...')
                json.dump(anchor_idx_list, f)
        return anchor_idx_list
