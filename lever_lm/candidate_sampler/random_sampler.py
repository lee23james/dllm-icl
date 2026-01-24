import random

from .base_sampler import BaseSampler
#这里要搭配一些数量比较多的数据集进行采样推理
#返回{anchor_idx: [candidate_idx1, candidate_idx2, ...]}的结构
class RandSampler(BaseSampler):
    def __init__(
        self,
        candidate_num,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        dataset_name,
        cache_dir,
        overwrite,
        anchor_idx_list=None,
    ):
        super().__init__(
            candidate_num=candidate_num,
            dataset_name=dataset_name,
            sampler_name=sampler_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_idx_list=anchor_idx_list,
        )

    def sample(self, anchor_set, train_ds):
        candidate_set_idx = {}
        for s_idx in anchor_set:
            random_candidate_set = random.sample(
                range(0, len(train_ds)), self.candidate_num
            )
            #便利数据集,防止让idx出现在后续集合中,选择重新再抽一次
            while s_idx in random_candidate_set:
                random_candidate_set = random.sample(
                    list(range(0, len(train_ds))), self.candidate_num
                )
            #找到了就选择适配s_idx的候选集
            candidate_set_idx[s_idx] = random_candidate_set
        return candidate_set_idx
# 返回: {
#     "anchor_set": [100, 250, 500, ...],
#     "candidate_set": {100: [10, 20, ...], 250: [15, 25, ...], ...}
# }
