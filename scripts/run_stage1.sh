python /home/lzh/llada-icl/generate_data_main.py \
  sampler.anchor_sample_num=200 \
  sampler.candidate_num=64 \
  few_shot_num=4 \
  beam_size=5 \
  mc_num=1 \
  batch_size=1 \
  use_multi_gpu=true \
  gpu_ids=[0,1] \
#   sampler.coarse_k=200 \
#   sampler.mmr_lambda=0.1 \






