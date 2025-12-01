# #主要是想看看countdown的特性,因为我看到她出现了最后一种位置是最好的我还是比较怀疑,我想多做一点看看是什么原因
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 3 
# python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 300 --nshot 4 --gen_length 32 --steps 32  --data_path ./data/countdown.jsonl --version answer_token
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 5 --gen_length 32
# python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 300 --nshot 6 --gen_length 32 --steps 32 --data_path ./data/countdown.jsonl --version answer_token
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 7 --gen_length 32
# python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 300 --nshot 8 --gen_length 32 --steps 32 --data_path ./data/countdown.jsonl --version answer_token


# python improve/generate_improve.py --task gsm8k --device cuda:2 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/gsm8k_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 4 --gen_length 256 --steps 128 --block_length 256 --data_path ./data/gsm8k.jsonl --version answer_token
# python improve/generate_improve.py --task gsm8k --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/gsm8k_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 6 --gen_length 256 --steps 128 --block_length 256 --data_path ./data/gsm8k.jsonl --version answer_token
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 7 --gen_length 32
# python improve/generate_improve.py --task gsm8k --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/gsm8k_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 8 --gen_length 256 --steps 256 --block_length 256 --data_path ./data/gsm8k.jsonl --version answer_token

# python improve/generate_improve.py --task math500 --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/math500_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 4 --gen_length 512 --steps 256 --block_length 512 --data_path ./data/math500.jsonl --version answer_token
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 5 --gen_length 32
# python improve/generate_improve.py --task math500 --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/math500_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 6 --gen_length 512 --steps 256 --block_length 512 --data_path ./data/math500.jsonl --version answer_token
# # python improve/generate_improve.py --task countdown --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/countdown_results --lamda1 0.8 --lamda2 0.2 --samples_num 288 --nshot 7 --gen_length 32
# python improve/generate_improve.py --task math500 --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/math500_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 8 --gen_length 512 --steps 256 --block_length 512 --data_path ./data/math500.jsonl --version answer_token

# python improve/generate_improve.py --task  --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/math500_results --lamda1 0.8 --lamda2 0.2 --samples_num 200 --nshot 8 --gen_length 512 --steps 256 --block_length 512 --data_path ./data/math500.jsonl --version answer_token

python improve/generate_improve.py --task mbpp --device cuda:0 --model_name /home/share/model_weight/llada/LLaDA-8B-Base --result_path ./results/mbpp_results --lamda1 0.8 --lamda2 0.2 --samples_num 1 --nshot 2 --gen_length 128 --steps 128 --block_length 128 --data_path ./data/mbpp.json --version answer_token
python utils/judge_python_code.py \
    --folder_path results/mbpp_results \
    --output_path params/conf_params/mbpp/accuracy \
    --nshot 2 \
    --steps 128 \
    --gen_length 128 \
    --find_not_position