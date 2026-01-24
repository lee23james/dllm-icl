# 若尚未创建环境，可以这样新建（假设你想用 Python 3.10）：
```bash
conda create -n llada python=3.10.18 -y
conda activate llada
```

# 安装 requirements.txt 里的依赖
```bash
pip install -r requirements.txt
```

# 运行脚本
```bash
bash scripts/eval_position_conf_base.sh
bash scripts/eval_position_semiar_inst.sh
```

# 你可能需要下载模型到对应文件夹,然后脚本里的模型地址

目前只考虑llada对gsm8k的过程
#这里再想一个分类器的做法
generate_data.py只是筛选候选集的过程
utils.py存的是分数的计算情况,其中用到之前的函数
open_mmicl文件夹存的是之前的怎么用来进行推理包括拼接prompt(就是拿来计算最终效果的)
configs文件夹主要是用来区分数据集的,确定数据集应该怎么被使用
