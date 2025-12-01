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