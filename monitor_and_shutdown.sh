#!/bin/bash
# Monitor experiment and shutdown when complete

LOG_FILE="/hy-tmp/latency_exp_2gpu.log"
RESULT_DIR="/hy-tmp/dllm-icl/results/latency_exp_2gpu"

echo "[$(date)] 开始监控实验..."

while true; do
    sleep 60

    # 检查实验是否还在运行
    if ! pgrep -f "eval.py" > /dev/null; then
        echo "[$(date)] 实验进程已结束"

        # 等待结果文件写入完成
        sleep 30

        # 整理结果
        echo "[$(date)] 开始整理结果..."
        python3 << 'PYTHON_SCRIPT'
import os
import json

result_dir = "/hy-tmp/dllm-icl/results/latency_exp_2gpu"

# 收集所有结果
results = {
    "vanilla": {},
    "auto_icl": {}
}

datasets = ["gsm8k", "sudoku", "countdown", "math500", "mbpp"]

for mode in ["vanilla", "auto_icl"]:
    for dataset in datasets:
        result_file = os.path.join(result_dir, mode, dataset, "result.txt")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                content = f.read()
                # 提取准确率
                for line in content.split('\n'):
                    if 'Accuracy:' in line or 'Total Accuracy:' in line:
                        try:
                            acc = float(line.split(':')[-1].strip())
                            results[mode][dataset] = acc
                        except:
                            pass

# 打印表格
print("\n" + "="*60)
print("实验结果汇总表")
print("="*60)
print(f"{'数据集':<15} {'Vanilla ICL':<15} {'Auto-ICL':<15}")
print("-"*60)

for dataset in datasets:
    v_acc = results["vanilla"].get(dataset, "N/A")
    a_acc = results["auto_icl"].get(dataset, "N/A")
    if v_acc != "N/A":
        v_acc = f"{v_acc*100:.2f}%"
    if a_acc != "N/A":
        a_acc = f"{a_acc*100:.2f}%"
    print(f"{dataset:<15} {v_acc:<15} {a_acc:<15}")

print("="*60)

# 保存到文件
with open("/hy-tmp/dllm-icl/final_results.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("实验结果汇总表\n")
    f.write("="*60 + "\n")
    f.write(f"{'数据集':<15} {'Vanilla ICL':<15} {'Auto-ICL':<15}\n")
    f.write("-"*60 + "\n")
    for dataset in datasets:
        v_acc = results["vanilla"].get(dataset, "N/A")
        a_acc = results["auto_icl"].get(dataset, "N/A")
        if v_acc != "N/A":
            v_acc = f"{v_acc*100:.2f}%"
        if a_acc != "N/A":
            a_acc = f"{a_acc*100:.2f}%"
        f.write(f"{dataset:<15} {v_acc:<15} {a_acc:<15}\n")
    f.write("="*60 + "\n")

print("\n[完成] 结果已保存到 /hy-tmp/dllm-icl/final_results.txt")
PYTHON_SCRIPT

        echo "[$(date)] 结果整理完成，准备关机..."
        sleep 5
        shutdown -h now
        exit 0
    fi

    # 每10分钟输出一次进度
    if [ $(($(date +%s) % 600)) -lt 60 ]; then
        echo "[$(date)] 实验仍在运行中..."
        tail -5 "$LOG_FILE" | head -3
    fi
done
