#!/bin/bash

# Stage 1 数据生成脚本
# 用于控制各种参数，包括test_sample数量、候选池数量等

# ============================================================================
# 配置参数（可以根据需要修改）
# ============================================================================

# 基础配置
CONFIG_NAME="generate_data.yaml"
PROJECT_ROOT="/home/lzh/llada-icl"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# 参数说明
# ============================================================================
# 
# 【Sampler参数】
# sampler.anchor_sample_num: anchor数量（要处理的test_sample数量）
# sampler.candidate_num: 候选池大小（每个anchor的候选ICD数量）
# sampler.overwrite: 是否覆盖缓存（true/false）
# sampler.cache_dir: 缓存目录（默认: "./cache/gsm8k"）
# 
# 【核心算法参数】
# few_shot_num: 要选择的ICD数量（循环轮数）
# beam_size: Beam Search的beam大小（每轮保留的序列数）
# metric: 插入策略（"no_order" 或 "order"）
# scorer: 评分函数（目前只支持 "infoscore"）
# 
# 【性能参数】
# batch_size: 批处理大小（一般设为1）
# mc_num: Monte Carlo采样次数（影响评分精度和速度，默认128）
# cfg_scale: CFG scale（默认0.0）
# 
# 【多GPU参数】
# use_multi_gpu: 是否使用多GPU（true/false）
# gpu_ids: GPU ID列表，例如 [0,1,2,3]
# sleep_time: 多卡模式下，各进程启动间隔（秒，默认10）
# merge_results: 多卡模式下，是否合并所有rank的结果（true/false）
# 
# 【输出参数】
# output_dir: 输出目录（默认: "./generated_icd_data"）
# 
# 【模型参数】（通常在config文件中配置，但可以通过shell覆盖）
# infer_model.model_path: 模型路径（必需）
# infer_model.mask_id: mask token的ID（默认126336）
# infer_model.mask_length: mask token长度（默认256）
# infer_model.torch_dtype: 模型精度（"bfloat16", "float16", "float32"）
# 
# 【数据集参数】（通常在config文件中配置）
# dataset.train_path: 训练集路径
# dataset.version: 数据集版本（"local"）
# 
# 【任务参数】（通常在config文件中配置）
# task.task_name: 任务名称（"gsm8k"）
# task.split_token: ICD之间的分隔符（默认"\n\n"）
# 
# ============================================================================

# ============================================================================
# 预设配置（可以根据需要选择）
# ============================================================================

# 小批量测试配置（快速验证）
SMALL_TEST_CONFIG=(
    "sampler.anchor_sample_num=10"      # 处理50个test_sample
    "sampler.candidate_num=10"          # 每个候选池20个ICD
    "few_shot_num=4"                    # 选择4个ICD
    "beam_size=3"                      # beam_size=10
    "metric=no_order"                   # 只在两侧插入
    "mc_num=20"                         # Monte Carlo采样次数（加快速度）
    "batch_size=1"                      # 批处理大小
)

# 中等规模测试配置
MEDIUM_TEST_CONFIG=(
    "sampler.anchor_sample_num=100"    # 处理100个test_sample
    "sampler.candidate_num=32"         # 每个候选池32个ICD
    "few_shot_num=4"                   # 选择4个ICD
    "beam_size=20"                     # beam_size=20
    "metric=no_order"                  # 只在两侧插入
    "mc_num=128"                       # Monte Carlo采样次数（默认）
    "batch_size=1"                     # 批处理大小
)

# 大规模配置（完整运行）
LARGE_CONFIG=(
    "sampler.anchor_sample_num=5000"  # 处理5000个test_sample
    "sampler.candidate_num=64"         # 每个候选池64个ICD
    "few_shot_num=4"                   # 选择4个ICD
    "beam_size=50"                     # beam_size=50
    "metric=no_order"                  # 只在两侧插入
    "mc_num=128"                       # Monte Carlo采样次数（默认）
    "batch_size=1"                     # 批处理大小
    "merge_results=true"               # 多卡模式下合并结果
)

# 完整配置（关注顺序）
FULL_ORDER_CONFIG=(
    "sampler.anchor_sample_num=1000"  # 处理1000个test_sample
    "sampler.candidate_num=64"         # 每个候选池64个ICD
    "few_shot_num=4"                   # 选择4个ICD
    "beam_size=20"                     # beam_size=20
    "metric=order"                     # 所有位置插入（更慢但更精确）
    "mc_num=128"                       # Monte Carlo采样次数（默认）
    "batch_size=1"                     # 批处理大小
)

# ============================================================================
# 函数定义
# ============================================================================

# 打印帮助信息
print_help() {
    cat << EOF
用法: $0 [选项] [额外参数...]

选项:
    -h, --help              显示帮助信息
    -s, --small             小批量测试（50个test_sample，20个候选，4个ICD）
    -m, --medium            中等规模测试（100个test_sample，32个候选，4个ICD）
    -l, --large             大规模运行（5000个test_sample，64个候选，4个ICD）
    -o, --order             完整配置（关注顺序，1000个test_sample）
    -c, --custom            自定义配置（需要提供参数）
    
额外参数:
    可以覆盖预设配置，例如：
    $0 -s sampler.candidate_num=30 few_shot_num=3 mc_num=32
    
常用参数示例:
    # Sampler参数
    sampler.anchor_sample_num=100      # test_sample数量
    sampler.candidate_num=32           # 候选池大小
    sampler.overwrite=true              # 覆盖缓存
    
    # 核心算法参数
    few_shot_num=4                      # ICD数量
    beam_size=10                       # Beam Search大小
    metric=no_order                     # 插入策略（no_order/order）
    mc_num=128                          # Monte Carlo采样次数
    
    # 多GPU参数
    use_multi_gpu=true                  # 启用多GPU
    gpu_ids=[0,1,2,3]                   # GPU列表
    sleep_time=10                       # 进程启动间隔（秒）
    merge_results=true                  # 合并多卡结果
    
    # 输出参数
    output_dir="./my_results"           # 输出目录
    
    # 模型参数（覆盖config）
    infer_model.model_path="/path/to/model"
    infer_model.torch_dtype="float16"
    
示例:
    # 小批量测试
    $0 -s
    
    # 中等规模测试，并覆盖候选池大小和Monte Carlo次数
    $0 -m sampler.candidate_num=50 mc_num=64
    
    # 自定义配置
    $0 -c sampler.anchor_sample_num=100 sampler.candidate_num=32 few_shot_num=3 beam_size=10
    
    # 使用多GPU
    $0 -l use_multi_gpu=true gpu_ids=[0,1,2,3] merge_results=true
    
    # 加快速度（降低精度）
    $0 -s mc_num=32 batch_size=1
    
    # 提高精度（降低速度）
    $0 -m mc_num=256 beam_size=30
    
    # 覆盖模型路径
    $0 -s infer_model.model_path="/path/to/your/model"
    
    # 覆盖缓存设置
    $0 -s sampler.overwrite=true sampler.cache_dir="./my_cache"

EOF
}

# 运行Stage 1
run_stage1() {
    local config_args=("$@")
    
    echo "============================================================================"
    echo "Stage 1: 数据生成"
    echo "============================================================================"
    echo "配置参数:"
    printf "  %s\n" "${config_args[@]}"
    echo "============================================================================"
    echo ""
    
    # 运行Python脚本
    python generate_data_main.py \
        "${config_args[@]}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "============================================================================"
        echo "✅ Stage 1 完成！"
        echo "============================================================================"
        echo "结果文件: ./generated_icd_data/gsm8k_icd_results.json"
        echo "日志文件: ./outputs/*/generate_data.log"
    else
        echo ""
        echo "============================================================================"
        echo "❌ Stage 1 失败！退出码: $exit_code"
        echo "============================================================================"
    fi
    
    return $exit_code
}

# ============================================================================
# 主逻辑
# ============================================================================

# 解析命令行参数
if [ $# -eq 0 ]; then
    print_help
    exit 1
fi

case "$1" in
    -h|--help)
        print_help
        exit 0
        ;;
    -s|--small)
        shift
        run_stage1 "${SMALL_TEST_CONFIG[@]}" "$@"
        ;;
    -m|--medium)
        shift
        run_stage1 "${MEDIUM_TEST_CONFIG[@]}" "$@"
        ;;
    -l|--large)
        shift
        run_stage1 "${LARGE_CONFIG[@]}" "$@"
        ;;
    -o|--order)
        shift
        run_stage1 "${FULL_ORDER_CONFIG[@]}" "$@"
        ;;
    -c|--custom)
        shift
        if [ $# -eq 0 ]; then
            echo "错误: 自定义配置需要提供参数"
            echo "示例: $0 -c sampler.anchor_sample_num=100 sampler.candidate_num=32"
            exit 1
        fi
        run_stage1 "$@"
        ;;
    *)
        # 如果没有指定选项，直接使用提供的参数
        run_stage1 "$@"
        ;;
esac
