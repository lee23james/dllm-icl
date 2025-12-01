import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
import numpy as np
import json, os
import argparse
from pathlib import Path
#画出我想要的指标(总共画五张图)-》一个样本五个子图
# 1. conf变化,关注最后的曲线变化情况(digit、zero、other digit)
# 2. area变化,绘制平均值(长、宽、高)
# 3. token change变化,各个部分占比平均值(zero、other digit、answer tag、space、other positions)
# 目前就是,关于conf,那三张图x轴坐标是step,y轴坐标是conf值,绘制曲线,
# area和token change x轴是括号中的各个指标(因为都是平均值),y轴是各个位置对应的值

#读取data的详细数据
def load_json_or_jsonl(file_path):
    #不影响功能,最终会被覆盖
    data = []
    #只读
    with open(file_path,'r',encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            #变成list[dict]
            for line in f:
                data.append(json.loads(line))
        elif file_path.endswith('.json'):
            #变成dict(json.loads)
            data=json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    return data

def generate_conf_plots(json_dir, sample_index=0, task='sudoku',output_dir=None):
    """
    从JSON文件中读取数据，为指定样本生成5个子图的综合图表
    
    Args:
        json_dir: JSON文件所在目录（包含position_0.json到position_5.json）
        sample_index: 要绘制的样本索引
        output_dir: 输出目录，如果为None则使用json_dir
    """
    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取6个position文件
    positions_data = []
    position_indices = []  # 记录实际的position编号
    num_positions = 6
    
    for pos in range(num_positions):
        json_file = os.path.join(json_dir, f'{task}_position_{pos}.json')
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping position {pos}")
            continue
        data = load_json_or_jsonl(json_file)
        positions_data.append(data)
        position_indices.append(pos)  # 记录实际的position编号
    
    if len(positions_data) == 0:
        raise ValueError(f"No valid JSON files found in {json_dir}")
    
    # 检查sample_index是否有效
    sample_data = positions_data[0].get('sample_data', {})
    result_list = sample_data.get('result', [])
    if sample_index >= len(result_list):
        raise ValueError(f"sample_index {sample_index} out of range (max: {len(result_list)-1})")
    
    # 准备数据：收集所有positions的数据
    conf_digit_data = []  # 每个position的digit conf step_means
    conf_zero_mapped_data = []  # 每个position的zero_mapped conf step_means
    conf_other_digit_data = []  # 每个position的other_digit conf step_means
    token_change_data = []  # 每个position的token_change total_valid_count
    change_bbox_data = []  # 每个position的change_bbox
    
    position_labels = []
    
    for idx, (data, actual_pos) in enumerate(zip(positions_data, position_indices)):
        sample_data = data.get('sample_data', {})
        result_list = sample_data.get('result', [])
        history_token_cover_list = sample_data.get('history_token_cover', [])
        
        if sample_index >= len(result_list):
            continue
        
        result = result_list[sample_index]
        history_token_cover = history_token_cover_list[sample_index] if sample_index < len(history_token_cover_list) else {}
        
        # 提取conf数据
        conf = result.get('conf', {})
        digit_step_means = conf.get('digit', {}).get('step_means', [])
        zero_mapped_step_means = conf.get('zero_mapped', {}).get('step_means', [])
        other_digit_step_means = conf.get('other_digit', {}).get('step_means', [])
        
        # 处理null值，保留为np.nan以便断开绘制
        def process_step_means(step_means, target_length=32):
            processed = []
            for val in step_means:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    processed.append(np.nan)  # 保留为nan，用于断开绘制
                else:
                    processed.append(float(val))
            # 如果长度不足，补nan
            while len(processed) < target_length:
                processed.append(np.nan)
            # 如果长度超过，截断
            return processed[:target_length]
        
        conf_digit_data.append(process_step_means(digit_step_means))
        conf_zero_mapped_data.append(process_step_means(zero_mapped_step_means))
        conf_other_digit_data.append(process_step_means(other_digit_step_means))
        
        # 提取token_change数据，使用token_change_position_ratio作为y轴
        token_change = result.get('token_change', {})
        token_change_values = []
        ranges_order = ['digit', 'zero_mapped', 'other_digit', 'answer_tag', 'space_enter', 'other_positions']
        for range_name in ranges_order:
            token_change_position_ratio = token_change.get(range_name, {}).get('token_change_position_ratio', 0)
            token_change_values.append(float(token_change_position_ratio) if token_change_position_ratio is not None else 0.0)
        token_change_data.append(token_change_values)
        
        # 提取change_bbox数据
        change_bbox = history_token_cover.get('change_bbox', {})
        step_span = change_bbox.get('step_span')
        token_span = change_bbox.get('token_span')
        area = change_bbox.get('area')
        
        # 确保所有值都被提取，即使为0或None也记录
        bbox_values = [
            float(step_span) if step_span is not None else 0.0,
            float(token_span) if token_span is not None else 0.0,
            float(area) if area is not None else 0.0
        ]
        change_bbox_data.append(bbox_values)
        
        # 调试输出：检查每个position的数据
        print(f"Position {actual_pos}: step_span={bbox_values[0]}, token_span={bbox_values[1]}, area={bbox_values[2]}")
        
        position_labels.append(f'Position {actual_pos}')
    
    # 创建5个子图
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 设置颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_labels)))
    
    # 子图1: digit conf变化
    ax1 = fig.add_subplot(gs[0, 0])
    steps = list(range(32))
    for idx, (data, label) in enumerate(zip(conf_digit_data, position_labels)):
        # 使用plot会自动处理nan值，在nan处断开
        ax1.plot(steps, data, label=label, color=colors[idx], linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Conf Value', fontsize=12)
    ax1.set_title('Digit Conf Changes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: zero_mapped conf变化
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (data, label) in enumerate(zip(conf_zero_mapped_data, position_labels)):
        ax2.plot(steps, data, label=label, color=colors[idx], linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Conf Value', fontsize=12)
    ax2.set_title('Zero Mapped Conf Changes', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 子图3: other_digit conf变化
    ax3 = fig.add_subplot(gs[0, 2])
    for idx, (data, label) in enumerate(zip(conf_other_digit_data, position_labels)):
        ax3.plot(steps, data, label=label, color=colors[idx], linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Conf Value', fontsize=12)
    ax3.set_title('Other Digit Conf Changes', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: token_change占比
    ax4 = fig.add_subplot(gs[1, 0])
    x_labels = ['digit', 'zero_mapped', 'other_digit', 'answer_tag', 'space_enter', 'other_positions']
    x_pos = np.arange(len(x_labels))
    num_positions = len(position_labels)
    # 计算合适的柱宽，确保6个柱子能清晰显示
    total_width = 0.8  # 每组柱子的总宽度
    width = total_width / num_positions  # 每个柱子的宽度
    
    # 绘制分组柱状图
    for idx, (data, label) in enumerate(zip(token_change_data, position_labels)):
        # 计算每个柱子的偏移量，使柱子居中分布
        offset = (idx - (num_positions - 1) / 2) * width
        ax4.bar(x_pos + offset, data, width, label=label, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Token Change Range', fontsize=12)
    ax4.set_ylabel('Token Change Position Ratio', fontsize=12)
    ax4.set_title('Token Change Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
    ax4.legend(fontsize=8, ncol=2, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 子图5: change_bbox变化 - 使用断裂y轴放大0-50范围
    ax5 = fig.add_subplot(gs[1, 1])
    # 使用分段y轴：step_span和token_span在0-50范围（细粒度），area在200-1000范围（粗粒度）
    # x轴是三个标签位置：step_span, token_span, area
    
    bbox_labels = ['step_span', 'token_span', 'area']
    x_positions = [0, 1, 2]  # 三个标签的x轴位置
    
    # 使用断裂y轴：0-50范围占据更多空间，200-1000范围压缩
    # 设置y轴范围：让0-50占据大部分空间（0-80%），200-1000占据小部分空间（80-100%）
    # 使用非线性映射：将0-50映射到0-400，200-1000映射到450-550
    def map_y_value(val):
        if val <= 50:
            # 0-50映射到0-400，放大8倍
            return val * 8
        else:
            # 200-1000映射到450-550，压缩
            # 先归一化到0-1，然后映射到450-550
            normalized = (val - 200) / 800.0  # 200-1000 -> 0-1
            return 450 + normalized * 100  # 映射到450-550
    
    # 重新绘制数据点，使用映射后的y值
    ax5.clear()
    print(f"Total positions to plot: {len(change_bbox_data)}")
    for idx, (data, label) in enumerate(zip(change_bbox_data, position_labels)):
        step_span_val = data[0]
        token_span_val = data[1]
        area_val = data[2]
        
        # 调试输出：确认每个点都被绘制
        print(f"Plotting {label}: step_span={step_span_val}, token_span={token_span_val}, area={area_val}")
        
        # 使用映射后的y值绘制，每个position用对应颜色
        # 添加透明度，让坐标点更透明
        # 确保即使值为0也绘制（可能显示在y=0附近）
        ax5.plot(x_positions[0], map_y_value(step_span_val), marker='o', color=colors[idx], 
                markersize=8, label=label, linestyle='None', alpha=0.6, zorder=10)
        ax5.plot(x_positions[1], map_y_value(token_span_val), marker='s', color=colors[idx], 
                markersize=8, linestyle='None', alpha=0.6, zorder=10)
        # area值如果为0，可能不在200-1000范围内，需要特殊处理
        if area_val > 0:
            ax5.plot(x_positions[2], map_y_value(area_val), marker='^', color=colors[idx], 
                    markersize=8, linestyle='None', alpha=0.6, zorder=10)
        else:
            # 如果area为0，也绘制，但显示在较低位置
            ax5.plot(x_positions[2], map_y_value(0), marker='^', color=colors[idx], 
                    markersize=8, linestyle='None', alpha=0.6, zorder=10)
    
    # 设置x轴
    ax5.set_xlim(-0.5, 2.5)
    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(bbox_labels)
    
    # 设置y轴：显示映射后的范围
    ax5.set_ylim(-20, 560)
    
    # 创建自定义y轴刻度：显示原始值，但位置是映射后的
    fine_y_ticks_original = list(range(0, 55, 10))  # 0, 10, 20, 30, 40, 50
    coarse_y_ticks_original = list(range(200, 1100, 200))  # 200, 400, 600, 800, 1000
    
    fine_y_ticks_mapped = [map_y_value(t) for t in fine_y_ticks_original]
    coarse_y_ticks_mapped = [map_y_value(t) for t in coarse_y_ticks_original]
    
    all_y_ticks_mapped = fine_y_ticks_mapped + coarse_y_ticks_mapped
    all_y_labels = [str(t) for t in fine_y_ticks_original] + [str(t) for t in coarse_y_ticks_original]
    
    ax5.set_yticks(all_y_ticks_mapped)
    ax5.set_yticklabels(all_y_labels)
    
    # 添加y轴断裂标记
    ax5.axhline(y=420, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    # 添加断裂符号
    ax5.plot([-0.5, -0.3], [410, 410], 'k-', linewidth=2)
    ax5.plot([-0.5, -0.3], [430, 430], 'k-', linewidth=2)
    
    # 添加分段说明（调整位置，避免遮挡）
    ax5.text(2.35, 200, 'Fine-grained\n(0-50)', 
            ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.text(2.35, 500, 'Coarse-grained\n(200-1000)', 
            ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 添加图例说明
    from matplotlib.lines import Line2D
    # Metrics图例（标记类型）
    legend_elements = [Line2D([0], [0], marker='o', color='gray', label='step_span', 
                              markersize=8, linestyle='None', markeredgecolor='black', markeredgewidth=0.5),
                       Line2D([0], [0], marker='s', color='gray', label='token_span', 
                              markersize=8, linestyle='None', markeredgecolor='black', markeredgewidth=0.5),
                       Line2D([0], [0], marker='^', color='gray', label='area', 
                              markersize=8, linestyle='None', markeredgecolor='black', markeredgewidth=0.5)]
    
    # 先添加Metrics图例
    metrics_legend = ax5.legend(handles=legend_elements, fontsize=8, loc='lower left', 
                                title='Metrics', framealpha=0.9)
    ax5.add_artist(metrics_legend)
    
    # 创建Positions图例，使用实际绘制的颜色
    position_handles = [Line2D([0], [0], marker='o', color=colors[idx], label=label,
                               markersize=8, linestyle='None', markeredgecolor='black', markeredgewidth=0.5)
                        for idx, label in enumerate(position_labels)]
    
    # 添加Positions图例，放在右下角，避免遮挡数据
    positions_legend = ax5.legend(handles=position_handles, fontsize=7, loc='lower right', 
                                  title='Positions', ncol=2, framealpha=0.9)
    
    ax5.set_xlabel('BBox Metric', fontsize=12)
    ax5.set_ylabel('Value (Segmented Scale)', fontsize=12)
    ax5.set_title('Change BBox Metrics', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 保存图片
    output_file = os.path.join(output_dir, f'conf_plot_sample_{sample_index}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'已保存图片到: {output_file}')
    plt.close()


#这边我想创建一个函数,用于生成相应的关系函数
#给出json所在的目录,方便去获取对应位置的json文件夹,之后去获取熵值和准确率的关系
def generate_conf_accuracy_plot(
    json_dir,
    task='sudoku',
    output_dir=None,
    num_shots=None,
    num_steps=None,
):
    """
    在“同一 task + shot + step”组里，对所有 position 的
    digit_symbol mean_of_means（x轴）与 accuracy（y轴）画散点图。
    """
    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)

    default_shots = [3, 4, 5, 6, 7, 8, 9, 10]
    default_steps = [16, 32]
    shot_list = num_shots if num_shots is not None else default_shots
    step_list = num_steps if num_steps is not None else default_steps

    groups = {}  # key=(nshot, step), value=list of dicts
    all_entries = []
    #这里是按照key、value的方式进行计算,看能不能满足需求,同时通过相关区域进行离散点的计算
    #现在问题就是看能不能通过不同的task来区分不同的数据情况
    for nshot in shot_list:
        for step in step_list:
            key = (nshot, step)
            groups[key] = []
            for pos in range(nshot + 1):
                json_file = os.path.join(
                    json_dir, f'{task}_shot_{nshot}_position_{pos}_step_{step}.json'
                )
                if not os.path.exists(json_file):
                    print(
                        f"Warning: {json_file} not found, skipping shot_{nshot}_position_{pos}_step_{step}"
                    )
                    continue

                data = load_json_or_jsonl(json_file)
                metadata = data.get('metadata', {})
                all_samples = data.get('all_samples_result', {})
                #不同的位置,想要观察的信息不太一致,所以我在这里对于task进行相应的分析
                #不同的文件但是遍历那个映射的是相同的需要去关注的地方
                if task =='countdown':
                    conf_digit_symbol = (
                        all_samples.get('conf', {})
                        .get('digit_symbol', {})
                        .get('mean_of_means')
                    )
                elif task=='sudoku':
                    conf_digit_symbol =(
                        all_samples.get('conf',{})
                        .get('digit',{})
                        .get('mean_of_means')
                    )
                else:
                    raise NotImplementedError(f"Task {task} not implemented.")

                accuracy = metadata.get('accuracy')

                if conf_digit_symbol is None or accuracy is None:
                    print(
                        f"Warning: Missing conf or accuracy in {json_file}, skipping."
                    )
                    continue
                entry = {
                    'position': pos,
                    'conf': float(conf_digit_symbol),
                    'accuracy': float(accuracy),
                    'shot': nshot,
                    'step': step,
                }
                groups[key].append(entry)
                all_entries.append(entry)

    for (nshot, step), entries in groups.items():
        if len(entries) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        conf_vals = [entry['conf'] for entry in entries]
        acc_vals = [entry['accuracy'] for entry in entries]

        scatter = ax.scatter(
            conf_vals, acc_vals, c=conf_vals, cmap='viridis', s=60, edgecolor='black'
        )

        for entry in entries:
            ax.text(
                entry['conf'],
                entry['accuracy'],
                f"pos {entry['position']}",
                fontsize=8,
                ha='left',
                va='bottom',
            )

        ax.set_xlabel('digit_symbol mean_of_means', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{task} | shot={nshot}, step={step}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label='Conf Value')

        output_path = os.path.join(output_dir, f'{task}_shot_{nshot}_step_{step}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'已保存图像: {output_path}')

    if len(all_entries) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        conf_vals = [entry['conf'] for entry in all_entries]
        acc_vals = [entry['accuracy'] for entry in all_entries]
        colors = [entry['shot'] for entry in all_entries]

        scatter = ax.scatter(
            conf_vals,
            acc_vals,
            c=colors,
            cmap='plasma',
            s=60,
            edgecolor='black',
        )
        ax.set_xlabel('digit_symbol mean_of_means', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{task} | All shots & steps', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Shot Number')

        output_path = os.path.join(output_dir, f'{task}_all_shots_steps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'已保存图像: {output_path}')

def generate_tokenchange_accuracy_plot(
    json_dir,
    task='sudoku',
    output_dir=None,
    num_shots=None,
    num_steps=None,
):
    """
    在"同一 task + shot + step"组里，对所有 position 的
    area/token_span/step_span/token_change_total（x轴）与 accuracy（y轴）画散点图。
    每个 (shot, step) 组合生成一张包含四个子图的图：area vs accuracy, token_span vs accuracy, step_span vs accuracy, token_change_total vs accuracy
    """
    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)

    default_shots = [3, 4, 5, 6, 7, 8, 9, 10]
    default_steps = [16, 32]
    shot_list = num_shots if num_shots is not None else default_shots
    step_list = num_steps if num_steps is not None else default_steps

    groups = {}  # key=(nshot, step), value=list of dicts
    all_entries = []
    
    for nshot in shot_list:
        for step in step_list:
            key = (nshot, step)
            groups[key] = []
            for pos in range(nshot + 1):
                json_file = os.path.join(
                    json_dir, f'{task}_shot_{nshot}_position_{pos}_step_{step}.json'
                )
                if not os.path.exists(json_file):
                    print(
                        f"Warning: {json_file} not found, skipping shot_{nshot}_position_{pos}_step_{step}"
                    )
                    continue

                data = load_json_or_jsonl(json_file)
                metadata = data.get('metadata', {})
                all_samples = data.get('all_samples_result', {})
                
                # 从 change_bbox 中提取 area、token_span、step_span、token_change_total
                change_bbox = all_samples.get('change_bbox', {})
                area = change_bbox.get('area', {}).get('mean')
                token_span = change_bbox.get('token_span', {}).get('mean')
                step_span = change_bbox.get('step_span', {}).get('mean')
                token_change_total = change_bbox.get('token_change_total', {}).get('mean')
                
                accuracy = metadata.get('accuracy')

                if area is None or token_span is None or step_span is None or token_change_total is None or accuracy is None:
                    print(
                        f"Warning: Missing change_bbox metrics or accuracy in {json_file}, skipping."
                    )
                    continue
                    
                entry = {
                    'position': pos,
                    'area': float(area),
                    'token_span': float(token_span),
                    'step_span': float(step_span),
                    'token_change_total': float(token_change_total),
                    'accuracy': float(accuracy),
                    'shot': nshot,
                    'step': step,
                }
                groups[key].append(entry)
                all_entries.append(entry)

    # 为每个 (shot, step) 组合生成一张包含四个子图的图
    metrics = ['area', 'token_span', 'step_span', 'token_change_total']
    metric_labels = {
        'area': 'Area (step_span × token_span)',
        'token_span': 'Token Span',
        'step_span': 'Step Span',
        'token_change_total': 'Token Change Total'
    }
    
    for (nshot, step), entries in groups.items():
        if len(entries) == 0:
            continue

        # 为每个 (shot, step) 组合生成一张包含四个子图的图
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_vals = [entry[metric] for entry in entries]
            acc_vals = [entry['accuracy'] for entry in entries]

            scatter = ax.scatter(
                metric_vals, acc_vals, c=metric_vals, cmap='viridis', s=60, edgecolor='black'
            )

            for entry in entries:
                ax.text(
                    entry[metric],
                    entry['accuracy'],
                    f"pos {entry['position']}",
                    fontsize=8,
                    ha='left',
                    va='bottom',
                )

            ax.set_xlabel(metric_labels[metric], fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(metric_labels[metric], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 为整张图添加总标题
        fig.suptitle(f'{task} | shot={nshot}, step={step}', fontsize=16, fontweight='bold', y=1.02)
        
        # 先调整布局，为 colorbar 留出空间
        plt.tight_layout(rect=[0, 0, 0.93, 0.98])
        
        # 为整张图添加统一的 colorbar
        cbar = fig.colorbar(scatter, ax=axes, label='Metric Value', pad=0.02)
        output_path = os.path.join(output_dir, f'{task}_shot_{nshot}_step_{step}_tokenchange.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'已保存图像: {output_path}')

    # 生成总图：所有 shot 和 step 的数据点，四张子图合并为一张
    if len(all_entries) > 0:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        colors = [entry['shot'] for entry in all_entries]
        scatter_list = []  # 保存所有 scatter 对象
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_vals = [entry[metric] for entry in all_entries]
            acc_vals = [entry['accuracy'] for entry in all_entries]

            scatter = ax.scatter(
                metric_vals,
                acc_vals,
                c=colors,
                cmap='plasma',
                s=60,
                edgecolor='black',
            )
            scatter_list.append(scatter)  # 保存 scatter 对象
            ax.set_xlabel(metric_labels[metric], fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'{metric_labels[metric]}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 为整张图添加总标题
        fig.suptitle(f'{task} | All shots & steps', fontsize=16, fontweight='bold', y=1.02)
        
        # 先调整布局，为 colorbar 留出空间
        plt.tight_layout(rect=[0, 0, 0.93, 0.98])
        
        # 为整张图添加统一的 colorbar（使用最后一个 scatter，因为所有子图使用相同的颜色映射）
        cbar = fig.colorbar(scatter_list[-1], ax=axes, label='Shot Number', pad=0.02)
        output_path = os.path.join(output_dir, f'{task}_all_shots_steps_tokenchange.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'已保存图像: {output_path}')

#准备进行对应的画图
def main(args):
    json_dir=args.json_dir
    task=args.task
    output_dir=args.output_dir
    sample_index=args.sample_index
    #这里是三个bool量,去决定应该怎么画图
    paint_conf_acc=args.paint_conf_acc
    paint_conf=args.paint_conf
    paint_tokenchange_acc=args.paint_tokenchange_acc
    #这里一般采用默认配置环节
    # num_shots=args.num_shots
    # num_steps=args.num_steps
    #这里要进行相关的标记进行画图
    if paint_conf_acc:
        generate_conf_accuracy_plot(json_dir, task, output_dir)
    if paint_conf:
        generate_conf_plots(json_dir, sample_index, task, output_dir)
    if paint_tokenchange_acc:
        generate_tokenchange_accuracy_plot(json_dir, task, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, required=True, help='JSON文件所在目录')
    parser.add_argument('--sample_index', type=int, default=0, help='要绘制的样本索引')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--task', type=str, required=True, help='任务名称')
    parser.add_argument('--paint_conf_acc',action='store_true',default=False,help='是否绘制置信度-准确率图')
    parser.add_argument('--paint_conf',action='store_true',default=False,help='是否绘制置信度图')
    parser.add_argument('--paint_tokenchange_acc',action='store_true',default=False,help='是否绘制token change指标-准确率图')
    args = parser.parse_args()
    main(args)
    # generate_conf_plots(args.json_dir, args.sample_index, args.task, args.output_dir)
