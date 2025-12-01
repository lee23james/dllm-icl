import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
import numpy as np
import json, os
import argparse
from pathlib import Path

#生成的是dict,直接用data['rollout_list']来获取list[dict[list[float]]],第一个list是步骤,第二个list是层,dict是各种字段
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

def _plot_rollout_heatmap_to_file(data, vmin, vmax, filename, token_position=None, first_mask_pos=None, last_mask_pos=None):
    """
    绘制rollout热力图并保存到文件
    仿照用户提供的_plot_heatmap_to_file函数
    
    Args:
        data: rollout数据，形状为[S, S]
        vmin: 颜色条最小值
        vmax: 颜色条最大值
        filename: 保存文件名
        token_position: dict，包含各个token位置信息，如{'question_token_positions': [start, end], ...}
        first_mask_pos: int，第一个mask token的位置（answer区域起始位置）
        last_mask_pos: int，最后一个mask token的位置（answer区域结束位置）
    """
    # 根据是否有token_position或mask位置决定布局
    has_segments = token_position is not None or (first_mask_pos is not None and last_mask_pos is not None)
    
    if has_segments:
        # 有分段条：主图 + 底部分段条
        figsize = (6.8, 6.0)  # 增加高度以容纳分段条
        fig = plt.figure(figsize=figsize)
        right_margin = 0.82
        gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], left=0.15, right=right_margin, top=0.98, bottom=0.12, hspace=0.05)
        ax = fig.add_subplot(gs[0])
    else:
        # 没有分段条：只有主图
        figsize = (6.8, 5.5)
        fig = plt.figure(figsize=figsize)
        right_margin = 0.82
        gs = gridspec.GridSpec(1, 1, left=0.15, right=right_margin, top=0.98, bottom=0.15)
        ax = fig.add_subplot(gs[0])
    # 根据文件格式决定是否使用rasterized（PNG不需要）
    is_png = filename.lower().endswith('.png')
    # 使用反转的colormap，让黄色表示最高值，蓝色表示最低值
    hmap = sns.heatmap(
        data,
        cmap='YlGnBu_r',  # 反转colormap：YlGnBu_r表示黄色(Y)最高，蓝色(Bu)最低
        annot=False,
        cbar=False,
        square=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        rasterized=False if is_png else True,
        linewidths=0  # 去掉网格线，让图片更平滑
    )
    matrix_size = data.shape[0]
    ticks = np.linspace(0, matrix_size, 5, dtype=int)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=24, rotation=0)
    ax.set_ylabel('query position', fontsize=28)
    ax.set_xlabel('key position',fontsize=28)
    ax.set_yticklabels(ticks, fontsize=24)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    #这里并没有设置图片
    ax.set_title("") 
    #在这里手动创建和自定义颜色条
    ax_pos = ax.get_position()
    shrink_factor = 0.9 
    cax_width = 0.02   
    cax_pad = 0.01
    cax_left = ax_pos.x1 + cax_pad
    cax_height = ax_pos.height * shrink_factor
    cax_bottom = ax_pos.y0 + (ax_pos.height * (1 - shrink_factor) / 2.0)
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    cbar = fig.colorbar(hmap.collections[0], cax=cax)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    #我想看一下能不能设置颜色条
    #这里是硬编码    
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])

    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(axis='y', pad=5, labelsize=24)
    cbar.set_label('attention rollout', size=28, labelpad=0)
    
    # 添加底部分段条（如果有token_position信息）
    if has_segments:
        ax_seg = fig.add_subplot(gs[1])
        matrix_size = data.shape[0]
        ax_seg.set_xlim(0, matrix_size)
        ax_seg.set_ylim(0, 1)
        ax_seg.axis('off')  # 隐藏坐标轴
        
        # 定义各个区域的颜色和标签
        segment_config = {
            'system_token_positions': ('S', '#87CEEB'),      # 天蓝色
            'question_token_positions': ('Q', '#FFA500'),    # 橙色
            'example1_token_positions': ('E1', '#90EE90'),   # 浅绿色
            'example2_token_positions': ('E2', '#FFB6C1'),   # 浅粉色
            'example3_token_positions': ('E3', '#DDA0DD'),    # 梅红色
            'example4_token_positions': ('E4', '#F0E68C'),    # 卡其色
            'example5_token_positions': ('E5', '#ADD8E6'),    # 浅蓝色
        }
        
        # 收集所有非None的位置段
        segments = []
        
        # 从token_position中收集位置段
        if token_position is not None:
            for key, (label, color) in segment_config.items():
                if key in token_position and token_position[key] is not None:
                    pos = token_position[key]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        start, end = pos[0], pos[1]
                        segments.append((label, start, end, color))
        
        # 添加answer区域（first_mask_pos到last_mask_pos）
        if first_mask_pos is not None and last_mask_pos is not None:
            segments.append(('A', first_mask_pos, last_mask_pos, '#FFD700'))  # 金色
        
        # 绘制分段矩形和标签
        for label, start, end, color in segments:
            # 绘制矩形
            rect = Rectangle(
                (start, 0),
                end - start + 1,
                1,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax_seg.add_patch(rect)
            
            # 添加标签（居中）
            mid_x = (start + end + 1) / 2
            ax_seg.text(
                mid_x, 0.5, label,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='black'
            )
        
        # 添加图例（显示在右侧）
        if segments:
            legend_elements = [
                Patch(facecolor=color, edgecolor='black', label=label)
                for label, _, _, color in segments
            ]
            ax_seg.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1.0, 1.0), fontsize=8)
    
    # 提高PNG的DPI和优化参数
    plt.savefig(filename, format='png', dpi=600, bbox_inches='tight', pad_inches=0.0, 
                facecolor='white', edgecolor='none', transparent=False)
    plt.close(fig)
    print(f"Saved figure: {filename}")


def generate_rollout_heatmaps(
    json_file_path: str,#输入的json文件路径
    output_dir: str = None,#输出目录
    basename: str = None#输出文件的前缀
):
    """
    从JSON文件中读取rollout数据，为每个step生成热力图
    
    Args:
        json_file_path: JSON文件路径，包含rollout_list
        output_dir: 输出目录，如果为None则使用JSON文件所在目录
        basename: 输出文件名前缀，如果为None则使用JSON文件名
    """
    # 设置字体
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
    except Exception as e:
        print(f"Times New Roman font not found. Using default font. Error: {e}")
    
    # 加载JSON数据
    data = load_json_or_jsonl(json_file_path)
    
    # 获取rollout_list
    if isinstance(data, dict):
        rollout_list = data.get('rollout_list', [])
    elif isinstance(data, list):
        rollout_list = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    if not rollout_list:
        raise ValueError("rollout_list is empty or not found")
    
    # 设置输出目录和文件名前缀
    json_path = Path(json_file_path)
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if basename is None:
        basename = json_path.stem
    
    # 将所有rollout数据转换为numpy数组，并计算全局的vmin和vmax
    rollout_arrays = []
    for item in rollout_list:
        if 'rollout' not in item:
            print(f"Warning: item missing 'rollout' key, skipping. Item keys: {item.keys()}")
            continue
        rollout_data = item['rollout']
        # 转换为numpy数组
        rollout_array = np.array(rollout_data)
        rollout_arrays.append(rollout_array)
    
    if not rollout_arrays:
        raise ValueError("No valid rollout data found")
    
    # 计算全局的vmin和vmax
    vmin = min(np.min(arr) for arr in rollout_arrays)
    vmax = max(np.max(arr) for arr in rollout_arrays)
    
    # 为每个step生成热力图
    for item in rollout_list:
        if 'rollout' not in item:
            continue
        
        step = item.get('step', 'unknown')
        rollout_data = item['rollout']
        rollout_array = np.array(rollout_data)
        
        # 获取token_position信息（每个item都有相同的token_position）
        token_position = item.get('token_position', None)
        first_mask_pos = item.get('first_mask_pos', None)
        last_mask_pos = item.get('last_mask_pos', None)
        
        # 生成文件名
        filename = output_dir / f"{basename}_step_{step}.png"
        
        # 绘制并保存热力图
        _plot_rollout_heatmap_to_file(
            data=rollout_array,
            vmin=vmin,
            vmax=0.01,
            filename=str(filename),
            token_position=token_position,
            first_mask_pos=first_mask_pos,
            last_mask_pos=last_mask_pos
        )
    
    print(f"Generated {len(rollout_arrays)} heatmap figures in {output_dir}")

#目前要进行的是层分析,因为光是看到总体,我并没有看出什么很明显的变化,主要更多的还是各个区域的位置变化
#目前能看到的就是,模型更加关注的是S位置,还有每一个shot结束之后的anchor token位置,这个是总体影响
#接下来可能还需要去做的是层分析,分析不同层关注的重点是什么,进而更加细粒度的比较
#这里主要是做层的关注度,对比的是32层,在不同位置下面关注的是
#所以说想要做的就是总共6条线x轴是32层,然后进行step平均,关注的是其他区域\answertoken\ex1\ex2\ex3\ex4\ex5\questiontoken的关注度进行比较,所以总共是8张图
def _plot_layer_analysis_line_graph(plot_data,title,xlabel,ylabel,filename):
    """
    绘制折线图并保存

    Args:
        plot_data (dict): 包含绘图数据的字典。
        键是图例标签 (如文件名)，值是 Y 轴数据 (32个层的值)
        title (str): 图表标题 (例如 "answer_token")
        xlabel (str): X 轴标签 (例如 "Layer Number")
        ylabel (str): Y 轴标签 (例如 "Average Value")
        filename (str): 保存的文件路径
    """
    figsize=(10,6) #折线图使用较宽的比例
    fig=plt.figure(figsize=figsize)

    gs=gridspec.GridSpec(1,1,left=0.1,right=0.9,top=0.9,bottom=0.1)
    ax=fig.add_subplot(gs[0])

    #循环绘制字典中的每一条线
    for label,y_values in plot_data.items():
        if len(y_values)!=32:
            raise ValueError(f"Expected 32 values for layer analysis, got {len(y_values)}")
        
        #这里硬编码为32来进行层分析(默认都是32层)
        x_axis = range(32)
        ax.plot(x_axis,y_values,label=label,marker='o',markersize=4,linewidth=2)

    #设置标签和标题
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_title(title,fontsize=24)

    #设置刻度
    ax.tick_params(axis='both',which='major',labelsize=14)
    ax.set_xticks(list(range(0,32,4)))#每隔4个层显示一个刻度

    #添加图例和网格
    ax.legend(fontsize=12)
    ax.grid(True,linestyle='--',alpha=0.6)

    #利用高dpi进行保存
    plt.savefig(filename,format='png',dpi=600,bbox_inches='tight',pad_inches=0.1, 
                facecolor='white', edgecolor='none', transparent=False)
    plt.close(fig)
    print(f"Saved figure: {filename}")

#其实我应该先是收集50个样本再进行分析的,但是在我电脑上依然跑不起来,所以我这里先对单个样本进行分析
def  generate_layer_analysis_plots(
    task: str,
    nshot: int,
    max_samples: int,
    output_dir: str
    ):
    """
    从多个样本的json文件中读取layer_anaylse数据
    先对每个样本的所有step计算平均值,然后对所有样本计算平均值,生成8个对比折线图
    args:
        task (str): 任务名称
        nshot (int): few-shot数量
        max_samples (int): 样本数量(从0到max_samples-1)
        output_dir (str): 输出目录
    """
    #设置字体
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
    except Exception as e:
        print(f"Times New Roman font not found. Using default font. Error: {e}")

    #定义要分析的8个字段
    #要进行比较的8张图
    #list(str)
    #这里直接硬编码了,后期可以从rollout_list取出字段
    token_fields=[
        "answer_token",
        "question_token",
        "example1_token",
        "example2_token",
        "example3_token",
        "example4_token",
        "example5_token",
        "other_token",
    ]

    #设置输出目录
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    #初始化大数据结构存储所有数据,创建dict
    all_plot_data={field:{} for field in token_fields}
    
    #对每个position进行处理 (0到nshot)
    print(f"Processing {nshot+1} positions with {max_samples} samples each...")
    for position in range(nshot + 1):
        print(f"\n--- Processing position {position} ---")
        
        #存储该position下所有样本的平均值
        #samples_averages[field] = list of sample averages, each sample average is a list of 32 values
        samples_averages = {field: [] for field in token_fields}
        
        #遍历所有样本
        successful_samples = 0
        for sample_idx in range(max_samples):
            json_path_str = f'params/rollout_params/{task}_{position}_{sample_idx}.json'
            
            try:
                print(f"  Loading sample {sample_idx}: {json_path_str}")
                data = load_json_or_jsonl(json_path_str)
                #开始提取参数值
                if isinstance(data, dict):
                    rollout_list = data.get('rollout_list', [])
                elif isinstance(data, list):
                    rollout_list = data
                else:
                    print(f"  Warning: Unexpected data format in {json_path_str}, skipping...")
                    continue

                if not rollout_list:
                    print(f"  Warning: rollout_list is empty in {json_path_str}, skipping...")
                    continue

                #处理每个step的数据,计算该样本的step平均值
                num_steps = len(rollout_list)
                #为该样本初始化一个累加器
                step_sums = {field: [0.0]*32 for field in token_fields}
                
                #遍历所有step
                for step_dict in rollout_list:
                    layer_anaylse = step_dict.get('layer_anaylse', [])
                    if not layer_anaylse or len(layer_anaylse) != 32:
                        print(f"  Warning: layer_anaylse_sum invalid in {json_path_str}, skipping this sample...")
                        break
                    
                    #累加每一层的值
                    for layer_index, layer_dict in enumerate(layer_anaylse):
                        for field in layer_dict.keys():
                            if field in step_sums:
                                step_sums[field][layer_index] += layer_dict[field]
                            else:
                                print(f"  Warning: Unexpected field {field} in {json_path_str}")
                else:
                    #只有当所有step都成功处理时才计算平均值(使用for-else语法)
                    #计算该样本的step平均值
                    for field in token_fields:
                        sample_average = [total_sum / num_steps for total_sum in step_sums[field]]
                        samples_averages[field].append(sample_average)
                    
                    successful_samples += 1
                
            except FileNotFoundError:
                print(f"  Warning: File not found {json_path_str}, skipping...")
                continue
            except Exception as e:
                print(f"  Warning: Error processing {json_path_str}: {e}, skipping...")
                continue
        
        if successful_samples == 0:
            raise ValueError(f"No valid samples found for position {position}")
        
        print(f"  Successfully processed {successful_samples}/{max_samples} samples for position {position}")
        
        #计算所有样本的平均值
        for field in token_fields:
            if len(samples_averages[field]) == 0:
                raise ValueError(f"No valid data for field {field} at position {position}")
            
            #将所有样本的32层数据进行平均
            #samples_averages[field] 是一个 list[list[float]], 每个子列表有32个元素
            num_valid_samples = len(samples_averages[field])
            final_average = [0.0] * 32
            
            for sample_average in samples_averages[field]:
                for layer_idx in range(32):
                    final_average[layer_idx] += sample_average[layer_idx]
            
            final_average = [val / num_valid_samples for val in final_average]
            
            #存储该position的最终平均值
            all_plot_data[field][position] = final_average
        
    print("\nGenerating 8 plots...")
    for field in token_fields:
        plot_data = all_plot_data[field]

        if not plot_data:
            raise ValueError(f"No data found for field: {field}")
        
        filename=output_dir / f"layer_analysis_{field}.png"

        _plot_layer_analysis_line_graph(
            plot_data=plot_data,
            title=f"Layer Analysis - {field} (avg over {max_samples} samples)",
            xlabel="Layer Number",
            ylabel="Average Contribution Value",
            filename=str(filename)
        )
    
    print(f"\nSuccessfully generated plots in {output_dir}")


def main(args):
    #作为借口进行扩展,因为之后一次就会把所有数据进行探索
    max_samples=args.max_samples  # 修改为max_samples,表示要处理的样本数量
    nshot=args.nshot
    task=args.task
    layer_analysis=args.layer_analysis#是否对attention进行层分析
    heatmap_generate=args.heatmap_generate#是否进行rollout生成
    basename=args.basename#输出文件的前缀
    output_dir=args.output_dir#输出目录
    json_file=args.json_file#输入的json文件路径
    
    if layer_analysis is True:
        my_output_directory=f'rollout_results/{task}'
        generate_layer_analysis_plots(
            task=task,
            nshot=nshot,
            max_samples=max_samples,
            output_dir=my_output_directory
        )
    
    if heatmap_generate is True:
        generate_rollout_heatmaps(
        json_file_path=json_file,
        output_dir=output_dir,
        basename=basename
        ) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate rollout heatmaps from JSON file')
    parser.add_argument('--json_file', type=str, default=None, help='Path to JSON file containing rollout data (required if heatmap_generate=True)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for heatmaps (default: same as JSON file)')
    parser.add_argument('--basename', type=str, default=None, help='Base name for output files (default: JSON file name without extension)')
    parser.add_argument('--layer_analysis', type=lambda x: x.lower() == 'true', default=False, help='Whether to perform layer analysis')
    parser.add_argument('--heatmap_generate', type=lambda x: x.lower() == 'true', default=False, help='Whether to generate rollout heatmaps')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum number of samples to process (from 0 to max_samples-1)')
    parser.add_argument('--nshot', type=int, default=5, help='Number of shots')
    parser.add_argument('--task', type=str, default='task', help='Task name')
    #这里后续还有一个参数,可选项,是可视化哪些图,不可视化哪些图
    args = parser.parse_args()
    # generate_rollout_heatmaps(
    #     json_file_path=args.json_file,
    #     output_dir=args.output_dir,
    #     basename=args.basename
    # )
    main(args)