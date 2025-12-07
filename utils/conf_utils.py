from matplotlib.pyplot import step
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os, sys, json
import numpy as np
from typing import List, Dict, Tuple
from transformers.models.electra.modeling_electra import ElectraSelfAttention
#加入煮目录,方便搜索到相应模块
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
#导入相关的库,封装成相关函数

#获取token_change的相关信息
def cal_token_change_positive_total(token_change:List[np.ndarray]):
    #计算token_change所有值为1的数量
    token_change_positive_total = 0
    for step_data in token_change:
        # 统一转换为一维数组
        if isinstance(step_data, torch.Tensor):
            arr = step_data.detach().cpu().numpy()
        else:
            arr = np.asarray(step_data)
        if arr.ndim > 1:
            arr = arr.flatten()
        for value in arr:
            if np.isfinite(value) and value > 0:
                token_change_positive_total += 1
    return token_change_positive_total


#计算conf在每一步的相关参数进行保存,在关键的位置上计算conf的值(注意利用之前的位置)
def cal_accmulate_conf(
    conf:List[np.ndarray],
    answer_token_positions:Dict,  # 一个字段包含各种范围,当然也有可能什么都找不到而报错
    conf_diff:List[np.ndarray]=None,
    entropy:List[np.ndarray]=None,
    token_change:List[np.ndarray]=None,
):
    """
    计算conf、conf_diff、entropy在三种范围的平均值
    
    Args:
        conf: 每一步的置信度列表，每个元素是[gen_length]的numpy数组
        answer_token_positions: 答案区域位置信息字典
        conf_diff: 每一步的置信度差值列表，每个元素是[gen_length]的numpy数组
        entropy: 每一步的熵值列表，每个元素是[gen_length]的numpy数组
        token_change: 每一步的token变化列表，每个元素是[gen_length]的numpy数组(每一步如果有1了就代表进行了相应改变)
    Returns:
        dict: 包含每个变量在三种范围的平均值和有效值数量
    """
    # 检查输入
    if conf is None or token_change is None:
        raise ValueError("conf and token_change cannot be None")
    if len(conf) != len(token_change):
        raise ValueError(f"Length mismatch - conf length: {len(conf)}, token_change length: {len(token_change)}")
    assert len(conf) ==len(token_change), "所有列表长度必须相同"
    #想统一接口方便写代码
    # 使用 .get() 方法安全地获取位置信息，如果键不存在则使用空列表作为默认值
    # digit_positions = answer_token_positions.get('digit_positions', [])
    # zero_mapped_positions = answer_token_positions.get('zero_mapped_positions', [])
    # other_digit_positions = answer_token_positions.get('other_digit_positions', [])
    # space_enter_positions = answer_token_positions.get('space_enter_positions', [])
    # answer_tag_positions = answer_token_positions.get('answer_tag_positions', [])
    other_positions = answer_token_positions.get('other_positions', [])
    answer_positions=answer_token_positions.get('answer_token_positions', [])
    # 获取所有位置（用于 'all' 范围）
    # 从第一个 step_data 获取长度来确定所有位置
    if len(conf) > 0 and len(conf[0]) > 0:
        all_positions = list(range(len(conf[0])))
    else:
        all_positions = []
    # 定义三种范围
    ranges = {
        # 'digit': digit_positions,#所有数字
        # 'zero_mapped': zero_mapped_positions,#0映射
        # 'other_digit': other_digit_positions,#其他数字
        # 'space_enter': space_enter_positions,#空格和换行符
        # 'answer_tag': answer_tag_positions,#答案标签
        'other_positions': other_positions,#其他位置
        'all': all_positions,  # 所有位置
        'answer_positions': answer_positions,#答案位置
    }
    
    # 存储结果
    result = {}
    
    #计算token_change所有值为1的数量
    token_change_positive_total = cal_token_change_positive_total(token_change)

    # 对每个变量进行计算
    variables = {
        'conf': conf,
        # 'conf_diff': conf_diff,
        # 'entropy': entropy,
        'token_change': token_change    #需要依次进行计算的量

    }
    #要计算的量
    for var_name, var_list in variables.items():
        result[var_name] = {}
        #要计算的范围
        for range_name, positions in ranges.items():
            # 存储每一步的平均值和有效值数量
            step_means = []
            step_counts = []
            token_cover=[]#这个是记录每一步所有答案token中有多少被覆盖了,之后再处理
            # token_positive_counts=[]#记录每一步中>0的数量（用于统计占比）

            for step_data in var_list:
                assert step_data.ndim==1,f"step_data的维度应该是1,当前维度是{step_data.ndim}"
                # 提取指定位置的值
                values = step_data[positions]
                # 过滤掉 -np.inf
                valid_values = values[values != -np.inf]
                
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    # 转换numpy标量为Python float
                    step_means.append(float(mean_val) if np.isfinite(mean_val) else np.nan)
                    step_counts.append(int(len(valid_values)))
                    if var_name=='token_change':
                        # 对于token_change，计算指定位置上大于0的值总和
                        positive_sum = np.sum(valid_values[valid_values > 0])
                        # positive_count = np.count_nonzero(valid_values > 0)
                        if positive_sum > 0:
                            # 如果总和大于0，计算总和，转换为Python float
                            token_cover.append(float(positive_sum))
                        else:
                            token_cover.append(0.0)
                        # token_positive_counts.append(positive_count)
                else:
                    step_means.append(np.nan)  # 如果没有有效值，记录为nan
                    step_counts.append(0)
                    if var_name=='token_change':
                        token_cover.append(0.0)
                        # token_positive_counts.append(0)
   
            # 计算所有步的平均值（只考虑有有效值的步）
            valid_step_means = [m for m in step_means if not np.isnan(m)]
            if len(valid_step_means) > 0:
                overall_mean_val = np.mean(valid_step_means)
                # 转换numpy标量为Python float
                overall_mean = float(overall_mean_val) if np.isfinite(overall_mean_val) else np.nan
                total_valid_count = int(sum(step_counts))
            else:
                overall_mean = np.nan
                total_valid_count = 0
            
            overall_token_cover = float(sum(token_cover))
            
            # 转换step_means列表中的numpy标量为Python float
            step_means_converted = [float(m) if np.isfinite(m) else np.nan for m in step_means]


            result[var_name][range_name] = {
                'overall_mean': overall_mean,  # 所有步的平均值
                'step_means': step_means_converted,  # 每一步的平均值（已转换为list）
                # 'step_counts': step_counts,  # 每一步的有效值数量
                'total_valid_count': total_valid_count  # 总有效值数量
            }
            if var_name=='token_change':
                result[var_name][range_name]['token_cover'] = token_cover  # 平均每一步的数量（已转换为list）
                result[var_name][range_name]['overall_token_cover'] = overall_token_cover  # 所有步的数量
                # ratio计算：如果token_cover是列表，需要逐元素计算
                if overall_token_cover > 0:
                    ratio_list = [float(tc / overall_token_cover) for tc in token_cover]
                else:
                    ratio_list = [0.0] * len(token_cover)
                result[var_name][range_name]['token_change_step_ratio'] = ratio_list  # 在所有步的数量占比（已转换为list）
                if token_change_positive_total > 0:
                    token_change_positive_total_ratio = float(overall_token_cover / token_change_positive_total)
                else:
                    token_change_positive_total_ratio = 0.0
                result[var_name][range_name]['token_change_position_ratio'] = token_change_positive_total_ratio  # 在部分位置中的占比/所有位置所有步数的占比
    return result 


#画图函数,用于将所有的数据进行画图
def paint_conf(conf,conf_diff,entropy,answer_token_positions,token_change,orders_result,task,position,index:int=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle, Patch
    import seaborn as sns

    # 检查输入
    assert isinstance(conf, list), "conf must be a list"
    # assert isinstance(conf_diff, list), "conf_diff must be a list"
    # assert isinstance(entropy, list), "entropy must be a list"
    assert isinstance(answer_token_positions, dict), "answer_token_positions must be a dict"
    assert isinstance(token_change, list), "token_change must be a list"
    assert len(conf)  == len(token_change), "所有列表长度必须相同"

    #获取位置信息，根据任务类型选择不同的位置
    space_enter_positions=answer_token_positions.get('space_enter_positions',[])
    
    if task == 'sudoku':
        # sudoku 任务：记录 space_enter_positions、zero_mapped_positions、other_digit_positions
        zero_mapped_positions=answer_token_positions.get('zero_mapped_positions',[])
        other_digit_positions=answer_token_positions.get('other_digit_positions',[])
        # 用于 countdown 的位置设为空
        digit_positions = []
        symbol_positions = []
    elif task == 'countdown':
        # countdown 任务：记录 space_enter_positions、digit_positions、symbol_positions
        digit_positions=answer_token_positions.get('digit_positions',[])
        symbol_positions=answer_token_positions.get('symbol_positions',[])
        # 用于 sudoku 的位置设为空
        zero_mapped_positions = []
        other_digit_positions = []
    else:
        # 其他任务：默认使用 sudoku 的格式
        zero_mapped_positions=answer_token_positions.get('zero_mapped_positions',[])
        other_digit_positions=answer_token_positions.get('other_digit_positions',[])
        digit_positions = []
        symbol_positions = []

    # 将列表转换为矩阵：steps x gen_length
    # 注意：需要将每个数组转换为numpy数组，并确保长度一致
    def list_to_matrix(data_list):
        """将 List 中的元素统一转换成 token × step 的矩阵"""
        arrays = []
        for item in data_list:
            if item is None:
                continue
            if isinstance(item, torch.Tensor):
                tensor = item.detach()
                if tensor.dim() == 2:
                    tensor = tensor[0]
                arr = tensor.detach().cpu().numpy()
            elif isinstance(item, np.ndarray):
                arr = item
            else:
                arr = np.asarray(item)
            arrays.append(arr.flatten() if arr.ndim > 1 else arr)

        if not arrays:
            raise ValueError("data_list 内没有可转换为矩阵的数据")

        max_len = max(arr.shape[0] if arr.ndim > 0 else 1 for arr in arrays)

        matrix = np.full((max_len, len(arrays)), np.nan)
        for idx, arr in enumerate(arrays):
            if arr.ndim == 0:
                matrix[0, idx] = arr
            else:
                arr_flat = arr.flatten()
                matrix[:len(arr_flat), idx] = arr_flat

        return matrix   
    
    conf_matrix=list_to_matrix(conf)
    conf_diff_matrix=list_to_matrix(conf_diff)
    entropy_matrix=list_to_matrix(entropy)
    token_change_matrix=list_to_matrix(token_change)
    
    # 处理orders_result，转换为矩阵，标记解码位置
    orders_matrix = list_to_matrix(orders_result)  # shape: (gen_length, num_steps)
    # orders_matrix中值为1的位置表示该token在该步被解码

    #获取矩阵维度(统一维度)
    gen_length, num_steps = conf_matrix.shape

    #创建4个字图
    fig=plt.figure(figsize=(20,16))
    gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.3,wspace=0.3)

    #定义要绘制的数据
    data_dict={
        'conf':conf_matrix,
        'conf_diff':conf_diff_matrix,
        'entropy':entropy_matrix,
        'token_change':token_change_matrix
    }

    # 定义位置范围的颜色（根据任务类型）
    if task == 'sudoku':
        position_colors = {
            'space_enter': 'red',      # 红色
            'zero_mapped': 'green',     # 绿色
            'other_digit': 'blue',      # 蓝色
        }
        # 创建位置到类型的映射,统一到一个变量中
        pos_to_type = {}
        for pos in space_enter_positions:
            pos_to_type[pos] = 'space_enter'
        for pos in zero_mapped_positions:
            pos_to_type[pos] = 'zero_mapped'
        for pos in other_digit_positions:
            pos_to_type[pos] = 'other_digit'
    elif task == 'countdown':
        position_colors = {
            'space_enter': 'red',      # 红色
            'digit': 'green',          # 绿色
            'symbol': 'blue',          # 蓝色
        }
        # 创建位置到类型的映射,统一到一个变量中
        pos_to_type = {}
        for pos in space_enter_positions:
            pos_to_type[pos] = 'space_enter'
        for pos in digit_positions:
            pos_to_type[pos] = 'digit'
        for pos in symbol_positions:
            pos_to_type[pos] = 'symbol'
    else:
        # 默认使用 sudoku 的格式
        position_colors = {
            'space_enter': 'red',      # 红色
            'zero_mapped': 'green',     # 绿色
            'other_digit': 'blue',      # 蓝色
        }
        pos_to_type = {}
        for pos in space_enter_positions:
            pos_to_type[pos] = 'space_enter'
        for pos in zero_mapped_positions:
            pos_to_type[pos] = 'zero_mapped'
        for pos in other_digit_positions:
            pos_to_type[pos] = 'other_digit'    
    
    #绘制每个指标
    for idx,(name,data) in enumerate(data_dict.items()):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        #处理-np.inf(我的代码里面没有inf)
        data_display = data.copy()
        # 将 -np.inf 替换为一个很小的值，np.inf 替换为一个np.nan
        data_display[data_display == -np.inf] = np.nan    
        
        # 计算vmin和vmax（排除nan和inf）
        valid_data = data_display[~np.isnan(data_display) & ~np.isinf(data_display)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 1)
            vmax = np.percentile(valid_data, 99)
        else:
            vmin, vmax = 0, 1
        
        # 绘制热力图
        im = ax.imshow(data_display, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        
        # 如果是token_change子图，添加网格和解码顺序标记
        if name == 'token_change':
            # 添加网格
            ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, num_steps, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, gen_length, 1), minor=True)
            ax.grid(True, which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 在解码位置添加鲜艳的标记点
            # 找出每个step新解码的token位置（当前步为1，前一步为0或不存在）
            # 计算每个网格单元的大小（用于绘制小方块）
            # 由于imshow的extent是[0, num_steps] x [0, gen_length]，每个单元的大小是1x1
            # 绘制小方块，大小为网格单元的0.7倍（更大更明显）
            box_size = 0.7
            # 使用亮绿色，在深色背景上对比度很高
            decode_color = '#00FF00'  # 亮绿色，RGB: (0, 255, 0)
            
            # 找出每个step新解码的token
            for step in range(num_steps):
                if step == 0:
                    # 第一步：所有为1的位置都是新解码的
                    new_decoded = np.where(orders_matrix[:, step] == 1)[0]
                else:
                    # 后续步：当前步为1且前一步为0的位置是新解码的
                    current_step = orders_matrix[:, step] == 1
                    prev_step = orders_matrix[:, step-1] == 0
                    new_decoded = np.where(current_step & prev_step)[0]
                
                # 为每个新解码的token位置绘制鲜艳的标记点
                for token_pos in new_decoded:
                    # 计算小方块的中心位置（在网格单元的中心）
                    x_center = step
                    y_center = token_pos
                    
                    # 绘制亮绿色小方块（完全不透明，对比度高，位于最顶层）
                    rect = Rectangle(
                        (x_center - box_size/2, y_center - box_size/2),
                        box_size, box_size,
                        facecolor=decode_color, edgecolor='black', linewidth=1.0, alpha=1.0, zorder=10
                    )
                    ax.add_patch(rect)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(name, fontsize=12)
        
        # 设置坐标轴
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Token Position', fontsize=12)
        ax.set_title(f'{name.upper()}', fontsize=14, fontweight='bold')
        
        # 设置x轴刻度（steps）
        ax.set_xticks(range(num_steps))
        ax.set_xticklabels(range(num_steps))
        
        # 设置y轴刻度（token positions），并根据位置类型设置颜色
        ax.set_yticks(range(gen_length))
        yticklabels = []
        ytick_colors = []

        for pos in range(gen_length):
            yticklabels.append(str(pos))
            if pos in pos_to_type:
                color = position_colors[pos_to_type[pos]]
                ytick_colors.append(color)
            else:
                ytick_colors.append('black')  # 默认黑色
        
        ax.set_yticklabels(yticklabels)
        
        # 设置Y轴刻度标签的颜色
        for ticklabel, color in zip(ax.get_yticklabels(), ytick_colors):
            ticklabel.set_color(color)
            ticklabel.set_fontweight('bold')  # 加粗以更明显
    
    # 添加图例（根据任务类型）
    if task == 'sudoku':
        legend_elements = [
            Patch(facecolor='red', edgecolor='red', label='Space/Enter'),
            Patch(facecolor='green', edgecolor='green', label='Zero Mapped'),
            Patch(facecolor='blue', edgecolor='blue', label='Other Digit'),
            Patch(facecolor='#00FF00', edgecolor='black', label='Decoded Token (in TOKEN_CHANGE)')
        ]
    elif task == 'countdown':
        legend_elements = [
            Patch(facecolor='red', edgecolor='red', label='Space/Enter'),
            Patch(facecolor='green', edgecolor='green', label='Digit'),
            Patch(facecolor='blue', edgecolor='blue', label='Symbol'),
            Patch(facecolor='#00FF00', edgecolor='black', label='Decoded Token (in TOKEN_CHANGE)')
        ]
    else:
        # 默认使用 sudoku 的格式
        legend_elements = [
            Patch(facecolor='red', edgecolor='red', label='Space/Enter'),
            Patch(facecolor='green', edgecolor='green', label='Zero Mapped'),
            Patch(facecolor='blue', edgecolor='blue', label='Other Digit'),
            Patch(facecolor='#00FF00', edgecolor='black', label='Decoded Token (in TOKEN_CHANGE)')
        ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    # 保存图片
    output_dir = os.path.join(project_root, 'conf_results', 'conf_plots', task)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'conf_plot_task_{task}_index_{index}_position_{position}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'已保存图片到: {output_file}')
    plt.close()

#针对countdown进行适配,主要是字段不太一样
def cal_accmulate_conf_countdown(
    conf:List[np.ndarray],
    answer_token_positions:Dict,
    conf_diff:List[np.ndarray]=None,
    entropy:List[np.ndarray]=None,
    token_change:List[np.ndarray]=None,
    current_conf:np.ndarray=None,
):
    """
    计算conf、conf_diff、entropy在三种范围的平均值
    """
    # 检查输入,确认字段在对应任务中是存在的
    if conf is None or token_change is None:
        raise ValueError("conf and token_change cannot be None")
    if len(conf) != len(token_change):
        raise ValueError(f"Length mismatch - conf length: {len(conf)}, token_change length: {len(token_change)}")
    assert len(conf)==len(token_change), "所有列表长度必须相同"
    # assert 'digit_positions' in answer_token_positions, "answer_token_positions必须包含digit_positions"
    # assert 'symbol_positions' in answer_token_positions, "answer_token_positions必须包含symbol_positions"
    # assert 'space_enter_positions' in answer_token_positions, "answer_token_positions必须包含space_enter_positions"
    # assert 'answer_tag_positions' in answer_token_positions, "answer_token_positions必须包含answer_tag_positions"
    assert 'other_positions' in answer_token_positions, "answer_token_positions必须包含other_positions"

    # 使用 .get() 方法安全地获取位置信息，如果键不存在则使用空列表作为默认值
    # digit_positions = answer_token_positions.get('digit_positions', [])
    # symbol_positions = answer_token_positions.get('symbol_positions', [])
    # space_enter_positions = answer_token_positions.get('space_enter_positions', [])
    # answer_tag_positions = answer_token_positions.get('answer_tag_positions', [])
    other_positions = answer_token_positions.get('other_positions', [])
    # digit_symbol_positions = digit_positions + symbol_positions
    answer_positions = answer_token_positions.get('answer_token_positions', [])
    # 获取所有位置（用于 'all' 范围）
    if len(conf) > 0 and len(conf[0]) > 0:
        all_positions = list(range(len(conf[0])))
    else:
        all_positions = []
    #定义三种范围
    ranges={
        # 'digit_symbol': digit_symbol_positions,
        # 'digit': digit_positions,
        # 'symbol': symbol_positions,
        # 'space_enter': space_enter_positions,
        # 'answer_tag': answer_tag_positions,
        # 'answer_tag_positions': answer_tag_positions,
        'answer_positions': answer_positions,
        'other_positions': other_positions,
        'all': all_positions,  # 所有位置
    }

    result={}

    #计算token_change所有值为1的数量
    token_change_positive_total = cal_token_change_positive_total(token_change)

    #对每个变量进行计算
    if current_conf is not None:
    variables={
        'conf': conf,
            # 'conf_diff': conf_diff,
            # 'entropy': entropy,
            'token_change': token_change,
            'current_conf':current_conf,#当前的conf,我想看的是是不是有相关的指标可以进行记录(就是呈现正比),这个应该适合conf_diff是相关的(但是conf_diff作为解码策略他们已经用了,感觉累加的是比较适合说理的)
        }
    else:
        variables={
            'conf': conf,
            # 'conf_diff': conf_diff,
            # 'entropy': entropy,
        'token_change': token_change,
    }
    #要计算的量
    for var_name, var_list in variables.items():
        result[var_name]={}
        
        # 特殊处理 current_conf：它是一个单独的 numpy 数组，不是列表
        if var_name == 'current_conf':
            # current_conf 是一个长度为 gen_length 的 numpy 数组
            if current_conf is not None:
                # 确保是一维数组
                if current_conf.ndim > 1:
                    current_conf = current_conf.flatten()
                assert current_conf.ndim == 1, f"current_conf的维度应该是1,当前维度是{current_conf.ndim}"
                
                for range_name, positions in ranges.items():
                    # 提取指定位置的值
                    values = current_conf[positions]
                    # 过滤掉 -np.inf 和 nan
                    valid_mask = (values != -np.inf) & np.isfinite(values)
                    valid_values = values[valid_mask]
                    
                    if len(valid_values) > 0:
                        # 计算总和
                        value_sum = float(np.sum(valid_values))
                        total_valid_count = int(len(valid_values))
                    else:
                        value_sum = 0.0
                        total_valid_count = 0
                    
                    result[var_name][range_name] = {
                        'value_sum': value_sum,  # 在指定位置范围内的值的总和
                        'total_valid_count': total_valid_count  # 有效值的个数
                    }
            continue  # 处理完 current_conf 后继续下一个变量
        
        # 处理其他变量（conf, token_change等，它们是列表）
        for range_name, positions in ranges.items():
            #存储每一步的平均值和有效值数量
            step_means = []
            step_counts = []
            token_cover=[]#这个是记录每一步所有答案token中有多少被覆盖了,之后再处理
            # token_positive_counts=[]#记录每一步中>0的数量（用于统计占比）

            for step_data in var_list:
                assert step_data.ndim==1,f"step_data的维度应该是1,当前维度是{step_data.ndim}"
                 # 提取指定位置的值
                values = step_data[positions]
                # 过滤掉 -np.inf
                valid_values = values[values != -np.inf]
                
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    # 转换numpy标量为Python float
                    step_means.append(float(mean_val) if np.isfinite(mean_val) else np.nan)
                    step_counts.append(int(len(valid_values)))
                    if var_name=='token_change':
                        # 对于token_change，计算指定位置上大于0的值总和
                        positive_sum = np.sum(valid_values[valid_values > 0])
                        # positive_count = np.count_nonzero(valid_values > 0)
                        if positive_sum > 0:
                            # 如果总和大于0，计算总和，转换为Python float
                            token_cover.append(float(positive_sum))
                        else:
                            token_cover.append(0.0)
                        # token_positive_counts.append(positive_count)
                else:
                    step_means.append(np.nan)  # 如果没有有效值，记录为nan
                    step_counts.append(0)
                    if var_name=='token_change':
                        token_cover.append(0.0)
                        # token_positive_counts.append(0)  

            # 计算所有步的平均值（只考虑有有效值的步）
            valid_step_means = [m for m in step_means if not np.isnan(m)]
            if len(valid_step_means) > 0:
                overall_mean_val = np.mean(valid_step_means)
                # 转换numpy标量为Python float
                overall_mean = float(overall_mean_val) if np.isfinite(overall_mean_val) else np.nan
                total_valid_count = int(sum(step_counts))
            else:
                overall_mean = np.nan
                total_valid_count = 0  

            overall_token_cover = float(sum(token_cover))

             # 转换step_means列表中的numpy标量为Python float
            step_means_converted = [float(m) if np.isfinite(m) else np.nan for m in step_means]


            result[var_name][range_name] = {
                'overall_mean': overall_mean,  # 所有步的平均值
                'step_means': step_means_converted,  # 每一步的平均值（已转换为list）
                # 'step_counts': step_counts,  # 每一步的有效值数量
                'total_valid_count': total_valid_count  # 总有效值数量
            }
            if var_name=='token_change':
                result[var_name][range_name]['token_cover'] = token_cover  # 平均每一步的数量（已转换为list）
                result[var_name][range_name]['overall_token_cover'] = overall_token_cover  # 所有步的数量
                # ratio计算：如果token_cover是列表，需要逐元素计算
                if overall_token_cover > 0:
                    ratio_list = [float(tc / overall_token_cover) for tc in token_cover]
                else:
                    ratio_list = [0.0] * len(token_cover)
                result[var_name][range_name]['token_change_step_ratio'] = ratio_list  # 在所有步的数量占比（已转换为list）
                if token_change_positive_total > 0:
                    token_change_positive_total_ratio = float(overall_token_cover / token_change_positive_total)
                else:
                    token_change_positive_total_ratio = 0.0
                result[var_name][range_name]['token_change_position_ratio'] = token_change_positive_total_ratio  # 在部分位置中的占比/所有位置所有步数的占比
    return result

#针对gsm8k进行适配,主要是字段不太一样
#我觉得后面也可以加一个和task相关的参数,进行适配
def cal_accmulate_conf_gsm8k(
    conf:List[np.ndarray],
    answer_token_positions:Dict,#主要是这里的字段确实是不太一样的
    conf_diff:List[np.ndarray]=None,
    entropy:List[np.ndarray]=None,
    token_change:List[np.ndarray]=None,
    current_conf:np.ndarray=None,
):
    """
    计算conf、conf_diff、entropy在三种范围的平均值
    """
    # 检查输入,确认字段在对应任务中是存在的
    if conf is None or token_change is None:
        raise ValueError("conf and token_change cannot be None")
    if len(conf) != len(token_change):
        raise ValueError(f"Length mismatch - conf length: {len(conf)}, token_change length: {len(token_change)}")
    assert len(conf) ==len(token_change), "所有列表长度必须相同"
    # assert 'digit_positions' in answer_token_positions, "answer_token_positions必须包含digit_positions"
    assert 'answer_token_positions' in answer_token_positions, "answer_token_positions必须包含answer_token_positions"
    # assert 'struct_answer_token_positions' in answer_token_positions, "answer_token_positions必须包含struct_answer_token_positions"
    assert 'other_token_positions' in answer_token_positions, "answer_token_positions必须包含other_token_positions"

    # 使用 .get() 方法安全地获取位置信息，如果键不存在则使用空列表作为默认值
    #一个是只有答案,一个是还包含了一部分的结构性答案
    answer_positions = answer_token_positions.get('answer_token_positions', [])
    # struct_answer_positions = answer_token_positions.get('struct_answer_token_positions', [])
    other_positions = answer_token_positions.get('other_token_positions', [])
    # 获取所有位置（用于 'all' 范围）
    if len(conf) > 0 and len(conf[0]) > 0:
        all_positions = list(range(len(conf[0])))
    else:
        all_positions = []
    #定义三种范围
    ranges={
        'answer_positions': answer_positions,#答案性
        # 'answer_struct_positions': struct_answer_positions,#包含结构性语句
        'other_positions': other_positions,#其他位置
        'all': all_positions,  # 所有位置
    }

    result={}

    #计算token_change所有值为1的数量
    token_change_positive_total = cal_token_change_positive_total(token_change)

    #对每个变量进行计算
    if current_conf is not None:
    variables={
        'conf': conf,
            # 'conf_diff': conf_diff,
            # 'entropy': entropy,
            'token_change': token_change,
            'current_conf':current_conf,#当前的conf,我想看的是是不是有相关的指标可以进行记录(就是呈现正比),这个应该适合conf_diff是相关的(但是conf_diff作为解码策略他们已经用了,感觉累加的是比较适合说理的)
        }
    else:
        variables={
            'conf': conf,
            # 'conf_diff': conf_diff,
            # 'entropy': entropy,
        'token_change': token_change,
    }
    #要计算的量
    for var_name, var_list in variables.items():
        result[var_name]={}
        
        # 特殊处理 current_conf：它是一个单独的 numpy 数组，不是列表
        if var_name == 'current_conf':
            # current_conf 是一个长度为 gen_length 的 numpy 数组
            if current_conf is not None:
                # 确保是一维数组
                if current_conf.ndim > 1:
                    current_conf = current_conf.flatten()
                assert current_conf.ndim == 1, f"current_conf的维度应该是1,当前维度是{current_conf.ndim}"
                
                for range_name, positions in ranges.items():
                    # 提取指定位置的值
                    values = current_conf[positions]
                    # 过滤掉 -np.inf 和 nan
                    valid_mask = (values != -np.inf) & np.isfinite(values)
                    valid_values = values[valid_mask]
                    
                    if len(valid_values) > 0:
                        # 计算总和
                        value_sum = float(np.sum(valid_values))
                        total_valid_count = int(len(valid_values))
                    else:
                        value_sum = 0.0
                        total_valid_count = 0
                    
                    result[var_name][range_name] = {
                        'value_sum': value_sum,  # 在指定位置范围内的值的总和
                        'total_valid_count': total_valid_count  # 有效值的个数
                    }
            continue  # 处理完 current_conf 后继续下一个变量
        
        # 处理其他变量（conf, token_change等，它们是列表）
        for range_name, positions in ranges.items():
            #存储每一步的平均值和有效值数量
            step_means = []
            step_counts = []
            token_cover=[]#这个是记录每一步所有答案token中有多少被覆盖了,之后再处理
            # token_positive_counts=[]#记录每一步中>0的数量（用于统计占比）

            for step_data in var_list:
                assert step_data.ndim==1,f"step_data的维度应该是1,当前维度是{step_data.ndim}"
                 # 提取指定位置的值
                values = step_data[positions]
                # 过滤掉 -np.inf
                valid_values = values[values != -np.inf]
                
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    # 转换numpy标量为Python float
                    step_means.append(float(mean_val) if np.isfinite(mean_val) else np.nan)
                    step_counts.append(int(len(valid_values)))
                    if var_name=='token_change':
                        # 对于token_change，计算指定位置上大于0的值总和
                        positive_sum = np.sum(valid_values[valid_values > 0])
                        # positive_count = np.count_nonzero(valid_values > 0)
                        if positive_sum > 0:
                            # 如果总和大于0，计算总和，转换为Python float
                            token_cover.append(float(positive_sum))
                        else:
                            token_cover.append(0.0)
                        # token_positive_counts.append(positive_count)
                else:
                    step_means.append(np.nan)  # 如果没有有效值，记录为nan
                    step_counts.append(0)
                    if var_name=='token_change':
                        token_cover.append(0.0)
                        # token_positive_counts.append(0)  

            # 计算所有步的平均值（只考虑有有效值的步）
            valid_step_means = [m for m in step_means if not np.isnan(m)]
            if len(valid_step_means) > 0:
                overall_mean_val = np.mean(valid_step_means)
                # 转换numpy标量为Python float
                overall_mean = float(overall_mean_val) if np.isfinite(overall_mean_val) else np.nan
                total_valid_count = int(sum(step_counts))
            else:
                overall_mean = np.nan
                total_valid_count = 0  

            overall_token_cover = float(sum(token_cover))

             # 转换step_means列表中的numpy标量为Python float
            step_means_converted = [float(m) if np.isfinite(m) else np.nan for m in step_means]


            result[var_name][range_name] = {
                'overall_mean': overall_mean,  # 所有步的平均值
                'step_means': step_means_converted,  # 每一步的平均值（已转换为list）
                # 'step_counts': step_counts,  # 每一步的有效值数量
                'total_valid_count': total_valid_count  # 总有效值数量
            }
            if var_name=='token_change':
                result[var_name][range_name]['token_cover'] = token_cover  # 平均每一步的数量（已转换为list）
                result[var_name][range_name]['overall_token_cover'] = overall_token_cover  # 所有步的数量
                # ratio计算：如果token_cover是列表，需要逐元素计算
                if overall_token_cover > 0:
                    ratio_list = [float(tc / overall_token_cover) for tc in token_cover]
                else:
                    ratio_list = [0.0] * len(token_cover)
                result[var_name][range_name]['token_change_step_ratio'] = ratio_list  # 在所有步的数量占比（已转换为list）
                if token_change_positive_total > 0:
                    token_change_positive_total_ratio = float(overall_token_cover / token_change_positive_total)
                else:
                    token_change_positive_total_ratio = 0.0
                result[var_name][range_name]['token_change_position_ratio'] = token_change_positive_total_ratio  # 在部分位置中的占比/所有位置所有步数的占比
    return result

#针对gsm8k进行适配,主要是字段不太一样
#我觉得后面也可以加一个和task相关的参数,进行适配
#其他的我并没进行计算,因为先看看效果(主要是想找拟合的情况)
def cal_accmulate_conf_math(
    conf:List[np.ndarray],
    answer_token_positions:Dict,#主要是这里的字段确实是不太一样的
    conf_diff:List[np.ndarray]=None,
    entropy:List[np.ndarray]=None,
    token_change:List[np.ndarray]=None,
    current_conf:np.ndarray=None,
):
    """
    计算conf、conf_diff、entropy在三种范围的平均值
    """
    # 检查输入,确认字段在对应任务中是存在的
    if conf is None or token_change is None:
        raise ValueError("conf and token_change cannot be None")
    if len(conf) != len(token_change):
        raise ValueError(f"Length mismatch - conf length: {len(conf)}, token_change length: {len(token_change)}")
    assert len(conf) == len(token_change), "所有列表长度必须相同"
    # assert 'digit_positions' in answer_token_positions, "answer_token_positions必须包含digit_positions"
    assert 'answer_token_positions' in answer_token_positions, "answer_token_positions必须包含answer_token_positions"
    assert 'other_token_positions' in answer_token_positions, "answer_token_positions必须包含other_token_positions"

    # 使用 .get() 方法安全地获取位置信息，如果键不存在则使用空列表作为默认值
    #一个是包含boxed的答案(我主要是区分不出来),另一个则是其他位置,关注的区域也就是包含boxed的位置即可
    answer_positions = answer_token_positions.get('answer_token_positions', [])
    # struct_answer_positions = answer_token_positions.get('struct_answer_token_positions', [])
    other_positions = answer_token_positions.get('other_token_positions', [])
    # 获取所有位置（用于 'all' 范围）
    if len(conf) > 0 and len(conf[0]) > 0:
        all_positions = list(range(len(conf[0])))
    else:
        all_positions = []
    #定义三种范围
    ranges={
        'answer_positions': answer_positions,
        # 'answer_struct_positions': struct_answer_positions,不关注结构性语句
        'other_positions': other_positions,#包含其他位置
        'all': all_positions,  # 所有位置
    }

    result={}

    #计算token_change所有值为1的数量
    token_change_positive_total = cal_token_change_positive_total(token_change)
    if current_conf is not None:
    variables={
        'conf': conf,
            'token_change': token_change,
            'current_conf':current_conf,#当前的conf,我想看的是是不是有相关的指标可以进行记录(就是呈现正比),这个应该适合conf_diff是相关的(但是conf_diff作为解码策略他们已经用了,感觉累加的是比较适合说理的)
        }
    else:
        variables={
            'conf': conf,
        'token_change': token_change,
    }
    #要计算的量
    for var_name, var_list in variables.items():
        result[var_name]={}
        
        # 特殊处理 current_conf：它是一个单独的 numpy 数组，不是列表
        if var_name == 'current_conf':
            # current_conf 是一个长度为 gen_length 的 numpy 数组
            if current_conf is not None:
                # 确保是一维数组
                if current_conf.ndim > 1:
                    current_conf = current_conf.flatten()
                assert current_conf.ndim == 1, f"current_conf的维度应该是1,当前维度是{current_conf.ndim}"
                
                for range_name, positions in ranges.items():
                    # 提取指定位置的值
                    values = current_conf[positions]
                    # 过滤掉 -np.inf 和 nan
                    valid_mask = (values != -np.inf) & np.isfinite(values)
                    valid_values = values[valid_mask]
                    
                    if len(valid_values) > 0:
                        # 计算总和
                        value_sum = float(np.sum(valid_values))
                        total_valid_count = int(len(valid_values))
                    else:
                        value_sum = 0.0
                        total_valid_count = 0
                    
                    result[var_name][range_name] = {
                        'value_sum': value_sum,  # 在指定位置范围内的值的总和
                        'total_valid_count': total_valid_count  # 有效值的个数
                    }
            continue  # 处理完 current_conf 后继续下一个变量
        
        # 处理其他变量（conf, token_change等，它们是列表）
        for range_name, positions in ranges.items():
            #存储每一步的平均值和有效值数量
            step_means = []
            step_counts = []
            token_cover=[]#这个是记录每一步所有答案token中有多少被覆盖了,之后再处理
            # token_positive_counts=[]#记录每一步中>0的数量（用于统计占比）

            for step_data in var_list:
                assert step_data.ndim==1,f"step_data的维度应该是1,当前维度是{step_data.ndim}"
                 # 提取指定位置的值
                values = step_data[positions]
                # 过滤掉 -np.inf
                valid_values = values[values != -np.inf]
                
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    # 转换numpy标量为Python float
                    step_means.append(float(mean_val) if np.isfinite(mean_val) else np.nan)
                    step_counts.append(int(len(valid_values)))
                    if var_name=='token_change':
                        # 对于token_change，计算指定位置上大于0的值总和
                        positive_sum = np.sum(valid_values[valid_values > 0])
                        # positive_count = np.count_nonzero(valid_values > 0)
                        if positive_sum > 0:
                            # 如果总和大于0，计算总和，转换为Python float
                            token_cover.append(float(positive_sum))
                        else:
                            token_cover.append(0.0)
                        # token_positive_counts.append(positive_count)
                else:
                    step_means.append(np.nan)  # 如果没有有效值，记录为nan
                    step_counts.append(0)
                    if var_name=='token_change':
                        token_cover.append(0.0)
                        # token_positive_counts.append(0)  

            # 计算所有步的平均值（只考虑有有效值的步）
            valid_step_means = [m for m in step_means if not np.isnan(m)]
            if len(valid_step_means) > 0:
                overall_mean_val = np.mean(valid_step_means)
                # 转换numpy标量为Python float
                overall_mean = float(overall_mean_val) if np.isfinite(overall_mean_val) else np.nan
                total_valid_count = int(sum(step_counts))
            else:
                overall_mean = np.nan
                total_valid_count = 0  

            overall_token_cover = float(sum(token_cover))

             # 转换step_means列表中的numpy标量为Python float
            step_means_converted = [float(m) if np.isfinite(m) else np.nan for m in step_means]


            result[var_name][range_name] = {
                'overall_mean': overall_mean,  # 所有步的平均值
                'step_means': step_means_converted,  # 每一步的平均值（已转换为list）
                # 'step_counts': step_counts,  # 每一步的有效值数量
                'total_valid_count': total_valid_count  # 总有效值数量
            }
            if var_name=='token_change':
                result[var_name][range_name]['token_cover'] = token_cover  # 平均每一步的数量（已转换为list）
                result[var_name][range_name]['overall_token_cover'] = overall_token_cover  # 所有步的数量
                # ratio计算：如果token_cover是列表，需要逐元素计算
                if overall_token_cover > 0:
                    ratio_list = [float(tc / overall_token_cover) for tc in token_cover]
                else:
                    ratio_list = [0.0] * len(token_cover)
                result[var_name][range_name]['token_change_step_ratio'] = ratio_list  # 在所有步的数量占比（已转换为list）
                if token_change_positive_total > 0:
                    token_change_positive_total_ratio = float(overall_token_cover / token_change_positive_total)
                else:
                    token_change_positive_total_ratio = 0.0
                result[var_name][range_name]['token_change_position_ratio'] = token_change_positive_total_ratio  # 在部分位置中的占比/所有位置所有步数的占比
    return result

#这里多考虑的当时解码的情况(不过感觉作用稍微小一点,因为都已经是解码过后了)
def cal_accmulate_conf_mbpp(
    conf:List[np.ndarray],
    answer_token_positions:Dict,#主要是这里的字段确实是不太一样的
    conf_diff:List[np.ndarray]=None,
    entropy:List[np.ndarray]=None,
    token_change:List[np.ndarray]=None,
    current_conf:np.ndarray=None,
):
    """
    计算conf、conf_diff、entropy在三种范围的平均值
    """
    # 检查输入,确认字段在对应任务中是存在的
    assert len(conf) == len(token_change), "所有列表长度必须相同"
    assert 'answer_token_positions' in answer_token_positions, "answer_token_positions必须包含answer_token_positions"
    assert 'other_token_positions' in answer_token_positions, "answer_token_positions必须包含other_token_positions"

    # 使用 .get() 方法安全地获取位置信息，如果键不存在则使用空列表作为默认值
    answer_positions = answer_token_positions.get('answer_token_positions', [])
    other_positions = answer_token_positions.get('other_token_positions', [])

    if len(conf) > 0 and len(conf[0]) > 0:
        all_positions = list(range(len(conf[0])))
    else:
        all_positions = []

    #定义三种范围
    ranges={
        #计算需要关注的东西,最后计算的时候可以不用太考虑
        'answer_positions': answer_positions,
        'other_positions': other_positions,
        'all': all_positions,
    }
    result={}
    token_change_positive_total = cal_token_change_positive_total(token_change)
    if current_conf is not None:
        variables={
            'conf': conf,
            'token_change': token_change,
            'current_conf':current_conf,#当前的conf,我想看的是是不是有相关的指标可以进行记录(就是呈现正比),这个应该适合conf_diff是相关的(但是conf_diff作为解码策略他们已经用了,感觉累加的是比较适合说理的)
        }
    else:
        variables={
            'conf': conf,
            'token_change': token_change,
        }
    #要计算的量
    for var_name, var_list in variables.items():
        result[var_name]={}
        
        # 特殊处理 current_conf：它是一个单独的 numpy 数组，不是列表
        if var_name == 'current_conf':
            # current_conf 是一个长度为 gen_length 的 numpy 数组
            if current_conf is not None:
                # 确保是一维数组
                if current_conf.ndim > 1:
                    current_conf = current_conf.flatten()
                assert current_conf.ndim == 1, f"current_conf的维度应该是1,当前维度是{current_conf.ndim}"
                
                for range_name, positions in ranges.items():
                    # 提取指定位置的值
                    values = current_conf[positions]
                    # 过滤掉 -np.inf 和 nan
                    valid_mask = (values != -np.inf) & np.isfinite(values)
                    valid_values = values[valid_mask]
                    
                    if len(valid_values) > 0:
                        # 计算总和
                        value_sum = float(np.sum(valid_values))
                        total_valid_count = int(len(valid_values))
                    else:
                        value_sum = 0.0
                        total_valid_count = 0
                    
                    result[var_name][range_name] = {
                        'value_sum': value_sum,  # 在指定位置范围内的值的总和
                        'total_valid_count': total_valid_count  # 有效值的个数
                    }
            continue  # 处理完 current_conf 后继续下一个变量
        
        # 处理其他变量（conf, token_change等，它们是列表）
        for range_name, positions in ranges.items():
            #存储每一步的平均值和有效值数量
            step_means = []
            step_counts = []
            token_cover=[]#这个是记录每一步所有答案token中有多少被覆盖了,之后再处理
            # token_positive_counts=[]#记录每一步中>0的数量（用于统计占比）
            for step_data in var_list:
                assert step_data.ndim==1,f"step_data的维度应该是1,当前维度是{step_data.ndim}"
                 # 提取指定位置的值
                values = step_data[positions]
                # 过滤掉 -np.inf
                valid_values = values[values != -np.inf]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    # 转换numpy标量为Python float
                    step_means.append(float(mean_val) if np.isfinite(mean_val) else np.nan)
                    step_counts.append(int(len(valid_values)))
                    if var_name=='token_change':
                        # 对于token_change，计算指定位置上大于0的值总和
                        positive_sum = np.sum(valid_values[valid_values > 0])
                        # positive_count = np.count_nonzero(valid_values > 0)
                        if positive_sum > 0:
                            # 如果总和大于0，计算总和，转换为Python float
                            token_cover.append(float(positive_sum))
                        else:
                            token_cover.append(0.0)
                        # token_positive_counts.append(positive_count)
                else:
                    step_means.append(np.nan)  # 如果没有有效值，记录为nan
                    step_counts.append(0)
                    if var_name=='token_change':
                        token_cover.append(0.0)

            # 计算所有步的平均值（只考虑有有效值的步）
            valid_step_means = [m for m in step_means if not np.isnan(m)]
            if len(valid_step_means) > 0:
                overall_mean_val = np.mean(valid_step_means)
                # 转换numpy标量为Python float
                overall_mean = float(overall_mean_val) if np.isfinite(overall_mean_val) else np.nan
                total_valid_count = int(sum(step_counts))
            else:
                overall_mean = np.nan
                total_valid_count = 0  

            overall_token_cover = float(sum(token_cover))

             # 转换step_means列表中的numpy标量为Python float
            step_means_converted = [float(m) if np.isfinite(m) else np.nan for m in step_means]


            result[var_name][range_name] = {
                'overall_mean': overall_mean,  # 所有步的平均值
                'step_means': step_means_converted,  # 每一步的平均值（已转换为list）
                # 'step_counts': step_counts,  # 每一步的有效值数量
                'total_valid_count': total_valid_count  # 总有效值数量
            }
            if var_name=='token_change':
                result[var_name][range_name]['token_cover'] = token_cover  # 平均每一步的数量（已转换为list）
                result[var_name][range_name]['overall_token_cover'] = overall_token_cover  # 所有步的数量
                # ratio计算：如果token_cover是列表，需要逐元素计算
                if overall_token_cover > 0:
                    ratio_list = [float(tc / overall_token_cover) for tc in token_cover]
                else:
                    ratio_list = [0.0] * len(token_cover)
                result[var_name][range_name]['token_change_step_ratio'] = ratio_list  # 在所有步的数量占比（已转换为list）
                if token_change_positive_total > 0:
                    token_change_positive_total_ratio = float(overall_token_cover / token_change_positive_total)
                else:
                    token_change_positive_total_ratio = 0.0
                result[var_name][range_name]['token_change_position_ratio'] = token_change_positive_total_ratio  # 在部分位置中的占比/所有位置所有步数的占比
    return result