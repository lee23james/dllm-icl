# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import json, os
import argparse

def load_json_or_jsonl(file_path):
    data=[]
    #只读
    with open(file_path,'r',encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            #变成list[dict]
            for line in f:
                data.append(json.loads(line))
        elif file_path.endswith('.json'):
            #变成dict(jdon.loads)
            data=json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    return data

#直接调用他们的函数,再进行计算
def _plot_heatmap_to_file(data, vmin, vmax, filename):
    figsize = (6.8, 5.5)
    fig = plt.figure(figsize=figsize)
    right_margin = 0.82
    gs = gridspec.GridSpec(1, 1, left=0.15, right=right_margin, top=0.98, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    # 根据文件格式决定是否使用rasterized（PNG不需要）
    is_png = filename.lower().endswith('.png')
    hmap = sns.heatmap(
        data,
        cmap='YlGnBu',
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
    ax.set_ylabel('Steps', fontsize=28)
    ax.set_yticklabels(ticks, fontsize=24)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.set_title("") 
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
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(axis='y', pad=5, labelsize=24)
    cbar.set_label('Decoding Order', size=28, labelpad=0)
    # 提高PNG的DPI和优化参数
    plt.savefig(filename, format='png', dpi=600, bbox_inches='tight', pad_inches=0.0, 
                facecolor='white', edgecolor='none', transparent=False)
    plt.close(fig)
    print(f"Saved figure: {filename}")

def generate_individual_figures(
    confidence_result: np.ndarray,
    basename: str='figure_1'
):
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
    except Exception as e:
        print(f"Times New Roman font not found. Using default font. Error: {e}")
    #方便以后插入
    heatmap_datas=[confidence_result]
    vmin=min(np.min(d)for d in heatmap_datas)
    vmax=max(np.max(d)for d in heatmap_datas)
    _plot_heatmap_to_file(
        #conf
        data=heatmap_datas[0],vmin=vmin,vmax=vmax,
            filename=f"{basename}.png"
    )

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('--task',type=str,default='sudoku')
    #第一个是位置,第二个是index
    parser.add_argument('--data_path',type=str,default='./heatmap_params/sudoku_0_single_0.json')
    parser.add_argument('--task',type=str,default='sudoku')
    # parser.add_argument('--position',type=int,default=4)
    # parser.add_argument('--index',type=int,default='0')
    args=parser.parse_args()
    
    data_path=args.data_path
    task=args.task
    # position=args.position
    # index=args.index
    data=load_json_or_jsonl(data_path)
    confidence_result = np.array(data['confidence_result'])
    if confidence_result.ndim==3:
        confidence_result=confidence_result.squeeze()
    
    #创建保存路径
    # 从 data_path 提取最后一级文件名（去掉扩展名）
    data_basename = os.path.splitext(os.path.basename(data_path))[0]
    file_path=f'heatmap_results/{data_basename}'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    generate_individual_figures(confidence_result,basename=f"{file_path}/{task}")


