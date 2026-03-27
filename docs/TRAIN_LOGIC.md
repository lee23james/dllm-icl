平均置信度公式
设：
$S$：样本集合，$|S| = N$（样本数）
$T$：解码步数集合，$|T| = T_{\max}$（总步数）
$\mathcal{A}i$：第 $i$ 个样本的答案区域 token 位置集合
$\text{conf}{i,t,j}$：第 $i$ 个样本、第 $t$ 步、第 $j$ 个 token 位置的置信度（softmax 概率）
则平均置信度定义为：
$$
\text{MeanConf} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{T_{\max}} \sum_{t=1}^{T_{\max}} \left( \frac{1}{|\mathcal{A}i|} \sum{j \in \mathcal{A}i} \text{conf}{i,t,j} \right) \right)
$$
或者更紧凑地写成：
$$
\text{MeanConf} = \mathbb{E}{i \sim S} \left[ \mathbb{E}{t \sim T} \left[ \mathbb{E}{j \in \mathcal{A}_i}[\text{conf}{i,t,j}] \right] \right]
$$
其中 $\mathbb{E}$ 表示期望（平均值）。
分步说明
单步内平均（答案区域）：
$$
\bar{c}{i,t} = \frac{1}{|\mathcal{A}_i|} \sum{j \in \mathcal{A}i} \text{conf}{i,t,j}
$$
跨步平均（单个样本）：
$$
\bar{c}i = \frac{1}{T{\max}} \sum_{t=1}^{T_{\max}} \bar{c}{i,t}
$$
跨样本平均（最终值）：
$$
\text{MeanConf} = \frac{1}{N} \sum_{i=1}^{N} \bar{c}i
$$
论文中的表述建议
> 平均置信度（Mean Confidence）：我们计算每个样本在解码过程中，答案区域内所有 token 在所有步数上的平均置信度，然后对所有样本求平均：
> $$
> \text{MeanConf} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T} \sum_{t=1}^{T} \frac{1}{|\mathcal{A}i|} \sum{j \in \mathcal{A}i} p\theta(y_{i,t,j} \mid x_i, \text{context}t)
> $$
> 其中 $p_\theta(y_{i,t,j} \mid x_i, \text{context}t)$ 表示模型在第 $t$ 步对第 $i$ 个样本答案区域第 $j$ 个 token 位置的预测概率（softmax 输出），$\mathcal{A}_i$ 是答案区域的 token 索引集合，$N$ 是样本数，$T$ 是总步数。
如果需要，我可以把这段整理成 LaTeX 片段，直接粘贴到论文里。