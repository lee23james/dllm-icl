这个是开放的icl架构,专门制作一个类,进行专门的icl的类,方便进行放进各种方法
进行专门的icl评估或者推理

结构如下:
open_mmicl/
├── icl_inferencer.py   # ICL 推理主逻辑（统一入口）
├── interface/          # 各种模型接口（Flamingo, IDEFICS, LLM）
├── retriever/          # 各种 ICD 检索/选择策略
├── metrics/            # 评价指标（CIDEr, VQA Accuracy 等）
└── utils.py            # ICL 相关的工具（输出保存、PPL等）
方便进行更详尽的推理(之前的代码太💩了)