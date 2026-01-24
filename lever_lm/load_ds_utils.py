import os 
import datasets
from datasets import DatasetDict, load_dataset

#这里我需要加载hf数据集,同时还需要建立本地加载gsm8k数据集
#目前我打算弄一下gsm8k,看\看能不能进行提点
def load_hf_ds(
    name,
    split=None
):
    ds=load_dataset(name,split=split)
    if isinstance(ds,DatasetDict):
        ds["train"]=ds["train"].add_column("idx",list(range(len(ds["train"]))))
    else:
        ds=ds.add_column("idx",list(range(len(ds))))

#加载本地的gsm8k数据集,
#这里主要利用的是main的文件夹
#在这里就可以处理成一个数据方便之后进行处理
#我这里最好加一个是否是query的问题,方便之后可以进行筛选
#从hub上下载的还没有适配
def load_gsm8k_ds(
    version,
    train_path,
    val_path,
):
#注意一定要加载main的数据集文件夹
#最后我需要的是提取出来对某个问题的信心程度
#根据原来论文的结果,我应该对tarin去找结果,但是可以再val验证效果,所以还是要把val也加进来
    if version == "local":
        data_files={
            "train":train_path,"validation":val_path
        }
        ds=load_dataset("parquet",data_files=data_files)

        #添加全局唯一的idx适配杠杠模型,为所有数据集都添上idx的标识
        #这里加上任务的唯一标识
        for split in ds.keys():
            ds[split]=ds[split].add_column("idx",list(range(len(ds[split]))))
            #添加isquery字段，初始值为0（每个ICL样本都是0）
            ds[split]=ds[split].add_column("isquery",[0]*len(ds[split]))

        def process_gsm8k_data(batch):
            questions=[i for i in batch["question"]]
            answers=[q for q in batch["answer"]]
            #拼接成ICL的形式(之后再进行answer的消除)
            #这里已经拼接成独立的字符串,然后每个QA初始化为isquery=0,判断是test之后再变成1
            q_a=[f"question:{q}\n<answer>\n{a}\n</answer>"for q,a in zip(questions,answers)]
            batch["q_a"]=q_a
            return batch

        #应用数据处理函数
        ds=ds.map(process_gsm8k_data,batched=True,num_proc=12)
    else:
        raise ValueError(f"Invalid version: {version}")
    return ds
#其他数据集之后再去想,这里先进行gsm8k的数据处理


