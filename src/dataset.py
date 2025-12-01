import json
import time
import torch
import random
import pickle
import argparse
import logging

from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm

from itertools import permutations

from utils.eval_utils import corpus_sampling,create_prompt


logger=logging.getLogger()
logger.setLevel(logging.ERROR)

import os

#底层被调用代码放到src中
#处理情感分析类问题,进行数据处理
#这个文件的主要功能还是研究设计复杂的数据集在方法
class PromptCorpus:
    def __init__(
        self,

        train_data_path="data/train.jsonl",
        test_data_path="data/dev.jsonl",
        model_name="GSAI-ML/LLaDA-8B-Base",
        n_shot=4,
        label_mapping={0:"bad",1:"good"},
        corpus_params={"sentence_1_str":"","sentence_2_str":"","label_str":""},
        template="f'Review: {sentence_1}\nSentiment:{label_text}\n\n'",#str
        sample_mode="balance",
        sentence_pair=False,
        mask_length=24
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        self.kshot=n_shot
        self.max_sequence_length = 1022
        self.sample_mode=sample_mode
        self.label_mapping=label_mapping

        self.restricted_token=[]
        for label_str in self.label_mapping.values():
            label_index =self.tokenizer.encode(f"{label_str}")
            # 不再强制要求单token，允许多token的label
            print(f"[DEBUG] label='{label_str}' encodes to {label_index} ({len(label_index)} tokens)")
            self.restricted_token.extend(label_index)  # extend而不是+=，处理多token情况
        
        self.restriced_token=tuple(self.restricted_token)
        self.mask_length=mask_length
        #之前自己写的哦都市加载
        full_train_data=self.load_jsonl(train_data_path)
        #平衡采样训练集
        #一样的分别对test和train进行处理,变成通用prompts构造器
        self.train_data = corpus_sampling(full_train_data,kshot=self.kshot,mode=sample_mode,
                                          label_str=corpus_params["label_str"])

        #加载测试集List[Dict]字段,当成json使用
        self.test_data = self.load_jsonl(test_data_path)

        self.template=template
        #是一句还是两句
        self.sentence_pair = sentence_pair
        #字段
        self.corpus_params = corpus_params
        
        logger.info(f"{self.kshot}-shot, label_mapping: {label_mapping}, "
                    f"template: {template}")

        #list[dict]字段,当成json使用
        # self._cache={}


    #返回list[dict]字段,当成json使用
    #这个函数是用来加载jsonl文件的,返回的是一个list,list的每个元素是一个字典,字典的key是index,value是字典
    #字典的key是sentence_1_str,sentence_2_str,label_str,index
    #这个函数是用来加载jsonl文件的,返回的是一个list,list的每个元素是一个字典,字典的key是index,value是字典
    @staticmethod
    def load_jsonl(fp):
        data=[]
        with open(fp,'r',encoding='utf-8') as f:
            for i,line in enumerate(f):
                decoded = json.loads(line)
                decoded["index"] = i #add index for all samples
                data.append(decoded)
        return data
    

    def __getitem__(self,item):
        #训练prompts有哪些
        train_prompts=[]
        label_str=self.corpus_params["label_str"]
        if self.sentence_pair:
            sentence_1_str=self.corpus_params["sentence_1_str"]
            sentence_2_str=self.corpus_params["sentence_2_str"]
        else:
            sentence_1_str=self.corpus_params["sentence_1_str"]
        
        #训练的label list
        train_labels=[]
        #这里的train_data是一个list,里面的每个元素是一个字典,字典的key是sentence_1_str,sentence_2_str,label_str,所以每一个都得用[]表示取模
        #绝大部分数据格式都是list[dict]方便对各个组进行操作(每个组里有各种字段-》就是json格式)
        #处理好的List[Dict]字段,每个元素是一个字典,字典的key是sentence_1_str,sentence_2_str,label_str,index
        for data in self.train_data:
            if self.sentence_pair:
                #返回元组
                train_sentence = (data[sentence_1_str],data[sentence_2_str])
            else:
                #括号的是tuple(元组)形式,表示一个句子
                train_sentence =(data[sentence_1_str], )
            #这里label是数字,但是以字符串形式存储
            train_label=data[label_str]
            #train_labels是一个list,里面包含所有的label
            #train_label应该都是数字,所以需要通过label_mapping转化成字符串
            train_labels.append(train_label)
            #每一个数据对应的label名字
            train_label_text= self.label_mapping[train_label]
            #针对的是train_data中的每一个数据,构造一个prompt
            p=create_prompt(mask_length=self.mask_length,template=self.template,sentence=train_sentence,
                            label_text=train_label_text,test=False,
                            sentence_pair=self.sentence_pair)
            #收集每一个train_prompt
            train_prompts.append(p)


        #使用cache存储相关情感分析类的prompt,方便后续进行计算(这里我还不太清楚,后续考虑换成4个位置进行嵌入)
        #查看List[Dict]是否有相关字段
        #对于原来的有多少shot就有固定的多少排列方式,但是我现在需要的是根据kshot,有多少query嵌入方式
        # if "train_prompts_permutation" in self._cache:
        #     #返回的是list[dict]字段,当成json使用
        #     train_prompts_permutation = self._cache["train_prompts_permutation"]
        # else:
        #     #利用cache进行计算(这里记得去改变permute的配置,因为这个函数主要是用于排列,但是我们不太需要)
        #     #这里考虑要不要加上去做prompt的cache
        #     #train_prompts目前也就两个label,4shot,目前是进行全排列,并且生成list(str),其中中间结果是list(tuple)
        #     # train_prompts_permutation= self.permute_train_prompts(train_prompts)
        #     for query_position in range(len(train_prompts)+1):
        #         train_prompts_permutation.append(self.insert_query_at_position(query_position,train_prompts,test_sequence))
        #     self._cache={"train_prompts_permutation":train_prompts_permutation}
        #     print("train_prompts_length: ",len(self.tokenizer.encode(train_prompts_permutation[0])))

        #item表示当时的序列序号,可以通过这样的序号去访问test_data中的元素,这样子对于所有的test_data就可以有针对性的处理每一个数据
        if self.sentence_pair:
            #test_data是List[Dict]字段,当成json使用,现在取的是第item个元素的sentence_1_str和sentence_2_str字段
            test_sentence = (self.test_data[item][sentence_1_str],self.test_data[item][sentence_2_str])
        else:
            #返回的是元组
            test_sentence = (self.test_data[item][sentence_1_str],)
        test_label = self.test_data[item][label_str]
        #算是一种配置
        test_label_text = self.label_mapping[test_label]
        #构建了包含<|mdm_mask|>的prompt
        test_sequence = create_prompt(mask_length=self.mask_length,template=self.template, sentence=test_sentence,
                                    label_text=test_label_text,test=True,
                                    sentence_pair=self.sentence_pair)
        input_sequences=[] # train + test-》tokenizer
        # input_sequences_prompt=[] #train only
        raw_sequences=[]#train+ test
        
        # train_prompts_permutation是训练的prompt的排列,所以需要遍历每一个排列->这里能不能魔改成遍历所有的位置,变成在所有位置上嵌入query的prompt
        #我的想法就是一次构造所有位置的prompt,这样子方便所有实验条件固定来进行计算
        #train_prompts_permutation是找出四种排列,但其实不然,我要做的就是把全排列嵌入到test中
        #这里需要再写相关函数(先魔改排列)
        #先进行assert看看是不是位置+1=prompts的数量
        if self.sample_mode=="balance":
            #因为每个标签都拿n个,所以他这里构造了一个相对平衡的标签bias(根据相关论文影响位置的主要是标签bias)
            assert len(train_prompts)==self.kshot*len(self.label_mapping),"train_prompts的数量应该等于位置*label的数量"
        elif self.sample_mode=="random":
            assert len(train_prompts)==self.kshot,"train_prompts的数量应该等于位置"
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")
        for query_position in range(len(train_prompts)+1):
            #这里需要进行相应改进
            # raw_sequence=''.join([train_sequence,test_sequence])
            # raw_sequence=raw_sequence.strip(" ")
            # raw_sequence_train_only = train_sequence
            raw_sequence=self.insert_query_at_position(query_position,train_prompts,test_sequence)
            raw_sequence=raw_sequence.strip(" ")
            #这个encode需要改一下,这个是完整的train+test的prompt,我得需要改一下,我要识别到mdm_mask然后转化为特殊的token
            input_sequence = self.tokenizer(raw_sequence)['input_ids']

            #这个是训练的prompt进行转化(后面都要改)
            #input_sequence_prompt = self.tokenizer.encode(train_prompts,add_special_tokens=False)
            input_sequence=input_sequence[-self.max_sequence_length:]
            # 这里我需要用到我原来的函数,输入prompt,输出token_ids
            # input_sequence = self.tokenizer.encode(raw_sequence,add_specicial_tokens=True)
            #这个是将训练的prompt进行转化
            # input_sequence_prompt = self.tokenizer.encode(train_sequence,add_special_tokens=True)
            # 收集的是原始token转化的token_ids
            input_sequences.append(torch.tensor(input_sequence))
            # 收集的是原始的text
            raw_sequences.append(raw_sequence)

        #收录所有的train_prompt
        #这个labels应该是作为key,因为需要根据label来进行分组
        #并且还应该包括不同的排列组合(但是我是测试不同位置,并不需要)-》我觉得不用增加
        # _d={}
        # for train_label,train_prompt in zip(train_labels,train_prompts):
        #     if train_label in _d:
        #         _d[train_label].append(train_prompt)
        #     else:
        #         _d[train_label] =[train_prompt]
        # #返回的是dict[list]把所有train_prompt按照label进行分组,然后合成一个字符串,再转化为token_id
        # #我们不需要对于train_prompt进行分组
        # train_prompts_ids=[self.tokenizer.encode("".join(prompts))for prompts in _d.values()]

        #他们把input_sequences堆叠成多个batch,但是我们是不是可以徒增batch纬度(因为原来的模型是batch=1的)
        #在LLaDA中,对于生成只支持batch=1,所以我不能
        return {"input_sequence":input_sequences,#List[Tensor]
                "label": test_label,#生成的所有位置的label List[int]
                "raw_sequence":raw_sequences,#List[str],没有考虑到batch
                "train_metadata":self.train_data,#List[Dict]字段,当成json使用
                "test_index":self.test_data[item]["index"],#测试集的索引
                "query_position":[idx for idx in range(len(train_prompts)+1)]#List[int],表示所有插入位置的索引
                # "input_sequences_prompt":torch.stack(input_sequences_prompt,dim=0),
                # "train_prompts_ids":train_prompts_ids,
                }

    #根据query_position生成插入到任一位置的prompt
    #插入query的函数,主要用于插入到train_prompts的指定位置
    def insert_query_at_position(self,query_position,train_prompts,test_prompt):
        """
        根据query_position将test_sequence插入到train_prompts的指定位置
        :param train prompts: list of strings ["Sent: xxx\nLabel: xxx\n", "Sent: xxx\nLabel: xxx\n"]
        Args:
            train_prompts: 训练提示列表
            test_prompt:测试序列
            query_position:插入位置(0=最后,1=倒数第二,...,n_shot=最前)
        Returns:
            list: 插入后的提示列表
        """
        #这些都是根据数据集的yaml进行prompt的构建
        if query_position == 0:
            #放在最右边(最后)
            return ''.join(train_prompts)+test_prompt
            
        elif query_position == len(train_prompts):
            #放在最左边(最前)
            return test_prompt+'\n\n' + ''.join(train_prompts)
        else:
            #放在中间指定位置
            left_prompts = train_prompts[:len(train_prompts)-query_position]
            right_prompts= train_prompts[len(train_prompts)-query_position:]
            return ''.join(left_prompts)+test_prompt+'\n\n'+''.join(right_prompts)


    def __len__(self):
        return len(self.test_data)



    # Train example concat order: 1-1,2-2,3-6,4-24
    #一般来说我都会在nshot的不同位置之间找prompt,然后再去和test_prompts进行拼接
    # @staticmethod
    # def permute_train_prompts(train_prompts,test_prompts,max_count=24,kshot=4):
    #     """
    #     :param train_prompts: list of strings ["Sent: xxx\nLabel: xxx\n", "Sent: xxx\nLabel: xxx\n"]
    #     :param test_prompts: list of strings ["Sent: xxx\nLabel: xxx\n", "Sent: xxx\nLabel: xxx\n"]
    #     :return: 
    #     """ 
        


if __name__=="__main__":

    import yaml
    import easydict
    corpus_config = yaml.safe_load(open("config/rte.yaml"))
    cfg = easydict.EasyDict(corpus_config)
    #logger.INFO(CFG)
    print(cfg)
    rte=PromptCorpus(**cfg)
    datapoint = rte[0]