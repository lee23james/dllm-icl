import json
import yaml

def load_groundtruth(jsonl_path:str,yaml_path:str)->list[str]:
    #读取label映射(确保key是字符串)
    cfg=yaml.safe_load(open(yaml_path,'r',encoding='utf-8'))
    label_mapping = cfg['label_mapping']
    label_mapping ={str(k):v for k,v in label_mapping.items()}

    #2)提取数据集,映射为文本label
    gts=[]
    with open(jsonl_path,'r',encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            lab=str(obj['label'])
            if lab not in label_mapping:
                raise ValueError(f"Label {lab} not found in label_mapping")
            gts.append(label_mapping[lab])
    return gts

if __name__ == '__main__':
    # 示例：SST2 dev_subsample
    jsonl_path = 'data/sst2/dev_subsample.jsonl'
    yaml_path = 'config/sst2.yaml'
    groundtruth = load_groundtruth(jsonl_path, yaml_path)
    print("len:", len(groundtruth))
    print(groundtruth[:10])