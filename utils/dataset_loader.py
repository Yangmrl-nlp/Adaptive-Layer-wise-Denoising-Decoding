from abc import ABC, abstractmethod
import json
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, Type
import datasets
import yaml
from utils.config import Config
from PIL import Image

class Dataset(ABC):
    def __init__(self):
        self.raw_root_path = './data/raw_data'
        self.labeled_root_path = './data/labeled_data'
        with open("configs/shots.yaml", "r", encoding="utf-8") as f:
            self.shot_data = yaml.safe_load(f) or {}

    @abstractmethod
    def _load_raw_split(self, split):
        ...

    def load_raw(self, split='all'):
        if split in ["train", "valid", "test"]:
            return self._load_raw_split(split)
        elif split == "all":
            return {split: self._load_raw_split(split) for split in ["train", "valid", "test"]}
        else:
            raise ValueError(f"Unsupported split: {split}")

    @abstractmethod
    def _load_labeled_split(self, split):
        ...

    def load_labeled(self, split):
        if split in ["train", "valid", "test"]:
            return self._load_labeled_split(split)

        else:
            raise ValueError(f"Unsupportedsplit: {split}")
    

class TextVQA(Dataset):
    def __init__(self, args):
        super().__init__()
        self.prefix = 'textvqa_data/texts'
        self.image_prefix = 'textvqa_data/images'
        self.args = args
        self.shot_cfg = Config(self.shot_data)._cfg[self.args.dataset]

    def _load_raw_split(self, split):
        dir_path = Path(f"{self.raw_root_path}/{self.prefix}/{split}")
        list_data_dict = []
        for file in tqdm(dir_path.iterdir()):
            with open(file, "r") as f:
                data = json.load(f)
                ans_list = data["answers"]
                idx, mx = 0, 0
                mp = defaultdict(int)
                for ans in ans_list:
                    mp[ans] += 1
                    if mp[ans] > mx:
                        mx = mp[ans]
                        idx = ans_list.index(ans)
                list_data_dict.append({
                    "question": data["question"],
                    "answer": ans_list[idx],
                    "image_path": f"{self.raw_root_path}/{self.image_prefix}/{split}/{data['image_id']}.jpg"
                })
        return pd.DataFrame(list_data_dict)
    def _load_labeled_split(self, split):
        dir_path = f"{self.labeled_root_path}/{self.args.llm}/{self.args.dataset}/{split}"
        data_file = os.path.join(dir_path, 'data.csv')
        dataset = datasets.load_dataset("csv", data_files=data_file)['train']

        return dataset

    def _create_shot(self):
        shots = []
        imgs = []

        for i in self.shot_cfg.keys():
            shot = self.shot_cfg[i]
            shots.append(
            {"role": "user", 
             "content": [
                {"type": "text", "text": "Question: " + shot["question"]},
                {"type": "image"}]
            })
            shots.append(
            {"role": "assistant",
            "content": [
                {"type": "text", "text": shot['answer']+'\n'}
            ],
            })

            imgs.append(Image.open(shot['img_path']).convert("RGB"))

        return {'text': shots, 'imgs': imgs}

class POPE(Dataset):
    def __init__(self, args):
        super().__init__()
        self.prefix = f'pope/{args.pope}'
        self.image_prefix = f'pope/{args.pope}/images'
        self.args = args
        self.shot_cfg = Config(self.shot_data)._cfg[self.args.dataset]
        
    def _load_raw_split(self, split):
        file_path = Path(f"{self.raw_root_path}/{self.prefix}/{split}/{split}.json")
        list_data_dict = []
        pre = "this is image "
        with open(file_path, "r") as f:
            data = json.load(f)
            for d in data:
                list_data_dict.append({
                    "question": d["question"],
                    "answer": d["answer"],
                    "image_path": f"{self.raw_root_path}/{self.image_prefix}/{d['image_filename']}"
                })
        return pd.DataFrame(list_data_dict[109:110])
    
    def _load_labeled_split(self, split):
        dir_path = f"{self.labeled_root_path}/{self.args.llm}/{self.args.dataset}/{self.args.pope}/{split}"
        data_file = os.path.join(dir_path, 'data.csv')
        #print(dir_path)
        dataset = datasets.load_dataset("csv", data_files=data_file)['train']

        return dataset    
    def _create_shot(self):
        shots = []
        imgs = []

        for i in self.shot_cfg.keys():
            shot = self.shot_cfg[i]
            shots.append(
            {"role": "user", 
             "content": [
                {"type": "text", "text": "Question: " + shot["question"]},
                {"type": "image"}]
            })
            shots.append(
            {"role": "assistant",
            "content": [
                {"type": "text", "text": shot['answer']+'\n'}
            ],
            })

            imgs.append(Image.open(shot['img_path']).convert("RGB"))

        return {'text': shots, 'imgs': imgs}

class CHAIR(Dataset):
    def __init__(self, args):
        super().__init__()
        self.prefix = '/mnt/data1/zjx/project/eval/data/raw_data/chair/data/'
        self.image_prefix = '/mnt/data1/zjx/project/eval/data/raw_data/chair/data/val2017'
        self.args = args
        self.shot_cfg = Config(self.shot_data)._cfg[self.args.dataset]

    def _load_raw_split(self, split):
        
        json_path = Path(f"{self.prefix}/selected_500_images.json")
        image_root = Path(f"{self.image_prefix}")

        list_data_dict = []

        with open(json_path, "r") as f:
            data_list = json.load(f)

        for item in tqdm(data_list, desc=f"Loading {split} data"):
            image_id = item['image_id']
            question = item["query"]
            answer = item["gt_objects"]
            image_filename = item["image_path"]

            list_data_dict.append({
                "image_id":image_id,
                "question": question,
                "answer": answer,
                "image_path": str(image_root / image_filename)
            })

        return pd.DataFrame(list_data_dict)
     
    def _load_labeled_split(self, split):
        dir_path = f"{self.labeled_root_path}/{self.args.llm}/{self.args.dataset}/{split}"  # 路径
        data_file = os.path.join(dir_path, 'data.csv')  # CSV 文件
        dataset = datasets.load_dataset("csv", data_files=data_file)['train']  
        return dataset

    def _create_shot(self):

        if not hasattr(self, 'shot_cfg') or not self.shot_cfg:
            return {'text': [], 'imgs': []}
        
        shots = []  
        imgs = []   

        for i in self.shot_cfg.keys():
            shot = self.shot_cfg[i]
            shots.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: " + shot["question"]},
                    {"type": "image"} 
                ]
            })
    
            shots.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shot['answer']+'\n'}
                ],
            })

            imgs.append(Image.open(shot['img_path']).convert("RGB"))

        return {'text': shots, 'imgs': imgs}

class MME(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # MME 根路径
        self.mme_root = Path("/mnt/data1/zjx/project/datasets/mme")
        # few-shot 配置（如果有）
        self.shot_cfg = getattr(Config(self.shot_data), "_cfg", {}).get(self.args.dataset, None)

    def _load_raw_split(self, split="val"):
        """
        读取 MME 数据格式
        每个子任务目录包含：
          - <task>_QA.json
          - images 文件夹
        """
        list_data_dict = []

        # 遍历14个子任务文件夹
        for sub_dir in tqdm(sorted(os.listdir(self.mme_root)), desc=f"Loading MME ({split})"):
            sub_path = self.mme_root / sub_dir
            if not sub_path.is_dir():
                continue

            json_file = sub_path / f"{sub_dir}_QA.json"
            image_dir = sub_path / "images"

            if not json_file.exists():
                print(f"⚠️ {json_file} not found, skipping.")
                continue

            # 读取 JSON 文件
            with open(json_file, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            for item in data_list:
                question = item["question"]
                answer = item["answer"]
                image_filename = item["image"]
                image_path = image_dir / image_filename

                if not image_path.exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue

                list_data_dict.append({
                    "subcategory": sub_dir,
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                    "image": image_filename
                })

        # 返回 DataFrame 以兼容原逻辑
        return pd.DataFrame(list_data_dict)

    def _load_labeled_split(self, split):
        """
        若仍需从 labeled 路径加载标注，可保持原逻辑。
        """
        dir_path = f"{self.labeled_root_path}/{self.args.llm}/{self.args.dataset}/{split}"
        data_file = os.path.join(dir_path, 'data.csv')
        dataset = datasets.load_dataset("csv", data_files=data_file)['train']
        return dataset

    def _create_shot(self):
        """
        few-shot 示例生成逻辑保持原有。
        """
        if not hasattr(self, 'shot_cfg') or not self.shot_cfg:
            return {'text': [], 'imgs': []}
        
        shots = []
        imgs = []
        for i in self.shot_cfg.keys():
            shot = self.shot_cfg[i]
            shots.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: " + shot["question"]},
                    {"type": "image"}
                ]
            })
            shots.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shot['answer'] + '\n'}
                ],
            })
            imgs.append(Image.open(shot['img_path']).convert("RGB"))

        return {'text': shots, 'imgs': imgs}

# registry
DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "textvqa": TextVQA,
    "pope":POPE,
    "pope_alw":POPE,
    "chair":CHAIR,
    "mme":MME
}

def load_dataset(args):
    cls = DATASET_REGISTRY.get(args.dataset.lower(), {})
    if not cls:
        raise NotImplementedError

    return cls(args)

