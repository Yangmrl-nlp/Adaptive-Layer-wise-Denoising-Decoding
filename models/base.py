from abc import ABC, abstractmethod
from utils.config import Config
import yaml
from typing import Dict, Any
from utils.dataset_loader import load_dataset
from utils.utils import *
from utils.evaluate import *
import torch
import os
from tqdm import tqdm
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitsProcessorList
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from utils.evaluate_chair import CHAIR_SCORE
from utils.mme_eval import calculate_metrics

class BaseModel(ABC):
    def __init__(self, args):
        self.model = None
        self.tokenizer = None
        self.args = args
        self._load_config()
        self.processors = LogitsProcessorList()
        self.processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))
        
    @abstractmethod
    def load(self):
        ...

    @torch.inference_mode()
    def infer(self, classifier):
        mode_list = {
            'textvqa': self.generate,
            'pope':self.generate,
            'chair':self.generate,
            'mme':self.generate,
            'xxx': self.multichoice,
        }
        
        dataset = load_dataset(self.args)
        data = dataset.load_raw('test')
        
        if self.args.dataset == 'chair':
            write_dict = {'image_id':[],'caption': []}
        elif self.args.dataset == 'mme':
            write_dict = {'category':[],'prompt': [], 'text': [], 'answer': [],'image':[]}
        elif self.args.dataset == 'pope':
            write_dict = {'question': [], 'answer': [], 'predict': []}
        
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            pred = mode_list[self.args.dataset](dataset, row, classifier)
            
            if self.args.dataset == 'chair':
                write_dict['image_id'].append(row['image_id'])
                write_dict['caption'].append(pred)
            elif self.args.dataset == 'pope':
                write_dict['question'].append(row['question'])
                #print(f"debug: {len(row['answer'])}")
                write_dict['answer'].append(row['answer'])
                # preds = ""
                # for p in pred:
                #     preds = preds+" "+p
                write_dict['predict'].append(pred)
                #print(preds)
            elif self.args.dataset == 'mme':
                write_dict['category'].append(row['subcategory'])
                write_dict['prompt'].append(row['question'])
                write_dict['answer'].append(row['answer'])
                write_dict['image'].append(row['image'])
                write_dict['text'].append(pred)
        data = pd.DataFrame(write_dict)
            
        if self.args.decode_method != 'alw':
            output_path = f'./results/{self.args.llm}/{self.args.dataset}'
            os.makedirs(output_path, exist_ok=True)
            output_file = f'{output_path}/{self.args.decode_method}.csv'
        else:
            tp_pth = Path(self.args.tuned_path)
            tuned_dir = f'./results/{self.args.llm}/{self.args.dataset}/ALW/{tp_pth.stem}'
            os.makedirs(tuned_dir, exist_ok=True)
            output_file = f'{tuned_dir}/alw.csv'
        
        if self.args.dataset == 'pope':
            data.to_csv(os.path.join(output_file), index=False)
        else:
           if self.args.decode_method == 'alw':
              output_file = f'{tuned_dir}/alw.json'
           else:
               output_file = f'{output_path}/{self.args.decode_method}.json'
           if self.args.dataset == "mme":
            data.to_json(output_file, orient='records', lines = True,force_ascii=False)
           else:
            data.to_json(output_file, orient='records',force_ascii=False,indent=2)

    def generate(self, dataset, row, classifier):
        ...
        
    def multichoice(self, prompt, true_answer, false_answers):
        ...

    def evaluate(self, csv_path=None):
        evaluate_list = {
            'textvqa': TextVqa_score,
            'pope': F1,
            'chair':CHAIR_SCORE,
            'mme': ACC
        }

        if not csv_path:
            csv_path = f'./results/{self.args.llm}/{self.args.dataset}/{self.args.decode_method}.csv'
        
        if self.args.classifier:
            if self.args.dataset == 'pope':
               df = pd.read_csv(csv_path+f'/{self.args.decode_method}.csv')
            else:
                json_path = f'{csv_path}/{self.args.decode_method}.json'
        else:
             if self.args.dataset == 'pope':
                df = pd.read_csv(csv_path)
             else:
                 json_path = f'./results/{self.args.llm}/{self.args.dataset}/{self.args.decode_method}.json'
        
        metric = 0
        acc = 0
        row_index = 0
        gold = []
        if self.args.dataset == 'chair':  
                cap_file = json_path
                image_id_key='image_id'
                caption_key='caption'
                coco_path='/mnt/data1/zjx/project/alw/data/raw_data/chair/data/coco_path'
                cache='/mnt/data1/zjx/project/eval/cache/chair_2017.pkl'
                base_save_path = "/mnt/data1/zjx/project/eval/results/llava1.5_7b/chair/eval_results/"

                if "/alw/" in json_path:
                    sub_path = json_path.split("/alw/")[-1]
                    save_path = os.path.join(base_save_path, "alw", sub_path)
                else:
                    filename = os.path.basename(json_path)
                    save_path = os.path.join(base_save_path, filename)
            
                evaluate_list[self.args.dataset](cap_file,image_id_key,caption_key,coco_path,cache,save_path)
        
        elif self.args.dataset == 'mme':
            cal = calculate_metrics()
            cal.process_result(json_path)
        
        else:
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            unknown = 0
            yes_answers = 0
            for _, row in df.iterrows():
                row_index+=1
                if self.args.dataset == "textvqa":
                    gold.append(row['answer'])
                    if row_index % 10 == 1:
                        pred = str(row['predict'])
                    if row_index % 10 == 0:
                        metric_single, acc_single = evaluate_list[self.args.dataset](pred, gold)
                        metric += metric_single
                        gold = []
                elif self.args.dataset == 'pope':    
                    gold = str(row['answer'])
                    pred = str(row['predict'])
                    gt_answer = gold.lower()
                    gen_answer = pred.lower()
                    gt_answer = gt_answer.strip()
                    gen_answer = gen_answer.strip()
                    if gt_answer == 'yes':
                        if 'yes' in gen_answer:
                            true_pos += 1
                            yes_answers += 1
                        else:
                            false_neg += 1
                    elif gt_answer == 'no':
                        if 'no' in gen_answer:
                            true_neg += 1
                        else:
                            yes_answers += 1
                            false_pos += 1
                    else:
                        print(f'Warning: unknown gt_answer: {gt_answer}')
                        unknown += 1

        res = defaultdict(float)
        
        if self.args.dataset == "textvqa":
            n = len(df)/10
            print(f"TextVqa Score: {metric / n * 100}")
            
        elif self.args.dataset == "pope":
            total_questions = len(df)
            # print(true_pos)
            # print(true_neg)
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (true_pos + true_neg) / total_questions
            yes_proportion = yes_answers / total_questions
            unknown_prop = unknown / total_questions
            print(f'F1 Score: {f1 * 100}')
            print(f'ACC: {accuracy * 100}')
            
            # res["POPE Score"] = f1
            # with open(f"/mnt/data1/yangmrl/ALW_debug/results/llava1.5_7b/pope/{self.args.pope}.json",'w',encoding='utf-8') as f:
            #     json.dump(res,f,ensure_ascii=False, indent=4)

    def _load_config(self, path: str = "configs/models.yaml") -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.cfg = Config(data)

    def resolve_model_cfg(self, name: str):
        return self.cfg.find_by_name(name)
