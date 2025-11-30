
from abc import ABC, abstractmethod
from typing import Dict, Type
import os
import yaml
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitsProcessorList
from utils.model_loader import load_model
from utils.dataset_loader import load_dataset
from utils.utils import *
from utils.config import Config


class Maker(ABC):
    def __init__(self, args):
        self.raw_root_path = './data/raw_data'
        self.labeled_root_path = './data/labeled_data'
        self.labeled_path = os.path.join(self.labeled_root_path, args.llm, args.dataset)
        if args.pope:
            self.labeled_path = os.path.join(self.labeled_path,args.pope)
        self.args = args
        self.processors = LogitsProcessorList()
        self.processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))

    @abstractmethod
    def _forward_split(self):
        ...

    def __call__(self, split='all'):
        return self.forward(split)

    def forward(self, split):
        dataset = load_dataset(self.args)
        llm, _ = load_model(self.args)
        llm.load()
        self.llm = llm
        self.mature_layer = llm.cfg_dict['layer'] - 1
        self.early_exit_layers = [i for i in range(llm.cfg_dict['layer']//4 * 3)]
        self.map = {}

        with torch.no_grad():
            if split in ["train", "valid"]:
                print(f'Making {split} set...')
                self._forward_split(dataset, split)

            elif split == "all":
                #print("debug")
                for split in ["train", "valid"]:
                    print(f'Making {split} set...')
                    self._forward_split(dataset, split) 

            else:
                raise ValueError(f"Unsupported split: {split}")

    def get_outputs(self, logits: tuple):
        dict_outputs = {int(layer):0 for layer in self.early_exit_layers}
        dict_outputs[self.mature_layer] = 0
        for l in dict_outputs.keys():
            dict_outputs[l] = logits[l]
        return dict_outputs

class GenerateMaker(Maker):
    def __init__(self, args):
        super().__init__(args)
    
    def get_lr(self,input_ids):
      image_token_id = self.llm.tokenizer.convert_tokens_to_ids("<image>")
      r = 0
      l = 100000
      input_ids = input_ids.tolist()
      for i in range(len(input_ids)):
        if input_ids[i] == image_token_id:
          l = min(l,i) 
          r = max(r,i)
      if self.args.llm == 'llava1.5_7b':
        l = 602
      elif self.args.llm == 'llavanext_8b':
        l = 2172
      #print(l,r)
      return l,r
  
    def visionmask(self, hidden, best_layer, input_ids, **kwargs):
        
        #hidden[:, self.vis_start:self.vis_end+1, :] = 0
        #print(self.model)
        l,r = self.get_lr(input_ids)
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          q_proj = self.llm.model.language_model.layers[best_layer].self_attn.q_proj
          k_proj = self.llm.model.language_model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
          #print(q.shape)
          #print(k.shape)
          attn_score = (q @ k.transpose(-1,-2))
        elif self.args.llm == 'instructblip_vicuna_7b':
          q_proj = self.llm.model.language_model.model.layers[best_layer].self_attn.q_proj
          k_proj = self.llm.model.language_model.model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
          attn_score = (q @ k.transpose(-1,-2))
        
        token_score = attn_score.mean(dim=(0, 1))   # shape: [seq_len]
        sorted_indices = torch.argsort(token_score, descending=False)
        cnt = 0
        for idx in sorted_indices:
          if idx >= l and idx <= r:
            hidden[:,idx,:] = 0
            cnt+=1
            #print("debug: ",sorted_indices[idx])
            if cnt == math.ceil((r-l+1)*0.1):
              break

        #print("avg_attn_score: ", token_score)
        #print("token_indices: ", sorted_indices)
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          layers = self.llm.model.language_model.layers[best_layer:]
          rotary_emb = self.llm.model.language_model.rotary_emb
        elif self.args.llm == 'instructblip_vicuna_7b':
          layers = self.llm.model.language_model.model.layers[best_layer:]
          rotary_emb = self.llm.model.language_model.model.rotary_emb
        
        for i, layer in enumerate(layers):
            seq_len = hidden.shape[1]
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # (1, seq_len)
            cos, sin = rotary_emb(hidden, position_ids)
            hidden = layer(hidden, position_embeddings=(cos, sin))

        #hidden = self.model.language_model.norm(hidden)
        head_layer = self.llm.model.get_output_embeddings()
        logits = head_layer(hidden)
        return logits
    
    def define_best(self, dict_outputs, input_ids, question_ids, answer_ids):
        fence = input_ids.shape[-1] - answer_ids.shape[-1]
        head_layer = self.llm.model.get_output_embeddings()
        #print(f"debug fence token: {self.llm.processor.decode([input_ids[0,fence-1]])}")
        if "pope" in self.args.dataset:
            if self.args.llm == 'llava1.5_7b':
                final_logits = head_layer(dict_outputs[self.mature_layer])[0, fence - 2, :]
            elif self.args.llm == 'instructblip_vicuna_7b':
                final_logits = head_layer(dict_outputs[self.mature_layer])[0, fence - 1, :]
        else:
            final_logits = head_layer(dict_outputs[self.mature_layer])[0, fence - 2:-1, :]
        final_logits = final_logits.log_softmax(dim=-1).unsqueeze(0)
        #final_logits.unsqueeze(0) 
        #print(f"debug logits : {final_logits}")
        layer_probs = []
        
        for layer in self.early_exit_layers[:-1]+[self.mature_layer]:
            probs = []
            if layer == 0:
                continue
            if layer == self.mature_layer:
                diff_logits = final_logits
            else:
                premature_layer = layer
                
                base_logits = self.visionmask(dict_outputs[premature_layer-1],premature_layer,input_ids[0])[0,fence-1,:]
                #base_logits = head_layer(dict_outputs[premature_layer])[0,fence-1,:]
                base_logits = base_logits.log_softmax(dim=-1).unsqueeze(0)
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                mask = final_logits[0] < -1e3
                # print(mask.shape)
                # print(f"debug: {base_logits.shape}")
                # print(f"debug: {final_logits.shape}")
                # print(f"debug: {base_logits.shape}")
                base_logits[0][mask] = -1e3

                diff_logits = final_logits + base_logits
            #print(f"debug: {answer_ids}")
            #print(input_ids[0,input_ids.shape[-1]-10:])
            #print(f"debug shape: {diff_logits.shape}")
            #diff_logits.unsqueeze(0)
            # if self.args.dataset == "pope":
            #     diff_logits = diff_logits.unsqueeze(0)
            
            for i in range(answer_ids.shape[-1]):
                
                current_token = fence + i - 1
                if self.args.llm == 'instructblip_vicuna_7b':
                    current_token = fence + i
                #print(f"debug: {self.llm.processor.decode([input_ids[0,current_token]])}")
                input_ids_all = input_ids[0, :current_token].unsqueeze(0).to(diff_logits.device) 
                current_logits = self.processors(input_ids_all, diff_logits[i].unsqueeze(0))
                current_logits = current_logits.softmax(dim=-1)
                #print(current_logits.shape)
                probs.append(current_logits[0, input_ids[0, current_token]].item())
                #print(f"debug: {current_logits[0, input_ids[0, current_token]].item()}")

            layer_probs.append(probs)

        layer_probs = np.array(layer_probs)
        #print(layer_probs)
        best_layers = np.argmax(layer_probs, axis=0)
        max_values = layer_probs[best_layers, np.arange(best_layers.shape[0])]
        best_layers = np.where(max_values == layer_probs[-1, :], -1, best_layers)

        contexts = [self.llm.tokenizer.decode(question_ids, skip_special_tokens=True)]
        for i in range(answer_ids.shape[-1]-1):
            context = self.llm.tokenizer.decode(torch.cat((question_ids, answer_ids[:i+1])), skip_special_tokens=True)
            contexts.append(context)

        assert len(contexts) == len(best_layers)
        #print(contexts)
        return contexts, best_layers.tolist()

class LogprobMaker(Maker):
    def __init__(self, args):
        super().__init__(args)

    def define_best():

        return

class V2TGenerateMaker(GenerateMaker):
    def __init__(self, args):
        super().__init__(args)

    def _forward_split(self, dataset, split):
        output_path = os.path.join(self.labeled_path, split)
        os.makedirs(output_path, exist_ok=True)
        #print(f"debug: {split}")
        data = dataset.load_raw(split)
        shot_dict = dataset._create_shot()

        write_dict = {'context': [], 'best_layer': []}
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            if not (row['question'] and row['answer'] and row['image_path']):
                continue
            inputs, question_ids, answer_ids = prepare_input_for_v2t_train(shot_dict, self.llm, row, self.args)
            if self.args.llm == 'llava1.5_7b':
               logits = self.llm.model(**inputs,output_hidden_states = True).hidden_states
            elif self.args.llm == 'instructblip_vicuna_7b':
                logits = self.llm.model(**inputs,output_hidden_states = True).language_model_outputs.hidden_states
            #logits = logits[1:] # llava返回的logits有embedding层 (len(logits)=33)，要去掉，只考虑decoder层
            
            dict_outputs = self.get_outputs(logits)
            contexts, best_layers = self.define_best(dict_outputs, inputs['input_ids'], question_ids, answer_ids)

            for i in range(len(contexts)):
                write_dict['context'].append(contexts[i])
                write_dict['best_layer'].append(best_layers[i])

        data = pd.DataFrame(write_dict)
        print(f"debug path:{os.path.join(output_path, 'data.csv')}")
        data.to_csv(os.path.join(output_path, 'data.csv'), index=False)
        return

# registry
MAKER_REGISTRY: Dict[str, Type[Maker]] = {
    "textvqa": V2TGenerateMaker,
    "pope": V2TGenerateMaker,
    "pope_alw": V2TGenerateMaker,
}

def load_maker(args):
    cls = MAKER_REGISTRY.get(args.dataset.lower(), {})
    
    if not cls:
        raise NotImplementedError

    return cls(args)