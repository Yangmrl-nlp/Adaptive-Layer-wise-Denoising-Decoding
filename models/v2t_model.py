import torch
from typing import Optional, Dict, Any
import math
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



from .base import BaseModel
from utils.utils import *
import torch.nn.functional as F
from .plot import  plot_token_prob_bar

def _import_llava():
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    return AutoProcessor, LlavaForConditionalGeneration

def _import_instructblip():
   from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
   return InstructBlipProcessor, InstructBlipForConditionalGeneration

def _import_llavanext():
   from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
   return LlavaNextProcessor, LlavaNextForConditionalGeneration
 
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

class V2TLLM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        family, cfg = self.resolve_model_cfg(args.llm)
        self.family = family             
        self.name = args.llm
        self.cfg_dict = cfg             
        self.args =args
   
    def load(self):
        
        arch = self.cfg_dict.get("arch", "auto")
        pretrained = self.cfg_dict["pretrained"]
        torch_dtype = _DTYPE_MAP.get(self.cfg_dict.get("torch_dtype", "float16"), torch.float16)

        print(f"[V2TLLM] Loading {self.name} (arch={arch}) from '{pretrained}' ...")
        
        if self.args.llm == 'llava1.5_7b':
          AutoProcessor, LlavaForConditionalGeneration = _import_llava()
        elif self.args.llm == 'instructblip_vicuna_7b':
          AutoProcessor, InstructBlipForConditionalGeneration = _import_instructblip()
        elif self.args.llm == 'llavanext_8b':
          AutoProcessor, LlavaNextForConditionalGeneration = _import_llavanext()
        
        
        if self.args.llm == 'llava1.5_7b':
            self.model = LlavaForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
            
        elif self.args.llm == 'instructblip_vicuna_7b':
            self.model = InstructBlipForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch_dtype,low_cpu_mem_usage=True, device_map='auto')
        elif self.args.llm == 'llavanext_8b':
            self.model = LlavaNextForConditionalGeneration.from_pretrained(Path(pretrained), torch_dtype=torch_dtype,low_cpu_mem_usage=True, device_map='auto')
        
        self.processor = AutoProcessor.from_pretrained(pretrained, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

        print("[V2TLLM] Success.\n")
    
    def get_lr(self,input_ids):
      image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
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
     
      return l,r
      
    def visionmask(self, hidden, best_layer,input_ids):
        
       
        l,r = self.get_lr(input_ids)
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          q_proj = self.model.language_model.layers[best_layer].self_attn.q_proj
          k_proj = self.model.language_model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
         
          if k.shape[-1] != q.shape[-1]:
            k = torch.nn.functional.linear(k, q_proj.weight[:, :1024])
          attn_score = (q @ k.transpose(-1,-2))
          
        elif self.args.llm == 'instructblip_vicuna_7b':
          q_proj = self.model.language_model.model.layers[best_layer].self_attn.q_proj
          k_proj = self.model.language_model.model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
          attn_score = (q @ k.transpose(-1,-2))
       
        token_score = attn_score.mean(dim=(0, 1))   

        sorted_indices = torch.argsort(token_score, descending=False)
        cnt = 0
        l,r = self.get_lr(input_ids)
        for idx in sorted_indices:
          if idx >= l and idx <= r:
            hidden[:,idx,:] = 0
            cnt+=1
            if cnt == math.ceil((r-l+1) * self.args.Prune_portion):
              break
        
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          layers = self.model.language_model.layers[best_layer:]
          rotary_emb = self.model.language_model.rotary_emb
        elif self.args.llm == 'instructblip_vicuna_7b':
          layers = self.model.language_model.model.layers[best_layer:]
          rotary_emb = self.model.language_model.model.rotary_emb
  
        for i, layer in enumerate(layers):
            seq_len = hidden.shape[1]
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  
            cos, sin = rotary_emb(hidden, position_ids)
            hidden = layer(hidden, position_embeddings=(cos, sin))
            
        head_layer = self.model.get_output_embeddings()
        logits = head_layer(hidden)
        return logits[0,-1,:]
    
    def generate(self, dataset, item, classifier):
        shot_dict = dataset._create_shot()
        inputs, question_text = prepare_input_for_v2t_infer(shot_dict, self, item,self.args)
        input_ids_all = inputs['input_ids']
        
        max_new_tokens = self.args.max_new_tokens
        past_key_values = None
        new_tokens_list = []
        
        true_logits = None
        answer_id = None
        denoised_logits = None
        for _ in range(max_new_tokens):
            outputs = self.model(**inputs, 
                                output_hidden_states=True, 
                                use_cache=True,
                                past_key_values=past_key_values,
                                return_dict=True)
            
            if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
              logits = outputs.hidden_states[1:] 
              past_key_values = outputs.past_key_values
            elif self.args.llm == 'instructblip_vicuna_7b':
              logits = outputs.language_model_outputs.hidden_states[1:]
              past_key_values = outputs.language_model_outputs.past_key_values
              
            mature_layer = len(logits)-1
            head_layer = self.model.get_output_embeddings()
            
            if true_logits == None:
              true_logits = list(logits)
              answer_id = true_logits[0].shape[1]
            else:
              for i in range(len(true_logits)):
                true_logits[i] = torch.cat([true_logits[i],logits[i]],dim = 1)
            
            final_logits = head_layer(logits[mature_layer])[:, -1, :]
            final_logits = final_logits.log_softmax(dim=-1)

            best_layer = get_best_layer(self.args, classifier, question_text)
    
            if best_layer == None and self.args.decode_method == 'vanilla':
                next_token_logits = final_logits 

            elif best_layer == -1:
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                mask = final_logits[0] < -1e3
                final_logits[0][mask] = -1e3
                next_token_logits = final_logits
          
            elif self.args.decode_method == 'dola':
                if self.args.dola == 'dynamic':
                  candidate_premature_layers = []
                  for i in range(16,32):
                    candidate_premature_layers.append(i)
                  stacked_premature_layers = torch.stack([logits[i][:, -1 , :] for i in candidate_premature_layers], dim=0)
                  softmax_mature_layer = torch.softmax(logits[mature_layer][:,-1, :], dim=-1)  
                  softmax_premature_layers = torch.softmax(stacked_premature_layers, dim=-1)  
                  M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  
                # 4. Calculate log-softmax for the KL divergence
                  log_softmax_mature_layer = torch.log_softmax(logits[mature_layer][:, -1, :], dim=-1) 
                  log_softmax_premature_layers = torch.log_softmax(stacked_premature_layers, dim=-1)  

                  # 5. Calculate the KL divergences and then the JS divergences
                  kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  
                  kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1) 
                  js_divs = 0.5 * (kl1 + kl2)  
                # 6. Reduce the batchmean
                  js_divs = js_divs.mean(-1)  
                  best_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                  
                
                base_logits = head_layer(logits[best_layer])[:,-1,:]
                base_logits = base_logits.log_softmax(dim=-1)
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                next_token_logits = final_logits - base_logits
            
            else:
                if self.args.Prune == 'True':
                  base_logits = self.visionmask(logits[best_layer-1],best_layer,input_ids_all[0])
                else:
                  base_logits = head_layer(logits[best_layer])[:,-1,:]
               
                base_logits = base_logits.log_softmax(dim=-1)
                
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                
                if self.args.Prune == 'True':   
        
                  next_token_logits = final_logits + base_logits
                  if denoised_logits == None:
                    denoised_logits = [next_token_logits]
                  else:
                    for i in range(len(denoised_logits)):
                      denoised_logits[i] = torch.cat([denoised_logits[i],next_token_logits],dim = 0)
                else:
                  next_token_logits = final_logits - base_logits #ALW

            input_ids_all = input_ids_all.to(next_token_logits.device)
            
            next_token_logits = self.processors(input_ids_all, next_token_logits)
            
            next_token = torch.argmax(next_token_logits, dim=-1)
           
            new_tokens_list.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([inputs['attention_mask'], inputs['attention_mask'].new_ones((inputs['attention_mask'].shape[0], 1))], dim=-1)
            
            inputs['attention_mask'] = attention_mask
            
            inputs['input_ids'] = input_ids
           
            if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
              inputs["pixel_values"] = None
            question_ids = self.tokenizer.encode(question_text)
            
            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)
            question_text = question_text + self.tokenizer.decode(next_token)
    
        for t in new_tokens_list:
          print(self.tokenizer.decode([t]))
        denoised_logits = denoised_logits[0]
        probs = torch.exp(denoised_logits)
        entropy_per_timestep = -( probs * denoised_logits).sum(dim=-1)
        stacked = [entropy_per_timestep]
        
        final_logits = head_layer(true_logits[mature_layer])[0, answer_id-2: -1, :]
        final_logits = final_logits.log_softmax(dim=-1)
        probs = torch.exp(final_logits)                        
        entropy_per_timestep = -( probs * final_logits).sum(dim=-1)
        stacked.append(entropy_per_timestep)
        
        ls = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
        ls.reverse()
        for i in ls:
            base_logits = head_layer(true_logits[i])[0, answer_id-2: -1, :]
            base_logits = base_logits.log_softmax(dim=-1)
            probs = torch.exp(base_logits)  
            entropy_per_timestep = -(probs * base_logits).sum(dim=-1)
          
            stacked.append(entropy_per_timestep)
            
        stacked_tensors = torch.stack(stacked)
        
        HR_5 = stacked_tensors.cpu().numpy()
        sns.set_theme(font_scale=1.5)
        sns.set_context({"figure.figsize":(35,15)})
        heatmap = sns.heatmap(HR_5, annot=True, linewidths=1.5, fmt=".4f", cbar=False,
                            linecolor="black", cmap="Greens", 
                            yticklabels=['Denoised',32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0],
                            xticklabels=[i for i in range(1, HR_5.shape[1]+1)])
        heatmap.xaxis.tick_top()
        plt.yticks(rotation=0)
        plt.ylabel('i-th early exit layer')
        plt.savefig('heat.pdf', format="pdf")
        
        plt.clf()
        
        preds = self.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
        return preds
    
    def multichoice(self, dataset, item, classifier):
    
        return NotImplementedError
    
