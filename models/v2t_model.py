import torch
from typing import Optional, Dict, Any
import math


from .base import BaseModel
from utils.utils import *
import torch.nn.functional as F
# from MiniGPT_4.load_minigpt4 import load_model
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
    # processor = InstructBlipProcessor.from_pretrained('/mnt/data1/zjx/project/vcd/experiments/checkpoints/InstructBLIP', use_fast=False)
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
            self.model = LlavaForConditionalGeneration.from_pretrained(pretrained, dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
            
        elif self.args.llm == 'instructblip_vicuna_7b':
            self.model = InstructBlipForConditionalGeneration.from_pretrained(pretrained, dtype=torch_dtype,low_cpu_mem_usage=True, device_map='auto')
        elif self.args.llm == 'llavanext_8b':
            self.model = LlavaNextForConditionalGeneration.from_pretrained(pretrained, dtype=torch_dtype,low_cpu_mem_usage=True, device_map='auto')
        
        self.processor = AutoProcessor.from_pretrained(pretrained, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.model.eval()
        # print(self.model)
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
      #print(l,r)
      return l,r
      
    def visionmask(self, hidden, best_layer,input_ids):
        
        #hidden[:, self.vis_start:self.vis_end+1, :] = 0
        #print(self.model)
        l,r = self.get_lr(input_ids)
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          q_proj = self.model.language_model.layers[best_layer].self_attn.q_proj
          k_proj = self.model.language_model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
          #print(k.shape)
          #print(q.shape)
          if k.shape[-1] != q.shape[-1]:
            k = torch.nn.functional.linear(k, q_proj.weight[:, :1024])
          attn_score = (q @ k.transpose(-1,-2))
          #print(attn_score.shape)
        elif self.args.llm == 'instructblip_vicuna_7b':
          q_proj = self.model.language_model.model.layers[best_layer].self_attn.q_proj
          k_proj = self.model.language_model.model.layers[best_layer].self_attn.k_proj
          q = q_proj(hidden)
          k = k_proj(hidden)
          attn_score = (q @ k.transpose(-1,-2))
        #print(f"debug: {attn_score.shape}")
        #print(f"debug: {attn_score[0,1,:]}")
        token_score = attn_score.mean(dim=(0, 1))   # shape: [seq_len]

        sorted_indices = torch.argsort(token_score, descending=False)
        cnt = 0
        l,r = self.get_lr(input_ids)
        for idx in sorted_indices:
          if idx >= l and idx <= r:
            hidden[:,idx,:] = 0
            cnt+=1
            #print("debug: ",sorted_indices[idx])
            if cnt == math.ceil((r-l+1) * 0.1):
              break
        
        #print("avg_attn_score: ", token_score)
        #print("token_indices: ", sorted_indices)
        if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
          layers = self.model.language_model.layers[best_layer:]
          rotary_emb = self.model.language_model.rotary_emb
        elif self.args.llm == 'instructblip_vicuna_7b':
          layers = self.model.language_model.model.layers[best_layer:]
          rotary_emb = self.model.language_model.model.rotary_emb
  
        for i, layer in enumerate(layers):
            seq_len = hidden.shape[1]
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # (1, seq_len)
            cos, sin = rotary_emb(hidden, position_ids)
            hidden = layer(hidden, position_embeddings=(cos, sin))
            #hidden[:, self.vis_start:self.vis_end-1, :] = 0
      
        #hidden = self.model.language_model.norm(hidden)
        head_layer = self.model.get_output_embeddings()
        logits = head_layer(hidden)
        return logits[0,-1,:]
    
    def generate(self, dataset, item, classifier):
        shot_dict = dataset._create_shot()
        inputs, question_text = prepare_input_for_v2t_infer(shot_dict, self, item,self.args)
        input_ids_all = inputs['input_ids']
        #print(input_ids_all[0][3700:4316])
        max_new_tokens = 512
        past_key_values = None
        new_tokens_list = []
        # pre_list = []
        # for best_layer in range(-1,25):
        for _ in range(max_new_tokens):
            # if self.args.model == 'Qwen_VL_7b':
            #   ouputs = self.model.generate()
            # else:
            outputs = self.model(**inputs, 
                                output_hidden_states=True, 
                                use_cache=True,
                                past_key_values=past_key_values,
                                return_dict=True)
            
            if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
              logits = outputs.hidden_states[1:] # llava返回的logits有embedding层 (len(logits)=33)，要去掉，只考虑decoder层
              past_key_values = outputs.past_key_values
            elif self.args.llm == 'instructblip_vicuna_7b':
              logits = outputs.language_model_outputs.hidden_states[1:]
              past_key_values = outputs.language_model_outputs.past_key_values
              
            mature_layer = len(logits)-1
            head_layer = self.model.get_output_embeddings()
            final_logits = head_layer(logits[mature_layer])[:, -1, :]
            final_logits = final_logits.log_softmax(dim=-1)

            best_layer = get_best_layer(self.args, classifier, question_text)
            
            #best_layer = 6
            #print(best_layer)
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
                  M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)
                # 4. Calculate log-softmax for the KL divergence
                  log_softmax_mature_layer = torch.log_softmax(logits[mature_layer][:, -1, :], dim=-1)  # shape: (batch_size, num_features)
                  log_softmax_premature_layers = torch.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                  # 5. Calculate the KL divergences and then the JS divergences
                  kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                  kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                  js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
                # 6. Reduce the batchmean
                  js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                  #print(js_divs)
                  best_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                  #print(f'debug best_layer: {best_layer}')
                
                base_logits = head_layer(logits[best_layer])[:,-1,:]
                base_logits = base_logits.log_softmax(dim=-1)
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                next_token_logits = final_logits - base_logits
            
            else:
                # plot_token_prob_bar(final_logits,"final")
                # base_logits = head_layer(logits[best_layer])[:,-1,:]
                # base_logits = base_logits.log_softmax(dim=-1)
                # plot_token_prob_bar(base_logits,"original")
                #print(best_layer)
                if self.args.Prune == 'True':
                  base_logits = self.visionmask(logits[best_layer-1],best_layer,input_ids_all[0])
                else:
                  base_logits = head_layer(logits[best_layer])[:,-1,:]
                # plot_token_prob_bar(base_logits,"purning")
                base_logits = base_logits.log_softmax(dim=-1)
                #print(best_layer)
                # plot_token_prob_bar(base_logits,"purning")
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                
                if self.args.Prune == 'True':   
                  # print("yes")
                  next_token_logits = final_logits + base_logits
                  # plot_token_prob_bar(next_token_logits,"prune")
                else:
                  next_token_logits = final_logits - base_logits
                #next_token_logits = next_token_logits.log_softmax(dim=-1)

            input_ids_all = input_ids_all.to(next_token_logits.device)
            #print(f"debug shape: {next_token_logits.shape}")
            #next_token_logits = next_token_logits.unsqueeze(0)
            next_token_logits = self.processors(input_ids_all, next_token_logits)
            #print(next_token_logits.shape)
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            new_tokens_list.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # new_tokens_list, stop = if_stop(new_tokens_list, stopping_words) # TODO: 传入某些数据集的停止符
            # if stop:
            #     break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([inputs['attention_mask'], inputs['attention_mask'].new_ones((inputs['attention_mask'].shape[0], 1))], dim=-1)
            #print(f"debug attention: {attention_mask.shape}")
            inputs['attention_mask'] = attention_mask
            #print(input_ids)
            #print(inputs['input_ids']) torch.cat((inputs['input_ids'],input_ids),dim = -1)
            #print(type(inputs['input_ids']))
            
            inputs['input_ids'] = input_ids
            #print(f"debug: {self.tokenizer.decode(input_ids[0][0])}")
            if self.args.llm == 'llava1.5_7b' or self.args.llm == 'llavanext_8b':
              inputs["pixel_values"] = None

            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)
            question_text = question_text + self.tokenizer.decode(next_token)

        preds = self.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
        #pre_list.append(preds)
        return preds
    
    def multichoice(self, dataset, item, classifier):
    
        return NotImplementedError
    
'''
shot1:
    question: "what number is shown at the bottom?"
    img_path: "/mnt/data1/yangmrl/ALW_debug/data/raw_data/textvqa_data/images/train/0a0bc91825468c45.jpg"
    answer: "30"
    
shot2:
    question: "what does the woman's shirt say?"
    img_path: "/mnt/data1/yangmrl/ALW_debug/data/raw_data/textvqa_data/images/train/00a0d2280595043f.jpg"
    answer: "digg"

shot3:
    question: "what is the name of the bank?"
    img_path: "/mnt/data1/yangmrl/ALW_debug/data/raw_data/textvqa_data/images/train/00a0db6495982c1d.jpg"
    answer: "keybank"
    
LlavaForConditionalGeneration(
  (model): LlavaModel(
    (vision_tower): CLIPVisionModel(
      (vision_model): CLIPVisionTransformer(
        (embeddings): CLIPVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
          (position_embedding): Embedding(577, 1024)
        )
        (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-23): 24 x CLIPEncoderLayer(
              (self_attn): CLIPAttention(
                (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              )
              (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (multi_modal_projector): LlavaMultiModalProjector(
      (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
      (act): GELUActivation()
      (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
    )
    (language_model): LlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((4096,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
)
'''