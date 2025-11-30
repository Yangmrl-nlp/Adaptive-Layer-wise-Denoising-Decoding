# models/text_model.py
import torch
from typing import Optional, Dict, Any
from .base import BaseModel

def _import_llama():
    from transformers import LlamaTokenizer, LlamaForCausalLM
    return LlamaTokenizer, LlamaForCausalLM

def _import_auto():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoTokenizer, AutoModelForCausalLM

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

class TextLLM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        name = self.args.llm
        family, cfg = self.resolve_model_cfg(name)
        self.family = family
        self.name = name
        self.cfg_dict = cfg
        self.classifier = None

    def load(self):
        arch = self.cfg_dict.get("arch", "auto")
        pretrained = self.cfg_dict["pretrained"]
        torch_dtype = _DTYPE_MAP.get(self.cfg_dict.get("torch_dtype", "float16"), torch.float16)

        print(f"[TextLLM] Loading {self.name} (arch={arch}) from '{pretrained}' ...")

        if arch == "llama":
            LlamaTokenizer, LlamaForCausalLM = _import_llama()
            self.tokenizer = LlamaTokenizer.from_pretrained(pretrained, use_fast=False, padding_side="right")
            self.model = LlamaForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map='auto')

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = 1

        else:  # "auto"
            AutoTokenizer, AutoModelForCausalLM = _import_auto()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False, padding_side="right")
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map='auto')

        self.model.eval()
        print("[TextLLM] Success.\n")

    def generate(self, dataset, row, classifier):
        ...
        
    def multichoice(self, prompt, true_answer, false_answers):
        ...