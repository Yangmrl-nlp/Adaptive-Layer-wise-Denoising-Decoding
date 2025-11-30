import torch
from typing import Optional, Dict, Any
from .base import BaseModel
from utils.dataset_loader import load_dataset
from utils.utils import prepare_input_for_predictor
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os


def _import_roberta():
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    return RobertaTokenizer, RobertaForSequenceClassification


class Classifier(BaseModel):
    def __init__(self, args, llm_cls):
        super().__init__(args)
        family, cfg = self.resolve_model_cfg(args.classifier)
        self.family = family             
        self.name = args.classifier
        self.cfg_dict = cfg
        self.num_labels = llm_cls.cfg_dict.get("layer", 32) // 4 * 3 + 2

        # TODO
        # self.save_path = f'./ckpts/{self.args.llm}/{self.args.dataset}/'
        self.save_path = f'./ckpts/{self.args.llm}/{args.classifier}/{self.args.dataset}/'

    def load(self, tuned_path=None):
        arch = self.cfg_dict.get("arch", "RobertaForMaskedLM")
        pretrained = self.cfg_dict["pretrained"]

        print(f"[Classifier] Loading {self.name} (arch={arch}) from '{pretrained}' ...")

        if arch == 'RobertaForMaskedLM':
            RobertaTokenizer, RobertaForSequenceClassification = _import_roberta()
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained)
            self.model = RobertaForSequenceClassification.from_pretrained(pretrained, num_labels=self.num_labels)

            if tuned_path:
                print(f"[Classifier] Replacing parameter from '{tuned_path}' ...")
                self.model.load_state_dict(torch.load(tuned_path))

            self.model.cuda()

            print("[Classifier] Success.\n")

    def train(self):
        dataset = load_dataset(self.args)
        
        train_set = dataset._load_labeled_split('train')
        train_set, weights = prepare_input_for_predictor(self, train_set)

        # set optimizer, scheduler and loss function
        total_steps = len(train_set) * self.args.epoch
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-7)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warm_up, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        
        # train
        self.model.train()
        print('start training...')

        step = 0
        total_loss = 0.0

        for epoch in range(self.args.epoch):
            pbar = tqdm(train_set, desc=f"Epoch: {epoch+1}")

            for batch in pbar:
                optimizer.zero_grad()

                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = loss_fn(outputs.logits, batch['labels'])

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                step += 1

                if step % self.args.print_every == 0:
                    avg_loss = total_loss / self.args.print_every
                    pbar.set_description(f"Epoch: {epoch+1}, Step: {step}, Avg Loss: {avg_loss:.4f}")
                    total_loss = 0.0

                # save
                if step % self.args.save_every == 0:
                    template = 'lr-epoch-bs-{:}-{:}-{:}'
                    current_dict = os.path.join(self.save_path, self.args.pope,template.format(self.args.lr, self.args.epoch, self.args.batch_size))

                    if not os.path.exists(current_dict):
                        os.makedirs(current_dict)

                    torch.save(self.model.state_dict(), os.path.join(current_dict, f"{str(step)}.pth"))
        print('Training success. \n')
