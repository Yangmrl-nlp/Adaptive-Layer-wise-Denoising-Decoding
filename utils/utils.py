from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

def prepare_input_for_v2t_train(shot_dict, llm, item, args):
    question, answer, image_path = item['question'], item['answer'], item['image_path']
    shots, imgs = shot_dict['text'], shot_dict['imgs']
    if answer == 'yes':
        answer = "Yes"
    elif answer == 'no': 
        answer = "No"
    
    if args.llm == 'llava1.5_7b':
        qa = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": 'Question: ' + question},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ],
            }]

        q = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": 'Question: ' + question},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": "",
            }]

        shots = shots
        shotques = shots + q
        inputs = shots + qa

        shots = llm.processor.apply_chat_template(shots, add_generation_prompt=False)
        shotques = llm.processor.apply_chat_template(shotques, add_generation_prompt=False)
        inputs = llm.processor.apply_chat_template(inputs, add_generation_prompt=False)
        
        shots = llm.processor(images=imgs, text=shots, return_tensors='pt').to(0, torch.float16)
        shots_ids = shots['input_ids'][:, :-1] # llava的 chat template 会补一个空格，要删掉
        imgs = imgs + [Image.open(image_path).convert("RGB")]
        shotques = llm.processor(images=imgs, text=shotques, return_tensors='pt').to(0, torch.float16)
        shotques_ids = shotques['input_ids'][:, :-1]  
        inputs = llm.processor(images=imgs, text=inputs, return_tensors='pt').to(0, torch.float16)
        input_ids = inputs['input_ids'][:, :-1]

        answer_ids = input_ids[0, shotques_ids.shape[-1]: ]
        question_ids = input_ids[0, shots_ids.shape[-1]: shotques_ids.shape[-1]]

        assert input_ids.shape[-1] == shots_ids.shape[-1] + question_ids.shape[-1] + answer_ids.shape[-1]
        
    elif args.llm == 'instructblip_vicuna_7b':
        image = Image.open(image_path).convert("RGB")
        q = 'Is there a person in the image?'
        a = 'yes'
        shots = f"Q: {q}\nA: {a}\n\n"
        shotques = f"Q: {q}\nA: {a}\n\nQ: {question}\nA:"
        inputs = f"Q: {q}\nA: {a}\n\nQ: {question}\nA: {answer}"
    
        shots = llm.processor(images=image, text=shots, return_tensors='pt').to(0, torch.float16)
        shotques = llm.processor(images=image, text=shotques, return_tensors='pt').to(0, torch.float16)
        inputs = llm.processor(images=image, text=inputs, return_tensors="pt").to(0, torch.float16)
        
        shots_ids = shots['input_ids']
        shotques_ids = shotques['input_ids']
        input_ids = inputs['input_ids']
        # print(shotques_ids)
        # print(input_ids)
        answer_ids = input_ids[0, shotques_ids.shape[-1]: ]
        question_ids = input_ids[0, shots_ids.shape[-1]: shotques_ids.shape[-1]]

        assert input_ids.shape[-1] == shots_ids.shape[-1] + question_ids.shape[-1] + answer_ids.shape[-1]
    
    return inputs, question_ids, answer_ids

def prepare_input_for_v2t_infer(shot_dict, llm, item, args):
    question, answer, image_path = item['question'], item['answer'], item['image_path']
    shots, imgs = shot_dict['text'], shot_dict['imgs']
    if args.llm == 'llava1.5_7b' or args.llm == 'llavanext_8b':
        q = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": 'Question: ' + question},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": "",
            }]
        
        #print(f"debug: {imgs}")
        inputs = shots + q
        #print(f"debug: {shots}")
        inputs = llm.processor.apply_chat_template(inputs, add_generation_prompt=False)
        question = llm.processor.apply_chat_template(q, add_generation_prompt=False)
        question = llm.processor(images=[Image.open(image_path).convert("RGB")], text=question, return_tensors='pt').to(0, torch.float16)
        question_text = llm.tokenizer.decode(question['input_ids'][0], skip_special_tokens=True)
        
        imgs = imgs + [Image.open(image_path).convert("RGB")]
        inputs = llm.processor(images=imgs, text=inputs, return_tensors='pt').to(0, torch.float16)
    
    elif args.llm == 'instructblip_vicuna_7b':
        image = Image.open(image_path).convert("RGB")
        if args.dataset != 'chair':
            q = 'Is there a person in the image?'
            a = 'yes'
            question = f"Q: {q}\nA: {a}\n\nQ: {question}\nA:"
            #print(question)
        inputs = llm.processor(images=image, text=question, return_tensors="pt").to(0, torch.float16)
        question_text = llm.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        #print(question_text)
        
    return inputs, question_text

def prepare_input_for_tllm_train():
    return

def prepare_input_for_tllm_infer():
    return

def prepare_input_for_predictor(classifier, dataset):
    args = classifier.args
    # 因为数据集中有-1，所以要将-1,0,1~16映射到0,1,2~17给分类器训练，和llm推理时反向映射即可
    label_to_int = {label: i for i, label in enumerate(range(-1, classifier.num_labels-1))}

    def collate_fn(data):
        context = [i['context'].strip() for i in data]
        label = [label_to_int[i['best_layer']] for i in data]

        inputs = classifier.tokenizer.batch_encode_plus(batch_text_or_text_pairs=context, truncation=True,
                        padding='max_length', max_length=args.max_len, return_tensors='pt').to('cuda')

        inputs['labels'] = torch.tensor(label).to('cuda')
        inputs['context'] = context
        
        return inputs

    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    layer_counts = Counter(dataset['best_layer'])

    sorted_counts = np.array([layer_counts.get(i, 0) for i in list(label_to_int.keys())], dtype=float)
    smoothed_counts = sorted_counts + 1.0 # 避免除0

    weights = 1.0 / smoothed_counts
    weights = weights / np.sum(weights)  # 归一化

    return dataloader, torch.from_numpy(weights).float().cuda()

def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh

def get_best_layer(args, classifier, question_text):
    if args.decode_method == 'vanilla':
        return None

    elif args.decode_method == 'dola':
        # TODO: dynamic dola 根据js散度算的，不复杂。可以参考dola源码，这里返回0是dola的静态版本
        if args.dola == 'static':
            return 0
            

    elif args.decode_method == 'alw':
        int_to_label = {i: label for i, label in enumerate(range(-1, classifier.num_labels))}
        classifier.model.eval()

        with torch.no_grad():
            inputs = classifier.tokenizer(question_text, 
                                        truncation=True,
                                        max_length=args.max_len, 
                                        return_tensors='pt').to('cuda')
            
            outputs = classifier.model(input_ids=inputs['input_ids'], 
                                    attention_mask=inputs['attention_mask'])

            classify_prob = outputs.logits.softmax(dim=-1)
            pred = torch.argmax(classify_prob, dim=-1).item()
            return int_to_label[pred]
    else:
        return -1
    
    
    # elif mode == 'dola':
    #     premature_layer_dist = {l:0 for l in candidate_premature_layers}
    #     picked_logits = []
    #     result_dict = {}
    #     premature_layers = []
    #     classifier.model.eval()
    #     with torch.no_grad():
    #         inputs = classifier.tokenizer(question_text, 
    #                                 truncation=True,
    #                                 max_length=args.max_len, 
    #                                 return_tensors='pt').to('cuda')
                
    #         outputs = classifier.model(input_ids=inputs['input_ids'], 
    #                             attention_mask=inputs['attention_mask'])
    #     dict_outputs, outputs = self.model(
    #         input_ids=input_ids,
    #         return_dict=True,
    #         output_attentions=False,
    #         output_hidden_states=False,
    #         early_exit_layers=candidate_premature_layers + [mature_layer],
    #     )

    #     for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
    #         # Pick the less like layer to contrast with
    #         # 1. Stacking all premature_layers into a new dimension
    #         stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

    #         # 2. Calculate the softmax values for mature_layer and all premature_layers
    #         softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
    #         softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

    #         # 3. Calculate M, the average distribution
    #         M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

    #         # 4. Calculate log-softmax for the KL divergence
    #         log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
    #         log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

    #         # 5. Calculate the KL divergences and then the JS divergences
    #         kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
    #         kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
    #         js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    #         # 6. Reduce the batchmean
    #         js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
    #         premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
    #         premature_layer_dist[premature_layer] += 1

    #         premature_layers.append(premature_layer)