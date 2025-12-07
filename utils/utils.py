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
        shots_ids = shots['input_ids'][:, :-1]
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
        
        inputs = shots + q
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

        inputs = llm.processor(images=image, text=question, return_tensors="pt").to(0, torch.float16)
        question_text = llm.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
    return inputs, question_text

def prepare_input_for_tllm_train():
    return

def prepare_input_for_tllm_infer():
    return

def prepare_input_for_predictor(classifier, dataset):
    args = classifier.args
    
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
    smoothed_counts = sorted_counts + 1.0 

    weights = 1.0 / smoothed_counts
    weights = weights / np.sum(weights) 

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
