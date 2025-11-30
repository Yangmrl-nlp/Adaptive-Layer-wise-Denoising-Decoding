import pandas as pd
import math
import json

def recorder(s:str) -> str:
    NEG_WORDS = ["No", "not", "no", "NO"]

    s = s.replace('.', '')
    s = s.replace(',', '')
    words = s.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        s = "yes"
    else:
        s = "no"
    return s

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    return s

def F1(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
        
    return f1

def TextVqa_score(pred,gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = [normalize_text(answer).split() for answer in gold]
    num_same = 0
    #print(pred_tokens)
    #print(gold_tokens)
    for i in range(len(gold_tokens)):
        if gold_tokens[i] == pred_tokens:
            num_same+=1   
    return min(num_same/3,1)
    
def ACC(json_path):
    correct = 0
    total = 0

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        answer = str(item.get('answer', '')).lower().strip()
        predict = str(item.get('text', '')).lower().strip()
        print(answer,predict)
        total += 1
        if answer in predict:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(accuracy)
    return accuracy

if __name__ == "__main__":
    raise NotImplementedError