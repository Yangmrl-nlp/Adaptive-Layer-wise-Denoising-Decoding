import json,os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict


eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        for i in range(0, len(l), n):
            yield l[i:i + n]
        return

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = { "yes": 1, "no": 0, "other": -1 }
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)
        clean_gts, clean_preds = [], []
        other_num = 0

        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        return {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "precision": precision, "recall": recall,
            "other_num": other_num, "acc": acc,
        }

    def process_result(self, jsonl_file):
        # 读取单个 JSONL 文件
        print(f"Reading from: {jsonl_file}")
        data = [json.loads(l) for l in open(jsonl_file, "r", encoding="utf-8").readlines()]
        
        # 按 category 分组（转为小写以统一处理）
        category_data = defaultdict(list)
        for item in data:
            category = item.get("category", "unknown").lower()
            category_data[category].append(item)
        
        print(f"Found {len(category_data)} categories: {list(category_data.keys())}\n")

        # 用于计算总分
        total_scores = 0
        
        # 按 Perception 和 Cognition 分类处理
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")

            scores = 0
            task_score_dict = {}

            for task_name in task_name_list:
                # 转为小写查找
                task_name_lower = task_name.lower()
                if task_name_lower not in category_data:
                    print(f"[WARNING] Missing category: {task_name}, skip.")
                    continue

                task_items = category_data[task_name_lower]
                print(f"Processing {task_name}: {len(task_items)} items")

                # 构造 lines 格式
                lines = []
                for item in task_items:
                    img_name = item["image"]
                    question = item["prompt"]
                    gt_ans = item["answer"]
                    pred_ans = item["text"]
                    lines.append(f"{img_name}\t{question}\t{gt_ans}\t{pred_ans}")

                chunk_lines = list(self.divide_chunks(lines))
                img_num = len(chunk_lines)

                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts, preds = [], []

                for img_items in chunk_lines:
                    if len(img_items) != 2:
                        print(f"[WARNING] Image has {len(img_items)} questions instead of 2, skip.")
                        continue
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")
                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        pred_ans = self.parse_pred_ans(pred_ans)

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        if pred_ans == "other":
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                metric_dict = self.compute_metric(gts, preds)
                metric_dict["acc_plus"] = acc_plus_correct_num / img_num

                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v * 100

                task_score_dict[task_name] = task_score
                scores += task_score

            print("total score:", scores, "\n")
            for name, score in task_score_dict.items():
                print("\t", name, "score:", score)
            print("\n")
            
            # 累加到总分
            total_scores += scores
        
        # 输出最终总分
        print("=" * 50)
        print(f"MME Total Score: {total_scores}")
        print("=" * 50)


if __name__ == "__main__":
    method = 'icd'
    if method == 'vcd':
        model = 'instructblip'
        jsonl_file = f"/mnt/data1/zjx/project/alw/vcd/result/zero-shot/{model}/mme/mme_answers_cd_seed55.jsonl"
        cal = calculate_metrics()
        cal.process_result(jsonl_file)
    else:
        model = 'instructblip'
        result_root = f'/mnt/data1/zjx/project/alw/icd/experiments/result/{model}/mme'
        for name in os.listdir(result_root):
            full_path = os.path.join(result_root, name)
            jsonl_file = os.path.join(full_path,"results.jsonl")
            cal = calculate_metrics()
            cal.process_result(jsonl_file)