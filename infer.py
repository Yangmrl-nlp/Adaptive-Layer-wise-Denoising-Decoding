import argparse
import glob
import os
from utils.maker_loader import load_maker
from utils.model_loader import load_model
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--dataset", required=True, help="which dataset you want to use")
    parser.add_argument("--pope", default=None, help="pope version")
    parser.add_argument("--dola", default=None, help="dola version")
    parser.add_argument("--decode-method", default='alw', help="vanilla, dola, alw")
    parser.add_argument("--Prune", default=None, help="whether Prune")

    # llm
    parser.add_argument("--llm", required=True, help="Model name in configs/models.yaml")

    # classifier
    parser.add_argument("--classifier", default=None, help="Model name in configs/models.yaml")
    parser.add_argument("--max-len", default=512, help="max len for roberta")
    parser.add_argument("--tuned-path", default=None, help="path of tuned lms")
    parser.add_argument("--tuned-list", default=None, help="list of tuned lms")

    args = parser.parse_args()    
    llm, classifier = load_model(args) # load cls instance
    llm.load()
    
    if classifier:
        
        if args.tuned_list:
            pth_files = glob.glob(os.path.join(args.tuned_list, '*.pth'))
            for file_path in pth_files:
                if args.pope:
                    print(f"running method: {args.decode_method} dataset: {args.pope}...")
                else:
                    print(f"running method: {args.decode_method} dataset: {args.dataset}...")
                tp_pth = Path(file_path)
                # if int(tp_pth.stem) >=200 and int(tp_pth.stem)<=900:
                args.tuned_path = os.path.abspath(file_path)
                classifier.load(args.tuned_path)
                llm.infer(classifier)
                llm.evaluate(f'./results/{args.llm}/{args.dataset}/ALW/{tp_pth.stem}')
        else:
            if args.pope:
               print(f"running method: {args.decode_method} dataset: {args.pope}...")
            else:
               print(f"running method: {args.decode_method} dataset: {args.dataset}...")
            classifier.load(args.tuned_path)
            llm.infer(classifier)
            tp_pth = Path(args.tuned_path)
            llm.evaluate(f'./results/{args.llm}/{args.dataset}/ALW/{tp_pth.stem}')

    else:
        if args.pope:
           print(f"running method: {args.decode_method} dataset: {args.pope}...")
        else:
            print(f"running method: {args.decode_method} dataset: {args.dataset}...")
        llm.infer(classifier)
        llm.evaluate()

if __name__ == "__main__":
    main()
