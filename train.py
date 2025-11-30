import argparse
from utils.maker_loader import load_maker
from utils.model_loader import load_model

def main():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--dataset", required=True, help="which dataset you want to use")
    parser.add_argument("--decode-method", default='alw', help="vanilla, dola, alw")
    parser.add_argument("--pope", default=None, help="pope version")

    # llm
    parser.add_argument("--llm", required=True, help="Model name in configs/models.yaml")

    # classifier
    parser.add_argument("--classifier", default=None, help="Model name in configs/models.yaml")
    parser.add_argument("--tuned-path", default=None, help="path of tuned lms")

    # train
    parser.add_argument('--epoch', default=3, type=int, help='training epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='training batch size')
    parser.add_argument('--max-len', default=512, type=int, help='max length of sentence')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--warm-up', default=1000, type=int, help='warm up step')
    parser.add_argument('--save-every', default=30, type=int, help='warm up step')
    parser.add_argument('--print-every', default=30, type=int, help='warm up step')
    args = parser.parse_args()    

    llm, classifier = load_model(args) # load cls instance
    classifier.load()
    classifier.train()


if __name__ == "__main__":
    main()
