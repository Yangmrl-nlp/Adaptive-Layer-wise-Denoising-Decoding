import argparse
from utils.maker_loader import load_maker
from utils.model_loader import load_model

def main():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--dataset", required=True, help="which dataset you want to use")
    parser.add_argument("--decode-method", default=None, help="vanilla, dola, alw")
    parser.add_argument("--pope", default=None, help="pope version")
    parser.add_argument("--Prune", default=None, help="whether Prune")

    # llm
    parser.add_argument("--llm", required=True, help="Model name in configs/models.yaml")

    args = parser.parse_args()    

    Maker = load_maker(args)
    Maker()


if __name__ == "__main__":
    main()
