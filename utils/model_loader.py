from typing import Dict, Type
from models.base import BaseModel
from models import TextLLM, V2TLLM, Classifier

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "text_lm": TextLLM,
    "v2t": V2TLLM,
    "classifier": Classifier,
}


def load_model(args):
    probe = TextLLM(args)
    family = probe.family

    cls = MODEL_REGISTRY.get(family)
    if cls is None:
        raise ValueError(f"Unknown family '{family}' for model '{args.llm}'.")
    llm = cls(args)

    classifier = Classifier(args, llm) if args.decode_method == 'alw' else None

    return llm, classifier
