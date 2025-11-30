# models/config.py
from typing import Dict, Tuple, Any

class Config:
    def __init__(self, cfg: Dict):
        self._cfg = cfg

    def find_by_name(self, name: str) -> Tuple[str, Dict[str, Any]]:
        for family in self._cfg.keys():
            group = self._cfg.get(family, {})
            if name in group:
                cfg = dict(group[name])
                return family, cfg

        raise KeyError(f"'{name}' not found in families: {list(self._cfg.keys())}")