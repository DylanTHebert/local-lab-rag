from abc import ABC, abstractmethod
from typing import Dict

import transformers
import torch
from sentence_transformers import SentenceTransformer


class LocalEmbedding(ABC):
    @abstractmethod
    def __call__(self, msg: str | Dict[str, str]):
        pass


class FlagEmbedding(LocalEmbedding):
    """Wrapper around an opensource embedding with 512 token length"""

    # TODO compile this for possible speed ups
    def __init__(self, model_id: str = 'BAAI/bge-large-zh-v1.5'):
        self.model = SentenceTransformer(model_id)

    def __call__(self, msg: str | Dict[str, str]):
        # accepts List[str]
        output = self.model.encode(msg, normalize_embeddings=True)
        return output

class Sent384(LocalEmbedding):
    """Wrapper around an opensource embedding with 512 token length"""

    # TODO compile this for possible speed ups
    def __init__(self, model_id: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_id)

    def __call__(self, msg: str | Dict[str, str]):
        # accepts List[str]
        output = self.model.encode(msg, normalize_embeddings=True)
        return output
