from abc import ABC, abstractmethod
from typing import Dict

import transformers
import torch

class LocalLLM(ABC):
    @abstractmethod
    def __call__(self, msg: str | Dict[str, str]):
        pass


class LocalLlama1B(LocalLLM):
    """Wrapper around basic llama implementation"""

    # TODO compile this for possible speed ups
    def __init__(self, model_id: str = None):
        if model_id is None:
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __call__(self, msg: str | Dict[str, str]):
        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": "Where's the treasure?"},
        # ]
        outputs = self.pipeline(
            msg,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]
