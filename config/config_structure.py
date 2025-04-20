from dataclasses import dataclass

@dataclass
class DataPrepConfig:
    input_dir: str
    out_put_dir: str
    embedding: str
    output_islocal: bool = False
