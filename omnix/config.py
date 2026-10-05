from dataclasses import dataclass


@dataclass
class OMNIXConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    learning_rate: float = 0.01
    version: str = "1.0.0"
