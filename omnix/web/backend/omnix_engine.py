import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnix import (
    OMNIX,
    OMNIXConfig,
    ByteTokenizer,
)


class OMNIXEngine:
    def __init__(self):
        self.config = OMNIXConfig()
        self.tokenizer = ByteTokenizer()
        self.model = OMNIX(self.config)

    def generate(self, message: str):
        tokens = self.tokenizer.encode(message)

        output = self.model.generate(
            tokens,
            max_new_tokens=120,
            temperature=0.8,
        )

        new_tokens = output[len(tokens):]

        return self.tokenizer.decode(
            new_tokens
        )
