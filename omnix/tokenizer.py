class ByteTokenizer:
    @property
    def vocab_size(self):
        return 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(
            int(token) % 256 for token in tokens
        ).decode("utf-8", errors="replace")
