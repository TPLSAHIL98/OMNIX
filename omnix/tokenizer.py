class ByteTokenizer:
    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return 256
