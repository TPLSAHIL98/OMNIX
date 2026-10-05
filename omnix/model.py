import numpy as np


class OMNIX:
    def __init__(self, config, seed=42):
        self.config = config

        rng = np.random.default_rng(seed)

        self.embedding = (
            rng.normal(
                0,
                0.02,
                (
                    config.vocab_size,
                    config.hidden_size
                )
            ).astype(np.float32)
        )

        self.output = (
            rng.normal(
                0,
                0.02,
                (
                    config.hidden_size,
                    config.vocab_size
                )
            ).astype(np.float32)
        )

        self.bias = np.zeros(
            config.vocab_size,
            dtype=np.float32
        )

    def _softmax(self, logits):
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities)

    def predict_next(self, token):
        hidden = self.embedding[token]

        logits = (
            hidden @ self.output
            + self.bias
        )

        return self._softmax(logits)

    def generate(
        self,
        tokens,
        max_new_tokens=120,
        temperature=0.8
    ):
        tokens = list(tokens)

        for _ in range(max_new_tokens):
            probabilities = self.predict_next(
                tokens[-1]
            )

            temperature = max(
                temperature,
                0.05
            )

            probabilities = np.log(
                np.clip(
                    probabilities,
                    1e-9,
                    1.0
                )
            ) / temperature

            probabilities -= np.max(
                probabilities
            )

            probabilities = np.exp(
                probabilities
            )

            probabilities /= np.sum(
                probabilities
            )

            next_token = int(
                np.random.choice(
                    len(probabilities),
                    p=probabilities
                )
            )

            tokens.append(next_token)

        return tokens
