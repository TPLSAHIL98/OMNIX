import os
import math
import torch

from omnix import OMNIX, OMNIXConfig, ByteTokenizer


DATASET = "dataset/train.txt"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
MAX_ITERS = 5000
LEARNING_RATE = 3e-4

EVAL_INTERVAL = 250
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    with open(DATASET, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = ByteTokenizer()

    tokens = tokenizer.encode(text)

    data = torch.tensor(tokens, dtype=torch.long)

    return data, tokenizer


def get_batch(data, block_size):
    if len(data) <= block_size + 1:
        raise ValueError(
            "Dataset is too small for the configured block size."
        )

    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()

    losses = []

    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, model.config.block_size)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def main():
    print("================================")
    print("          OMNIX 1.0")
    print("================================")
    print(f"Device: {DEVICE}")

    data, tokenizer = load_dataset()

    config = OMNIXConfig(
        vocab_size=tokenizer.vocab_size
    )

    model = OMNIX(config).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
    print(f"Training tokens: {len(data):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for iteration in range(MAX_ITERS):

        if iteration % EVAL_INTERVAL == 0:
            loss = estimate_loss(model, data)

            perplexity = math.exp(
                min(loss, 20)
            )

            print(
                f"step {iteration:5d} | "
                f"loss {loss:.4f} | "
                f"ppl {perplexity:.2f}"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "step": iteration,
                },
                f"{CHECKPOINT_DIR}/omnix-1.0-step-{iteration}.pt"
            )

        x, y = get_batch(
            data,
            config.block_size
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": MAX_ITERS,
        },
        f"{CHECKPOINT_DIR}/omnix-1.0-final.pt"
    )

    print("Training complete.")
    print("Saved: checkpoints/omnix-1.0-final.pt")


if __name__ == "__main__":
    main()
