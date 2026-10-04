import os
import math
import random

import torch

from omnix import OMNIX, OMNIXConfig, ByteTokenizer


DATASET = "dataset/train.txt"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

EVAL_INTERVAL = 250
EVAL_ITERS = 50

TRAIN_SPLIT = 0.9

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    with open(DATASET, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) < 1000:
        print(
            "WARNING: dataset contains very few tokens. "
            "OMNIX will train, but the model will be extremely limited."
        )

    data = torch.tensor(tokens, dtype=torch.long)

    split = int(len(data) * TRAIN_SPLIT)

    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data, tokenizer


def get_batch(data, block_size):
    if len(data) <= block_size + 1:
        raise ValueError(
            "Dataset split is too small for the configured block size."
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
def estimate_loss(model, train_data, val_data):
    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = []

        for _ in range(EVAL_ITERS):
            x, y = get_batch(
                data,
                model.config.block_size
            )

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


def save_checkpoint(model, config, optimizer, step, best_val):
    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "version": "1.0.0",
    }

    torch.save(
        checkpoint,
        f"{CHECKPOINT_DIR}/omnix-1.0-latest.pt"
    )


def main():
    set_seed(SEED)

    print("=" * 40)
    print("             OMNIX 1.0")
    print("=" * 40)

    print(f"Device: {DEVICE}")

    train_data, val_data, tokenizer = load_data()

    config = OMNIXConfig(
        vocab_size=tokenizer.vocab_size
    )

    model = OMNIX(config).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_ITERS
    )

    best_val = float("inf")

    for step in range(MAX_ITERS):

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            perplexity = math.exp(
                min(val_loss, 20)
            )

            print(
                f"step {step:5d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"ppl {perplexity:.2f}"
            )

            save_checkpoint(
                model,
                config,
                optimizer,
                step,
                best_val
            )

            if val_loss < best_val:
                best_val = val_loss

                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config.__dict__,
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_val": best_val,
                        "version": "1.0.0",
                    },
                    f"{CHECKPOINT_DIR}/omnix-1.0-best.pt"
                )

        x, y = get_batch(
            train_data,
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
        scheduler.step()

    save_checkpoint(
        model,
        config,
        optimizer,
        MAX_ITERS,
        best_val
    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "optimizer": optimizer.state_dict(),
            "step": MAX_ITERS,
            "best_val": best_val,
            "version": "1.0.0",
        },
        f"{CHECKPOINT_DIR}/omnix-1.0-final.pt"
    )

    print()
    print("Training complete.")
    print("Saved:")
    print("  checkpoints/omnix-1.0-final.pt")
    print("  checkpoints/omnix-1.0-best.pt")


if __name__ == "__main__":
    main()
