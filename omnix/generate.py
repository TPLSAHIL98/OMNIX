import argparse
import torch

from omnix import OMNIX, OMNIXConfig, ByteTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    config = OMNIXConfig(
        **checkpoint["config"]
    )

    model = OMNIX(config)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="checkpoints/omnix-1.0-final.pt"
    )

    parser.add_argument(
        "--prompt",
        default="OMNIX is"
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=100
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    args = parser.parse_args()

    tokenizer = ByteTokenizer()

    model = load_model(args.model)

    encoded = tokenizer.encode(args.prompt)

    x = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    output = model.generate(
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=50
    )

    text = tokenizer.decode(
        output[0].tolist()
    )

    print("\n" + text)


if __name__ == "__main__":
    main()
