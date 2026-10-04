import os
import torch

from omnix import OMNIX, OMNIXConfig, ByteTokenizer


MODEL_PATH = "checkpoints/omnix-1.0-best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are OMNIX, an experimental AI language model. "
    "Answer clearly and helpfully.\n\n"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Train OMNIX first with: python train.py"
        )

    checkpoint = torch.load(
        MODEL_PATH,
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


def generate(model, tokenizer, prompt):
    tokens = tokenizer.encode(prompt)

    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        output = model.generate(
            x,
            max_new_tokens=160,
            temperature=0.8,
            top_k=50
        )

    generated = output[0].tolist()

    new_tokens = generated[len(tokens):]

    return tokenizer.decode(new_tokens)


def main():
    print("=" * 40)
    print("             OMNIX 1.0")
    print("          Experimental AI")
    print("=" * 40)

    print(f"Device: {DEVICE}")
    print("Type 'exit' to leave.")
    print()

    model = load_model()
    tokenizer = ByteTokenizer()

    history = SYSTEM_PROMPT

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user:
            continue

        if user.lower() in {
            "exit",
            "quit"
        }:
            print("Goodbye.")
            break

        prompt = (
            history
            + "User: "
            + user
            + "\nOMNIX: "
        )

        response = generate(
            model,
            tokenizer,
            prompt
        )

        response = response.split(
            "\nUser:",
            1
        )[0].strip()

        print(f"OMNIX: {response}")
        print()

        history += (
            "User: "
            + user
            + "\nOMNIX: "
            + response
            + "\n"
        )

        # Prevent the conversation from growing beyond
        # the model's context window.
        history_tokens = tokenizer.encode(history)

        if len(history_tokens) > 96:
            history_tokens = history_tokens[-96:]
            history = tokenizer.decode(
                history_tokens
            )


if __name__ == "__main__":
    main()
