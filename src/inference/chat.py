"""Small terminal chatbot wrapper around the recurrent SSM decoder."""

from __future__ import annotations

import argparse
import time

from .common import decode, load_model_and_tokenizer, prompt_ids


EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def conversation_prompt(history: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for role, text in history:
        parts.append(f"<|{role}|>\n{text}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a trained SSM LM in a terminal.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, tokenizer, device, _ = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device=args.device)
    history: list[tuple[str, str]] = []
    eos_id = tokenizer.token_to_id("<eos>")
    print("SSM chat. Type /exit to leave. History is only the context in this process.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            break
        history.append(("user", user_text))
        text = conversation_prompt(history)
        inputs = prompt_ids(tokenizer, text, model.config.max_seq_len, device)
        started = time.perf_counter()
        output = model.generate(
            inputs,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            eos_token_id=eos_id,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        completion = decode(tokenizer, output[0, inputs.size(1) :]).strip()
        # Stop before a generated next-turn marker when the model learned one.
        for marker in ("<|user|>", "<|system|>"):
            completion = completion.split(marker, 1)[0].strip()
        history.append(("assistant", completion))
        # Keep the stored history bounded by tokenized context rather than an
        # arbitrary number of turns.  The next prompt is still hard-trimmed.
        while len(history) > 1 and len(tokenizer.encode(conversation_prompt(history)).ids) > model.config.max_seq_len:
            history.pop(0)
            if history and history[0][0] == "assistant":
                history.pop(0)
        print(f"Assistant: {completion}\n[{(output.size(1) - inputs.size(1)) / elapsed:.1f} tokens/s]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
