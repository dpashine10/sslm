"""Generate text from one prompt using a trained SSM checkpoint."""

from __future__ import annotations

import argparse
import time

from .common import decode, load_model_and_tokenizer, prompt_ids


def generate_text(
    checkpoint: str,
    tokenizer: str,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    do_sample: bool = True,
    device: str = "auto",
) -> tuple[str, float]:
    model, tokenizer_object, selected_device, _ = load_model_and_tokenizer(
        checkpoint, tokenizer, device=device
    )
    inputs = prompt_ids(tokenizer_object, prompt, model.config.max_seq_len, selected_device)
    started = time.perf_counter()
    output = model.generate(
        inputs,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=tokenizer_object.token_to_id("<eos>"),
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    generated_tokens = output.size(1) - inputs.size(1)
    return decode(tokenizer_object, output[0]), generated_tokens / elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a completion from an SSM LM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text, speed = generate_text(
        args.checkpoint,
        args.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device,
    )
    print(text)
    print(f"\nGenerated at {speed:.1f} tokens/s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
