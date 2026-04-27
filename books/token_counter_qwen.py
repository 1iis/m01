#!/usr/bin/env python3
"""
Qwen3.6 Token Counter - Original (with warning silenced)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Union

import warnings
# ←←← ADD THESE TWO LINES
warnings.filterwarnings("ignore", message="Token indices sequence length is longer")

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers", file=sys.stderr)
    sys.exit(1)


def get_tokenizer():
    """Load Qwen3 tokenizer (cached after first use)."""
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    # Optional but recommended: also remove the limit so it never warns again
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = 1_000_000_000
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description="Count tokens for Qwen3.6 tokenizer")
    parser.add_argument("files", nargs="*", type=Path,
                        help="Text file(s) to process. If omitted, reads from stdin.")
    parser.add_argument("--chat", action="store_true",
                        help="Apply Qwen chat template")
    args = parser.parse_args()

    tokenizer = get_tokenizer()

    def process_text(text: str, source: str):
        if not text.strip():
            return
        if args.chat:
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_count = len(tokenizer.encode(formatted))
        else:
            token_count = count_tokens(text, tokenizer)

        print(f"{token_count}")

    if args.files:
        for file_path in args.files:
            try:
                text = file_path.read_text(encoding="utf-8")
                process_text(text, str(file_path))
            except Exception as e:
                print(f"Error reading {file_path}: {e}", file=sys.stderr)
    else:
        text = sys.stdin.read()
        process_text(text, "stdin")


if __name__ == "__main__":
    main()
