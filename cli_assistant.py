"""
Day 1 Lab — AI CLI Assistant
==============================
Streams responses from Claude, GPT-4o, or Typhoon 2.
Counts tokens and estimates cost after every response.

Usage:
    python cli_assistant.py --question "What is an LLM?"
    python cli_assistant.py --question "AI คืออะไร?" --lang thai
    python cli_assistant.py --question "Describe this image" --image photo.jpg
    python cli_assistant.py --question "Compare RAG vs fine-tuning" --model gpt-4o
"""

import argparse
import os
import sys
import base64
from pathlib import Path

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Pricing (USD per million tokens, March 2026) ──────────────────────────────
PRICES = {
    "claude-sonnet-4-5":      {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5":       {"in": 0.80,  "out": 4.00},
    "gpt-4o":                 {"in": 5.00,  "out": 15.00},
    "gpt-4o-mini":            {"in": 0.15,  "out": 0.60},
    "typhoon-v2-70b-instruct":{"in": 0.90,  "out": 0.90},
}

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "english": (
        "You are a concise, helpful AI assistant. "
        "Answer clearly and directly. If you are unsure, say so."
    ),
    "thai": (
        "คุณเป็นผู้ช่วย AI ที่มีประโยชน์ ตอบคำถามเป็นภาษาไทยที่สุภาพและกระชับ "
        "ใช้ภาษาเป็นทางการ ลงท้ายด้วย ครับ/ค่ะ ตามความเหมาะสม "
        "ถ้าไม่แน่ใจ ให้บอกตรงๆ"
    ),
}

# ── CostTracker ────────────────────────────────────────────────────────────────
class CostTracker:
    def __init__(self):
        self.total_in  = 0
        self.total_out = 0
        self.total_usd = 0.0

    def record(self, model: str, in_tok: int, out_tok: int) -> float:
        p    = PRICES.get(model, {"in": 0, "out": 0})
        cost = (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000
        self.total_in  += in_tok
        self.total_out += out_tok
        self.total_usd += cost
        return cost

    def print_summary(self, model: str, in_tok: int, out_tok: int, cost: float):
        print(
            f"\n\033[90m[{model}  •  {in_tok:,} in / {out_tok:,} out  •  "
            f"${cost:.6f} this call  •  ${self.total_usd:.6f} total]\033[0m"
        )


tracker = CostTracker()


# ── Image helpers ──────────────────────────────────────────────────────────────
def load_image_base64(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type)."""
    suffix = Path(path).suffix.lower()
    type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".gif": "image/gif",
                ".webp": "image/webp"}
    media_type = type_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


# ── Claude (Anthropic) ─────────────────────────────────────────────────────────
def call_claude(question: str, model: str, system: str,
                image_path: str | None = None) -> tuple[int, int]:
    client = anthropic.Anthropic()

    # Build content
    if image_path:
        img_data, media_type = load_image_base64(image_path)
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": media_type, "data": img_data}},
            {"type": "text", "text": question},
        ]
    else:
        content = question

    print()  # blank line before response
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # newline after streaming
        usage = stream.get_final_message().usage
        return usage.input_tokens, usage.output_tokens


# ── OpenAI / Typhoon 2 ─────────────────────────────────────────────────────────
def call_openai(question: str, model: str, system: str,
                image_path: str | None = None) -> tuple[int, int]:

    # Route to correct base_url
    if "typhoon" in model:
        client = OpenAI(
            base_url="https://api.opentyphoon.ai/v1",
            api_key=os.getenv("TYPHOON_API_KEY", ""),
        )
    else:
        client = OpenAI()

    # Build content
    if image_path:
        img_data, media_type = load_image_base64(image_path)
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
            {"type": "text", "text": question},
        ]
    else:
        content = question

    print()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": content},
        ],
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
    )

    in_tok = out_tok = 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
        if chunk.usage:
            in_tok  = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens
    print()
    return in_tok, out_tok


# ── Router ─────────────────────────────────────────────────────────────────────
CLAUDE_MODELS = {"claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"}

def run(question: str, model: str, lang: str, image_path: str | None):
    system = SYSTEM_PROMPTS[lang]
    print(f"\033[36mModel: {model}  |  Language: {lang}\033[0m")
    if image_path:
        print(f"\033[36mAttachment: {image_path}\033[0m")

    if model in CLAUDE_MODELS:
        in_tok, out_tok = call_claude(question, model, system, image_path)
    else:
        in_tok, out_tok = call_openai(question, model, system, image_path)

    cost = tracker.record(model, in_tok, out_tok)
    tracker.print_summary(model, in_tok, out_tok, cost)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI CLI Assistant — Day 1 Lab")
    parser.add_argument("--question", "-q", required=True,
                        help="Question to ask the model")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-5",
                        choices=list(PRICES.keys()),
                        help="Model to use (default: claude-sonnet-4-5)")
    parser.add_argument("--lang", "-l", default="english",
                        choices=["english", "thai"],
                        help="Response language (default: english)")
    parser.add_argument("--image", "-i", default=None,
                        help="Path to an image file to attach (vision)")
    args = parser.parse_args()

    if args.image and not Path(args.image).exists():
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    run(args.question, args.model, args.lang, args.image)


if __name__ == "__main__":
    main()
