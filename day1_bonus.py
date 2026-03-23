"""
Day 1 Bonus — Tokenisation Benchmark & Embedding Similarity
============================================================
Measures token count across multiple tokenizers for Thai and English text.
Then computes semantic similarity between Thai sentences using multilingual embeddings.

Run:
    python day1_bonus.py
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Tokenisation benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_tokenisation_benchmark():
    try:
        import tiktoken
    except ImportError:
        print("Install tiktoken: pip install tiktoken")
        return

    thai_texts = [
        "สวัสดีครับ",
        "ผู้ป่วยชายอายุ 58 ปี มาด้วยอาการเจ็บหน้าอกเฉียบพลัน",
        "แพทย์สั่งจ่ายยา Aspirin 300mg และ Clopidogrel 300mg loading dose",
        "วินิจฉัยเบื้องต้น: Acute inferior STEMI ส่งผู้ป่วยทำ Primary PCI",
    ]
    english_texts = [
        "Hello",
        "A 58-year-old male presented with acute chest pain.",
        "Physician prescribed Aspirin 300mg and Clopidogrel 300mg loading dose.",
        "Preliminary diagnosis: Acute inferior STEMI — transfer for Primary PCI.",
    ]

    encodings = {
        "o200k_base (GPT-4o)":  tiktoken.get_encoding("o200k_base"),
        "cl100k_base (GPT-3.5)": tiktoken.get_encoding("cl100k_base"),
    }

    print("=" * 70)
    print("TOKENISATION BENCHMARK")
    print("=" * 70)
    print(f"\n{'Text':<45} {'Encoding':<24} {'Tokens':>7} {'Tok/char':>9}")
    print("-" * 90)

    for th, en in zip(thai_texts, english_texts):
        for enc_name, enc in encodings.items():
            th_n = len(enc.encode(th))
            en_n = len(enc.encode(en))
            print(f"TH: {th[:40]:<44} {enc_name:<24} {th_n:>7,} {th_n/max(len(th),1):>9.2f}")
            print(f"EN: {en[:40]:<44} {enc_name:<24} {en_n:>7,} {en_n/max(len(en),1):>9.2f}")
            ratio = th_n / max(en_n, 1)
            print(f"    Thai uses {ratio:.1f}× more tokens than English equivalent\n")

    # Cost comparison
    full_thai    = " ".join(thai_texts)
    full_english = " ".join(english_texts)
    enc          = tiktoken.get_encoding("o200k_base")
    th_tokens    = len(enc.encode(full_thai))
    en_tokens    = len(enc.encode(full_english))
    price_per_m  = 5.00  # GPT-4o input

    print("\n" + "=" * 50)
    print("COST COMPARISON (GPT-4o, $5/M input tokens)")
    print("=" * 50)
    print(f"Thai text:    {th_tokens:>6,} tokens  →  ${th_tokens/1e6*price_per_m:.6f}")
    print(f"English text: {en_tokens:>6,} tokens  →  ${en_tokens/1e6*price_per_m:.6f}")
    print(f"Thai costs {th_tokens/max(en_tokens,1):.1f}× more for equivalent content")
    print(f"\nAt 1M Thai queries/day (avg 50 tokens each):")
    print(f"  GPT-4o cost:     ${50*1e6/1e6*price_per_m:,.2f}/day")
    print(f"  Typhoon 2 cost:  ${50*1e6/1e6*0.90:,.2f}/day  (65% cheaper)")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Embedding similarity
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding_similarity():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("\nInstall: pip install sentence-transformers numpy")
        return

    print("\n" + "=" * 70)
    print("EMBEDDING SIMILARITY DEMO (multilingual-e5-large)")
    print("=" * 70)
    print("Loading model (first run may download ~1.3GB)...")

    model = SentenceTransformer("intfloat/multilingual-e5-large")

    sentences = [
        ("TH-medical-1",  "แพทย์ตรวจผู้ป่วยที่มีอาการปวดหัว"),
        ("TH-medical-2",  "หมอดูอาการคนไข้ที่ปวดศีรษะ"),
        ("EN-medical",    "The doctor examined the patient with a headache"),
        ("TH-engineering","วิศวกรออกแบบสะพานข้ามแม่น้ำ"),
        ("EN-finance",    "The stock market rose 2% today"),
    ]

    # E5 requires "query: " prefix
    texts    = ["query: " + s for _, s in sentences]
    embeddings = model.encode(texts, normalize_embeddings=True)

    print(f"\n{'Pair':<50} {'Similarity':>10}")
    print("-" * 62)
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            sim  = float(embeddings[i] @ embeddings[j])
            lbl1 = sentences[i][0]
            lbl2 = sentences[j][0]
            bar  = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
            print(f"{lbl1} ↔ {lbl2:<28} {sim:>8.3f}  {bar}")

    print("\nKey observations:")
    print("  • TH-medical-1 ↔ TH-medical-2 should be ~0.90+ (same meaning, different words)")
    print("  • TH-medical-1 ↔ EN-medical   should be ~0.85+ (cross-lingual semantic match)")
    print("  • TH-medical   ↔ TH-engineering should be ~0.50 (different domain)")
    print("  • TH-medical   ↔ EN-finance    should be ~0.30 (very different)")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Decoding experiment (temperature effect)
# ─────────────────────────────────────────────────────────────────────────────

def run_decoding_experiment():
    try:
        import anthropic
    except ImportError:
        print("\nInstall: pip install anthropic")
        return

    from dotenv import load_dotenv
    load_dotenv()

    client = anthropic.Anthropic()
    prompt = "สิ่งที่ AI ทำได้ดีที่สุดในปี 2026 คือ..."  # AI's best capability...

    print("\n" + "=" * 70)
    print("DECODING TEMPERATURE EXPERIMENT")
    print("=" * 70)
    print(f"Prompt: {prompt}\n")

    for temp in [0.0, 0.5, 1.0, 1.5]:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=80,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"temperature={temp}:")
        print(f"  {resp.content[0].text.strip()}")
        print()

    print("Observation: temperature=0.0 always produces the same answer.")
    print("Higher temperature → more creative/varied but less predictable.")


if __name__ == "__main__":
    run_tokenisation_benchmark()
    run_embedding_similarity()
    run_decoding_experiment()
