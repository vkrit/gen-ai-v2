import os

# Test pairs: (Thai query, relevant doc, irrelevant doc)
test_cases = [
  {
    "query": "ผลข้างเคียงของยาพาราเซตามอล",
    "relevant": "พาราเซตามอลอาจทำให้เกิดภาวะตับอักเสบหากรับประทานเกินขนาด",
    "irrelevant": "วัคซีนไข้หวัดใหญ่ควรฉีดปีละครั้ง",
  },
]

model_specs = [
    {
        "label": "multilingual-e5-large",
        "model_name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
    },
    {
        "label": "bge-m3",
        "model_name": "BAAI/bge-m3",
        "query_prefix": "",
    },
    {
        "label": "Thai Embedding",
        "model_name": "varin/embeddinggemma-thai-law",
        "query_prefix": "",
    },
]

SentenceTransformer = None

for spec in model_specs:
    if not spec["model_name"]:
        print(f'{spec["label"]:22} | model not available, skipping...', flush=True)
        continue

    if SentenceTransformer is None:
        print("Importing sentence_transformers...", flush=True)
        from sentence_transformers import SentenceTransformer as SentenceTransformerClass

        SentenceTransformer = SentenceTransformerClass

    print(f'Loading model: {spec["label"]}', flush=True)
    model = SentenceTransformer(spec["model_name"])
    for tc in test_cases:
        query = f'{spec["query_prefix"]}{tc["query"]}'
        q   = model.encode([query],            normalize_embeddings=True)
        rel = model.encode([tc["relevant"]],  normalize_embeddings=True)
        irr = model.encode([tc["irrelevant"]], normalize_embeddings=True)
        sim_rel = (q @ rel.T).item()
        sim_irr = (q @ irr.T).item()
        print(f'{spec["label"][:22]:22} | rel={sim_rel:.3f} '
              f"| irr={sim_irr:.3f} | gap={sim_rel-sim_irr:.3f}")
