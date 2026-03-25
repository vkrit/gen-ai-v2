from sentence_transformers import SentenceTransformer
import numpy as np

test_cases = [
    {
        "id": "drug_se",
        "query":     "ผลข้างเคียงของยาพาราเซตามอล",
        "relevant":  "พาราเซตามอลอาจทำให้เกิดภาวะตับอักเสบหากรับประทานเกินขนาด",
        "irrelevant":"วัคซีนไข้หวัดใหญ่ควรฉีดปีละครั้ง",
    },
    {
        "id": "negation_trap",
        "query":     "ขนาดยาพาราเซตามอลสำหรับผู้ใหญ่",
        "relevant":  "ผู้ใหญ่รับประทานพาราเซตามอลได้ 500-1000 มก. ทุก 4-6 ชั่วโมง",
        "irrelevant":"ห้ามรับประทานพาราเซตามอลเกิน 4000 มก. ต่อวัน",
    },
    {
        "id": "pdpa",
        "query":     "โทษของการละเมิดข้อมูลส่วนบุคคล",
        "relevant":  "ผู้ละเมิดข้อมูลส่วนบุคคลมีโทษจำคุกไม่เกินหนึ่งปี",
        "irrelevant":"โรงพยาบาลต้องจัดทำนโยบายความเป็นส่วนตัว",
    },
]

models_to_test = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

for model_name in models_to_test:
    model = SentenceTransformer(model_name)
    short = model_name.split("/")[-1]
    for tc in test_cases:
        q   = model.encode([tc["query"]],      normalize_embeddings=True)
        rel = model.encode([tc["relevant"]],   normalize_embeddings=True)
        irr = model.encode([tc["irrelevant"]], normalize_embeddings=True)
        sim_rel = (q @ rel.T).item()
        sim_irr = (q @ irr.T).item()
        gap = sim_rel - sim_irr
        flag = "PASS" if gap >= 0.30 else ("WARN" if gap >= 0.15 else "FAIL")
        print(f"{short} | {tc['id']} | rel={sim_rel:.3f} irr={sim_irr:.3f} gap={gap:.3f} [{flag}]")

model_safe = SentenceTransformer("BAAI/bge-m3")
q   = model_safe.encode(["ยาพาราเซตามอลปลอดภัยสำหรับผู้ป่วยโรคตับ"],  normalize_embeddings=True)
pos = model_safe.encode(["พาราเซตามอลปลอดภัยในขนาดปกติสำหรับผู้ป่วยทั่วไป"], normalize_embeddings=True)
neg = model_safe.encode(["ห้ามใช้พาราเซตามอลในผู้ป่วยโรคตับ"],           normalize_embeddings=True)
print(f"Safe doc similarity:   {(q @ pos.T).item():.3f}")
print(f"Danger doc similarity: {(q @ neg.T).item():.3f}")
print("If scores are close, embedding cannot distinguish safe vs dangerous!")
