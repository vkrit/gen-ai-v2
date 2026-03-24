from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-large")

sentences = [
 "แพทย์ตรวจผู้ป่วย", # Doctor examines patient
 "หมอดูแลคนไข้", # Physician cares for patient
 "วิศวกรออกแบบสะพาน", # Engineer designs bridge
 "Doctor examines patient",
]

# E5 models require this prefix
prefixed = ["query: " + s for s in sentences]
embeddings = model.encode(prefixed, normalize_embeddings=True)

# Cosine similarity = dot product (vectors are normalized)
sim = embeddings @ embeddings.T
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            print(f"{s1[:20]:22} | {s2[:20]:22} | sim={sim[i,j]:.3f}")
