"""
Day 2 Lab B — Thai Legal Assistant: RAG from Naive to Production
================================================================
Builds a complete RAG system on Thai legal documents (PDPA / Thai laws).

Pipeline:
  1. Ingest:   Load → Thai-aware chunk → Embed (bge-m3) → Store (Qdrant)
  2. Retrieve: Hybrid search (dense + BM25) with RRF fusion
  3. Generate: Citation-grounded Thai answer (Claude)
  4. Evaluate: RAGAS faithfulness, context recall, answer relevancy

Install:
    pip install anthropic sentence-transformers qdrant-client rank-bm25
    pip install pythainlp ragas langchain-anthropic datasets python-dotenv
    docker run -p 6333:6333 qdrant/qdrant   ← start Qdrant first
"""

import json
import os
from pathlib import Path
import numpy as np

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"
LLM_MODEL    = "claude-sonnet-4-5"
COLLECTION   = "thai_laws"
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
CHUNK_SIZE   = 400   # characters
CHUNK_OVERLAP = 50
TOP_K        = 5

# ─────────────────────────────────────────────────────────────────────────────
# Sample Thai legal documents (PDPA excerpts for demo)
# Replace with real dataset from:
#   from datasets import load_dataset
#   ds = load_dataset("iapp_wiki_qa_squad", split="train")
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DOCS = [
    {
        "source": "PDPA_มาตรา_37.txt",
        "text": (
            "มาตรา 37 ในกรณีที่มีการละเมิดข้อมูลส่วนบุคคล ให้ผู้ควบคุมข้อมูลส่วนบุคคล "
            "แจ้งเหตุการณ์ละเมิดข้อมูลส่วนบุคคลต่อสำนักงานโดยไม่ชักช้า และเท่าที่จะสามารถทำได้ "
            "ภายในเจ็ดสิบสองชั่วโมงนับแต่ทราบเหตุ เว้นแต่การละเมิดดังกล่าวไม่มีความเสี่ยงที่จะมีผล "
            "กระทบต่อสิทธิและเสรีภาพของบุคคล"
        ),
    },
    {
        "source": "PDPA_มาตรา_77.txt",
        "text": (
            "มาตรา 77 ผู้ใดฝ่าฝืนหรือไม่ปฏิบัติตาม มาตรา 26 มาตรา 27 หรือ มาตรา 28 "
            "ต้องระวางโทษจำคุกไม่เกินหนึ่งปีหรือปรับไม่เกินหนึ่งล้านบาท หรือทั้งจำทั้งปรับ "
            "การกระทำความผิดตามมาตรานี้ ถ้าผู้กระทำมิใช่บุคคลธรรมดา ให้ถือว่ากรรมการ "
            "ผู้จัดการ หรือบุคคลใดซึ่งรับผิดชอบในการดำเนินงานของนิติบุคคลนั้น ต้องรับโทษด้วย"
        ),
    },
    {
        "source": "PDPA_มาตรา_26.txt",
        "text": (
            "มาตรา 26 ห้ามมิให้เก็บรวบรวมข้อมูลส่วนบุคคลที่มีความอ่อนไหว ซึ่งได้แก่ "
            "ข้อมูลเกี่ยวกับเชื้อชาติ เผ่าพันธุ์ ความคิดเห็นทางการเมือง ความเชื่อในลัทธิ "
            "ศาสนาหรือปรัชญา พฤติกรรมทางเพศ ประวัติอาชญากรรม ข้อมูลสุขภาพ ความพิการ "
            "ข้อมูลสหภาพแรงงาน ข้อมูลพันธุกรรม ข้อมูลชีวภาพ เว้นแต่ได้รับความยินยอมโดยชัดแจ้ง"
        ),
    },
    {
        "source": "PDPA_สิทธิผู้เจ้าของ.txt",
        "text": (
            "เจ้าของข้อมูลส่วนบุคคลมีสิทธิดังต่อไปนี้ "
            "หนึ่ง สิทธิได้รับการแจ้งให้ทราบ "
            "สอง สิทธิในการเข้าถึงข้อมูลส่วนบุคคล "
            "สาม สิทธิในการคัดค้านการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคล "
            "สี่ สิทธิในการลบหรือทำลายข้อมูล หรือทำให้ข้อมูลส่วนบุคคลไม่สามารถระบุตัวบุคคล "
            "ห้า สิทธิในการระงับใช้ข้อมูลส่วนบุคคล "
            "หก สิทธิในการให้โอนย้ายข้อมูลส่วนบุคคล"
        ),
    },
    {
        "source": "PDPA_โทษปรับ.txt",
        "text": (
            "โทษปรับทางปกครองตาม PDPA มีดังนี้ "
            "การฝ่าฝืนมาตรา 23 (เก็บข้อมูลเกินความจำเป็น) ปรับสูงสุดไม่เกิน 1 ล้านบาท "
            "การฝ่าฝืนมาตรา 27 (ไม่รักษาความมั่นคงปลอดภัย) ปรับสูงสุดไม่เกิน 3 ล้านบาท "
            "การฝ่าฝืนมาตรา 41 (ไม่แจ้งเหตุการณ์ละเมิด) ปรับสูงสุดไม่เกิน 3 ล้านบาท "
            "นอกจากนี้ยังมีโทษทางอาญาสูงสุดจำคุกไม่เกิน 1 ปีสำหรับการละเมิดข้อมูลอ่อนไหว"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Thai-aware chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_thai(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Character-count chunking with overlap.
    Never splits on bytes — always character boundaries (safe for Thai).
    """
    if len(text) <= size:
        return [text]
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Ingest into Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def ingest(docs: list[dict], embedder: SentenceTransformer, qdrant: QdrantClient):
    print("\n[1] Ingesting documents into Qdrant...")
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    point_id = 0
    for doc in docs:
        chunks = chunk_thai(doc["text"])
        if not chunks:
            continue
        vecs = embedder.encode(chunks, normalize_embeddings=True)
        points = []
        for chunk, vec in zip(chunks, vecs):
            points.append(models.PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={"text": chunk, "source": doc["source"]},
            ))
            point_id += 1
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"  {doc['source']}: {len(chunks)} chunk(s)")
    print(f"  Total: {point_id} chunks indexed")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Hybrid search (dense + BM25)
# ─────────────────────────────────────────────────────────────────────────────

class ThaiHybridSearch:
    def __init__(self, embedder: SentenceTransformer, qdrant: QdrantClient):
        self.embedder = embedder
        self.qdrant   = qdrant
        # Build BM25 index from all stored chunks
        self._build_bm25()

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            try:
                from pythainlp.tokenize import word_tokenize
                self._tokenize = lambda t: word_tokenize(t, engine="newmm")
            except ImportError:
                self._tokenize = lambda t: t.split()

            # Scroll all chunks from Qdrant
            results, _ = self.qdrant.scroll(
                collection_name=COLLECTION, limit=10_000, with_payload=True
            )
            self._chunks   = [r.payload["text"] for r in results]
            self._sources  = [r.payload["source"] for r in results]
            self._ids      = [r.id for r in results]
            tokenized      = [self._tokenize(c) for c in self._chunks]
            self.bm25      = BM25Okapi(tokenized)
            self.bm25_ok   = True
        except ImportError:
            print("  Note: rank_bm25 not installed — using dense-only search")
            self.bm25_ok  = False

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]

        # Dense retrieval
        dense_results = self.qdrant.search(
            collection_name=COLLECTION,
            query_vector=q_vec.tolist(),
            limit=top_k * 2,
        )
        dense_ids    = [r.id for r in dense_results]
        dense_scores = {r.id: r.score for r in dense_results}

        if not self.bm25_ok:
            return [
                {"text": r.payload["text"], "source": r.payload["source"], "score": r.score}
                for r in dense_results[:top_k]
            ]

        # BM25 retrieval
        q_tokens    = self._tokenize(query)
        bm25_raw    = self.bm25.get_scores(q_tokens)
        bm25_ranked = np.argsort(bm25_raw)[::-1][:top_k * 2]
        bm25_ids    = [self._ids[i] for i in bm25_ranked]

        # RRF fusion
        def rrf_score(doc_id, ranked_list, k=60):
            try:
                rank = ranked_list.index(doc_id)
                return 1.0 / (k + rank + 1)
            except ValueError:
                return 0.0

        all_ids = list(set(dense_ids + bm25_ids))
        fused   = {
            did: rrf_score(did, dense_ids) + rrf_score(did, bm25_ids)
            for did in all_ids
        }
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]

        # Look up text from local index
        id_to_idx = {self._ids[i]: i for i in range(len(self._ids))}
        output    = []
        for did in top_ids:
            if did in id_to_idx:
                idx = id_to_idx[did]
                output.append({
                    "text":   self._chunks[idx],
                    "source": self._sources[idx],
                    "score":  fused[did],
                })
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Generate answer with citations
# ─────────────────────────────────────────────────────────────────────────────

def ask(question: str, searcher: ThaiHybridSearch) -> dict:
    chunks = searcher.search(question)

    context = "\n\n".join([
        f"[แหล่งที่มา: {c['source']}]\n{c['text']}"
        for c in chunks
    ])

    system = (
        "คุณเป็นผู้ช่วยด้านกฎหมายไทย เชี่ยวชาญ PDPA\n"
        "ตอบโดยอ้างอิงจากเอกสารที่ให้มาเท่านั้น\n"
        "ถ้าไม่พบข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลในฐานข้อมูลนี้'\n"
        "อ้างอิงแหล่งที่มา (ชื่อไฟล์) เสมอ\n"
        "ตอบเป็นภาษาไทยสุภาพ"
    )

    user_msg = f"เอกสารอ้างอิง:\n{context}\n\nคำถาม: {question}"

    client = anthropic.Anthropic()
    resp   = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {
        "question": question,
        "answer":   resp.content[0].text,
        "contexts": [c["text"] for c in chunks],
        "sources":  [c["source"] for c in chunks],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

EVAL_QA = [
    {"question": "ผู้ควบคุมข้อมูลต้องแจ้งเหตุข้อมูลรั่วภายในกี่ชั่วโมง?",
     "ground_truth": "ต้องแจ้งภายใน 72 ชั่วโมงนับจากทราบเหตุ"},
    {"question": "บทลงโทษสูงสุดสำหรับการละเมิด PDPA ทางอาญาคืออะไร?",
     "ground_truth": "จำคุกไม่เกิน 1 ปี และ/หรือปรับไม่เกิน 1 ล้านบาท"},
    {"question": "ข้อมูลสุขภาพถือเป็นข้อมูลประเภทใดตาม PDPA?",
     "ground_truth": "ข้อมูลส่วนบุคคลที่มีความอ่อนไหว (sensitive personal data)"},
    {"question": "เจ้าของข้อมูลมีสิทธิ์อะไรบ้างภายใต้ PDPA?",
     "ground_truth": "สิทธิ์ได้รับแจ้ง เข้าถึง คัดค้าน ลบ ระงับ และโอนย้ายข้อมูล"},
]


def run_ragas_eval(results: list[dict]):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        eval_dict = {
            "question":    [r["question"]     for r in results],
            "answer":      [r["answer"]       for r in results],
            "contexts":    [r["contexts"]     for r in results],
            "ground_truth":[r["ground_truth"] for r in results],
        }
        ds  = Dataset.from_dict(eval_dict)
        out = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])
        print("\n[RAGAS Results]")
        print(f"  Faithfulness:      {out['faithfulness']:.3f}  (target >0.80)")
        print(f"  Answer Relevancy:  {out['answer_relevancy']:.3f}  (target >0.75)")
        print(f"  Context Recall:    {out['context_recall']:.3f}  (target >0.70)")
    except ImportError:
        print("\nInstall RAGAS: pip install ragas langchain-anthropic datasets")
        print("Skipping automated evaluation.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 2 LAB B — THAI LEGAL ASSISTANT (RAG)")
    print("=" * 60)

    # Load models
    print("\nLoading embedding model (first run downloads ~570MB)...")
    embedder = SentenceTransformer(EMBED_MODEL)

    qdrant = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    # Ingest
    ingest(SAMPLE_DOCS, embedder, qdrant)

    # Build hybrid searcher
    print("\n[2] Building hybrid search index (dense + BM25)...")
    searcher = ThaiHybridSearch(embedder, qdrant)

    # Interactive Q&A demo
    print("\n[3] Running test questions...")
    test_questions = [
        "มาตรา 77 ของ PDPA กำหนดโทษอะไรบ้าง?",
        "ข้อมูลสุขภาพจัดอยู่ในประเภทข้อมูลอ่อนไหวหรือไม่?",
        "เจ้าของข้อมูลมีสิทธิ์อะไรบ้างภายใต้ PDPA?",
        "กรณีข้อมูลผู้ป่วยรั่วไหล โรงพยาบาลต้องทำอย่างไร?",
    ]

    all_results = []
    for q in test_questions:
        print(f"\nQ: {q}")
        result = ask(q, searcher)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Sources: {result['sources']}")
        all_results.append(result)

    # RAGAS evaluation
    print("\n[4] RAGAS Evaluation...")
    eval_results = []
    for qa in EVAL_QA:
        r = ask(qa["question"], searcher)
        r["ground_truth"] = qa["ground_truth"]
        eval_results.append(r)

    run_ragas_eval(eval_results)

    # Save results
    out_path = Path("rag_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == "__main__":
    main()
