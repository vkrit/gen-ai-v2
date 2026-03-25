import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path="./lab_db")
collection = client.get_or_create_collection(
    name="thai_medical_lab",
    metadata={"hnsw:space": "cosine"}
)

thai_docs = [
    {
        "text": "พาราเซตามอลเป็นยาแก้ปวดลดไข้ที่ใช้กันอย่างแพร่หลาย ขนาดยาสำหรับผู้ใหญ่คือ 500-1000 มิลลิกรัม ทุก 4-6 ชั่วโมง ไม่ควรเกิน 4000 มิลลิกรัมต่อวัน การใช้ยาเกินขนาดอาจทำให้ตับวายได้",
        "department": "pharmacy",
        "doc_type": "drug_info",
    },
    {
        "text": "วัคซีนไข้หวัดใหญ่ควรฉีดปีละครั้ง โดยเฉพาะในกลุ่มเสี่ยง ได้แก่ ผู้สูงอายุ เด็กเล็ก และผู้ป่วยโรคเรื้อรัง การฉีดวัคซีนช่วยลดความรุนแรงของโรคและลดการแพร่เชื้อในชุมชน",
        "department": "preventive_medicine",
        "doc_type": "vaccine_guideline",
    },
    {
        "text": "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 กำหนดให้โรงพยาบาลต้องขอความยินยอมจากผู้ป่วยก่อนเปิดเผยข้อมูลสุขภาพ ผู้ฝ่าฝืนมีโทษจำคุกไม่เกินหนึ่งปีหรือปรับไม่เกินหนึ่งล้านบาท",
        "department": "legal",
        "doc_type": "regulation",
    },
]

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
)

all_chunks, all_embeddings, all_metadatas, all_ids = [], [], [], []
for doc_idx, doc in enumerate(thai_docs):
    chunks = splitter.split_text(doc["text"])
    for chunk_idx, chunk in enumerate(chunks):
        chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
        emb = model.encode([chunk], normalize_embeddings=True)[0].tolist()
        all_chunks.append(chunk)
        all_embeddings.append(emb)
        all_metadatas.append({
            "department": doc["department"],
            "doc_type":   doc["doc_type"],
            "chunk_idx":  chunk_idx,
        })
        all_ids.append(chunk_id)

collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids,
)
print(f"Indexed {len(all_chunks)} chunks across {len(thai_docs)} documents")

query = "ขนาดยาที่ปลอดภัยสำหรับผู้ใหญ่"
query_vec = model.encode([query], normalize_embeddings=True).tolist()

print("\n--- Without filter (all departments) ---")
results = collection.query(query_embeddings=query_vec, n_results=3)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['department']}] {doc[:80]}")

print("\n--- With filter: pharmacy only ---")
results_filtered = collection.query(
    query_embeddings=query_vec,
    where={"department": "pharmacy"},
    n_results=2,
)
for doc, meta in zip(results_filtered["documents"][0], results_filtered["metadatas"][0]):
    print(f"[{meta['department']}] {doc[:80]}")
