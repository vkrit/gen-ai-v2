# Step 1: Install
#uv add chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer

# Initialize persistent client (saves data to 'db' directory)
client = chromadb.PersistentClient(path="./db")
model = SentenceTransformer("BAAI/bge-m3")

# Step 2: Create collection with cosine distance
collection = client.get_or_create_collection(
    name="thai_pdpa_docs",
    metadata={"hnsw:space": "cosine"}
)

# Step 3: Index documents with metadata
docs = [
    "มาตรา ๑ พระราชบัญญัตินี้เรียกว่า “พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. ๒๕๖๒",
    "มาตรา ๒ พระราชบัญญัตินี้ให้ใช้บังคับตั้งแต่วันถัดจากวันประกาศในราชกิจจานุเบกษา เป็นต้นไป เว้นแต่บทบัญญัติในหมวด ๒ หมวด ๓ หมวด ๕ หมวด ๖ หมวด ๗ และความใน มาตรา ๙๕ และมาตรา ๙๖ ให้ใช้บังคับเมื่อพ้นก าหนดหนึ่งปีนับแต่วันประกาศในราชกิจจานุเบกษาเป็นต้นไป"
]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    metadatas=[{"department": "legal"}, {"department": "records"}],
    ids=["id1", "id2"]
)

# Step 4: Query with optional metadata filter
query_vec = model.encode(["โทษจำคุกสำหรับการละเมิดข้อมูล"]).tolist()

results = collection.query(
    query_embeddings=query_vec,
    where={"department": "legal"},
    n_results=1
)

print(results['documents'])
