# Step 1: Install
#uv add chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer

# Initialize persistent client (saves data to 'db' directory)
client = chromadb.PersistentClient(path="./db")
model = SentenceTransformer("BAAI/bge-m3")

# Step 2: Create collection with cosine distance
collection = client.get_or_create_collection(
    name="thai_medical_docs",
    metadata={"hnsw:space": "cosine"}
)

# Step 3: Index documents with metadata
docs = [
    "มาตรา 1 พระราชบัญญัตินี้เรียกว่า ...",
    "มาตรา 2 ให้ใช้บังคับตั้งแต่..."
]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    metadatas=[{"department": "pharmacy"}, {"department": "records"}],
    ids=["id1", "id2"]
)

# Step 4: Query with optional metadata filter
query_vec = model.encode(["โทษจำคุกสำหรับการละเมิดข้อมูล"]).tolist()

results = collection.query(
    query_embeddings=query_vec,
    where={"department": "pharmacy"},
    n_results=1
)

print(results['documents'])
