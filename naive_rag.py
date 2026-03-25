from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "claude-sonnet-4-5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
SUPPORTED_SUFFIXES = {".txt", ".pdf"}
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "thai_laws.faiss"
METADATA_PATH = BASE_DIR / "thai_laws_metadata.json"


@dataclass
class ChunkRecord:
    text: str
    source: str


@dataclass
class RetrievalResult:
    text: str
    source: str
    score: float


class ThaiNaiveRAG:
    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        llm_model: str = LLM_MODEL,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._embedder: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._metadata: list[ChunkRecord] = []

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    def chunk_thai(self, text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _read_txt_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _read_pdf_file(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Install dependencies again after updating requirements."
            ) from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if not text:
            raise ValueError(f"Could not extract text from PDF: {path}")
        return text

    def _read_document(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._read_txt_file(path)
        if suffix == ".pdf":
            return self._read_pdf_file(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def ingest(self, docs_path: str) -> int:
        all_chunks: list[ChunkRecord] = []
        doc_paths = sorted(
            path for path in Path(docs_path).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

        for path in doc_paths:
            text = self._read_document(path)
            chunks = self.chunk_thai(text)
            all_chunks.extend(ChunkRecord(text=chunk, source=path.name) for chunk in chunks)
            print(f"Indexed {len(chunks)} chunks from {path.name}")

        if not all_chunks:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(f"No supported files ({supported}) found in {docs_path}")

        vectors = self._encode([chunk.text for chunk in all_chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in all_chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._index = index
        self._metadata = all_chunks
        return len(all_chunks)

    def _load_store(self) -> None:
        if self._index is not None and self._metadata:
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "Index not found. Run ingest first to create thai_laws.faiss and thai_laws_metadata.json."
            )

        self._index = faiss.read_index(str(self.index_path))
        raw_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._metadata = [ChunkRecord(**item) for item in raw_metadata]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._load_store()
        query_vector = self._encode([query])
        scores, indices = self._index.search(query_vector, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx]
            results.append(RetrievalResult(text=chunk.text, source=chunk.source, score=float(score)))
        return results

    def ask(self, question: str, top_k: int = 5) -> str:
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return "ไม่พบข้อมูลในฐานข้อมูลนี้"

        context = "\n\n".join(
            f"[แหล่งที่มา: {chunk.source}]\n{chunk.text}" for chunk in chunks
        )
        system = (
            "คุณเป็นผู้ช่วยด้านกฎหมายไทย ตอบคำถามโดยอ้างอิงจากเอกสารที่ให้มาเท่านั้น "
            'ถ้าไม่พบข้อมูลในเอกสาร ให้บอกว่า "ไม่พบข้อมูลในฐานข้อมูลนี้" '
            "อย่าเดาหรือใช้ความรู้จากแหล่งอื่น"
        )
        user_message = f"เอกสารอ้างอิง:\n{context}\n\nคำถาม: {question}"

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive Thai RAG using FAISS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index .txt and .pdf documents into FAISS")
    ingest_parser.add_argument("docs_path", help="Directory containing .txt and/or .pdf files")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rag = ThaiNaiveRAG()

    if args.command == "ingest":
        total = rag.ingest(args.docs_path)
        print(f"Saved {total} chunks to {rag.index_path.name}")
    elif args.command == "ask":
        print(rag.ask(args.question, top_k=args.top_k))


if __name__ == "__main__":
    main()
