import os
import json
import faiss
import httpx
import asyncio
import time  # For timing tokens/sec
import numpy as np
import chainlit as cl
from pathlib import Path
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from chainlit.input_widget import Select, Slider
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

load_dotenv()

# --- Config & RAG Class (Same as before) ---
EMBED_MODEL = "intfloat/multilingual-e5-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_SUFFIXES = {".txt", ".pdf"}
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "thai_laws.faiss"
METADATA_PATH = BASE_DIR / "thai_laws_metadata.json"

@dataclass
class ChunkRecord:
    text: str
    source: str
    page: str = "N/A"

@dataclass
class RetrievalResult:
    text: str
    source: str
    page: str
    score: float

class ThaiNaiveRAG:
    def __init__(self):
        self._embedder = None
        self._index = None
        self._metadata = []
        
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def retrieve(self, query: str, top_k: int = 5):
        if self._index is None:
            if INDEX_PATH.exists():
                self._index = faiss.read_index(str(INDEX_PATH))
                raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                self._metadata = [ChunkRecord(**item) for item in raw]
            else: return []
        
        top_k_int = int(top_k)
        query_v = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.asarray(query_v, dtype="float32"), top_k_int)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                c = self._metadata[idx]
                results.append(RetrievalResult(text=c.text, source=c.source, page=c.page, score=float(score)))
        return results

# --- UI Logic ---

@cl.on_chat_start
async def start():
    cl.user_session.set("rag", ThaiNaiveRAG())
    settings = await cl.ChatSettings([
        Select(id="provider", label="LLM Provider", values=["Ollama", "OpenAI", "Anthropic"], initial_index=0),
        Slider(id="top_k", label="Retrieval Count", initial=4, min=1, max=10)
    ]).send()
    cl.user_session.set("settings", settings)
    await cl.Message(content="👋 สวัสดีครับ! อัปโหลดไฟล์เพื่อเริ่ม RAG").send()

@cl.on_settings_update
async def update(settings):
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag")
    settings = cl.user_session.get("settings")

    # (Ingestion logic remains the same as previous response...)
    if message.elements:
        # Filter for supported files and handle ingestion (skipped here for brevity)
        pass 

    # --- QUERY & REFERENCES ---
    results = rag.retrieve(message.content, top_k=settings["top_k"])
    if not results:
        await cl.Message(content="ไม่พบข้อมูลในระบบ").send()
        return

    source_elements = [cl.Text(name=f"{r.source} (หน้า {r.page})", content=r.text, display="side") for r in results]
    context_text = "\n\n".join([f"--- [ที่มา: {res.source} หน้า {res.page}] ---\n{res.text}" for res in results])
    
    system_prompt = "คุณเป็นผู้ช่วย ตอบคำถามจากเอกสารที่ให้มาเท่านั้น อ้างอิงชื่อไฟล์และหน้าเสมอ เช่น [Filename.pdf หน้า 1]"
    user_prompt = f"เอกสารอ้างอิง:\n{context_text}\n\nคำถาม: {message.content}"
    
    resp_msg = cl.Message(content="")
    provider = settings["provider"]
    
    # Stats trackers
    model_name = ""
    token_count = 0
    start_time = time.perf_counter()

    try:
        if provider == "OpenAI":
            model_name = "gpt-4o"
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            stream = await client.chat.completions.create(
                model=model_name,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                stream=True,
                stream_options={"include_usage": True} # Critical for token counts
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    await resp_msg.stream_token(chunk.choices[0].delta.content)
                if chunk.usage:
                    token_count = chunk.usage.total_tokens

        elif provider == "Anthropic":
            model_name = "claude-3-5-sonnet-20240620"
            client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            async with client.messages.stream(
                model=model_name, max_tokens=1024, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    await resp_msg.stream_token(text)
                final = await stream.get_final_message()
                token_count = final.usage.input_tokens + final.usage.output_tokens

        else: # Ollama
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip("/")
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{base_url}/api/chat", json={
                    "model": model_name,
                    "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
                }, timeout=120.0) as r:
                    async for line in r.aiter_lines():
                        if line:
                            body = json.loads(line)
                            if t := body.get("message", {}).get("content"):
                                await resp_msg.stream_token(t)
                            if body.get("done"):
                                # Ollama provides tokens directly
                                token_count = body.get("prompt_eval_count", 0) + body.get("eval_count", 0)

        # --- CALCULATE METRICS ---
        end_time = time.perf_counter()
        duration = end_time - start_time
        tps = token_count / duration if duration > 0 else 0
        
        # Format the stats footer
        stats_footer = (
            f"\n\n---\n"
            f"**Model:** `{model_name}` | "
            f"**Tokens:** `{token_count}` | "
            f"**Speed:** `{tps:.2f} t/s`"
        )
        
        # Append stats to message content
        resp_msg.content += stats_footer

    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
    
    resp_msg.elements = source_elements
    await resp_msg.send()