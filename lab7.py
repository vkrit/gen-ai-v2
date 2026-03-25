from pythainlp.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter

def thai_sentence_splitter(text: str) -> list[str]:
    """Use PyThaiNLP CRFCut for Thai sentence tokenization."""
    return sent_tokenize(text, engine="crfcut")

# Custom separators respecting Thai document structure
thai_splitter = RecursiveCharacterTextSplitter(
    separators=[
    "\n\n", # paragraph break (highest priority)
    "\n", # line break
    "\u0e01", # Thai character boundary (fallback)
    " ", # space (for mixed Thai-English content)
    ],
    chunk_size=500, # characters, NOT bytes — critical for Thai
    chunk_overlap=50, # overlap preserves cross-boundary context
    length_function=len, # character count (not byte count)
)

# Production-preferred: token-based splitting aligned to your model
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

token_splitter = SentenceTransformersTokenTextSplitter(
    model_name="BAAI/bge-m3",
    chunk_size=256, # tokens within model's 8192-token window
    chunk_overlap=32 # ~12% overlap for context continuity
)

thai_medical_text = "ผู้ป่วยมีไข้ ไอ และเจ็บคอมา 3 วัน แพทย์แนะนำให้พักผ่อน ดื่มน้ำมากๆ และรับประทานยาตามอาการ"

# Usage
chunks = token_splitter.create_documents([thai_medical_text])
print(f"Created {len(chunks)} chunks")
