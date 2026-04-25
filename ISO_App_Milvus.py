import os
import re
import uuid
import time
import hashlib
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# LangChain & AI Imports
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import transformers
from langdetect import detect as _langdetect, LangDetectException

# Semantic Chunker
from langchain_experimental.text_splitter import SemanticChunker

# Vector DB - Milvus
from langchain_milvus import Milvus

# Graph
from langgraph.graph import StateGraph, END, START

# Image Processing
from PIL import Image
import pytesseract

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MILVUS_DB_DIR = os.path.join(BASE_DIR, "milvus_db")
MILVUS_URI = os.environ.get(
    "MILVUS_URI",
    os.path.join(MILVUS_DB_DIR, "milvus_automotive.db")
)
COLLECTION_NAME = "automotive_requirements"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llamanew29148-req:8b"
DE_EN_TRANSLATION_MODEL_PATH = os.path.join(BASE_DIR, "opus-mt-de-en-local")
EN_DE_TRANSLATION_MODEL_PATH = os.path.join(BASE_DIR, "opus-mt-en-de-local")

# ---------------------------------------------------------------------------
# Context budget & output budget constants
# With the streamlined Modelfile system prompt (~120 tokens instead of ~600),
# we can afford a larger context window AND a much bigger output budget.
# Budget breakdown at MAX_CONTEXT_CHARS=16000 (~4000 tokens):
#   System prompt:    ~120 tokens (streamlined Modelfile)
#   Generation prompt: ~200 tokens
#   RAG context:     ~4000 tokens
#   User query:        ~50 tokens
#   ──────────────────────────────
#   Total prompt:    ~4370 tokens
#   Remaining for output: ~28400 tokens (but num_predict caps at 8192)
# ---------------------------------------------------------------------------
MAX_CONTEXT_CHARS = 16_000   # raised from 10k — more context for grounding
MAX_OUTPUT_TOKENS  = 8192    # raised from 3072 — must match num_predict in Modelfile

# Bypass huggingface network requests to speed up initialization
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---------------------------------------------------------------------------
# Thread-safe module-level singletons
# These are intentionally NOT using @st.cache_resource because LangGraph
# runs nodes in a ThreadPoolExecutor that has no Streamlit session context.
# ---------------------------------------------------------------------------
import threading as _threading

_embedding_lock    = _threading.Lock()
_llm_lock          = _threading.Lock()
_vectorstore_lock  = _threading.Lock()
_embedding_model   = None
_llm               = None
_vectorstore       = None


def _is_remote_milvus_uri(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "tcp://", "grpc://"))


def resolve_milvus_uri() -> str:
    raw = os.path.expandvars(os.path.expanduser((MILVUS_URI or "").strip()))
    if not raw:
        raw = os.path.join(MILVUS_DB_DIR, "milvus_automotive.db")
    if _is_remote_milvus_uri(raw):
        return raw
    local_path = raw if os.path.isabs(raw) else os.path.join(BASE_DIR, raw)
    local_path = os.path.abspath(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                print("🔧 Initializing embedding model...")
                _embedding_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
    return _embedding_model


def get_llm():
    """Primary LLM instance — temperature=0.3 for deterministic structured output."""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = OllamaLLM(
                    model=LLM_MODEL_NAME,
                    temperature=0.3,
                    num_predict=MAX_OUTPUT_TOKENS,
                    num_ctx=32768,
                )
    return _llm


_llm_creative = None


def get_llm_creative():
    """
    Higher-temperature LLM variant (0.45) for sub-requirement generation,
    where diverse decomposition is desired. At 0.3 the model quickly falls
    into repetitive patterns and stops early; 0.45 encourages more varied
    outputs without sacrificing structure.
    """
    global _llm_creative
    if _llm_creative is None:
        with _llm_lock:
            if _llm_creative is None:
                _llm_creative = OllamaLLM(
                    model=LLM_MODEL_NAME,
                    temperature=0.45,
                    num_predict=MAX_OUTPUT_TOKENS,
                    num_ctx=32768,
                )
    return _llm_creative


# ---------------------------------------------------------------------------
# Translation singletons (DE→EN / EN→DE via local MarianMT)
# ---------------------------------------------------------------------------
_translator_lock = _threading.Lock()
_de_en_tokenizer = None
_de_en_model     = None
_en_de_tokenizer = None
_en_de_model     = None

_translation_cache_lock = _threading.Lock()
_translation_cache: Dict[Tuple, str] = {}


def _split_text_for_translation(text: str, max_chunk_chars: int = 400) -> List[str]:
    if not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
    chunks, current, current_len = [], [], 0
    for s in sentences:
        if not s:
            continue
        if current_len + len(s) > max_chunk_chars and current:
            chunks.append(" ".join(current).strip())
            current, current_len = [s], len(s)
        else:
            current.append(s)
            current_len += len(s)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def get_de_en_translator():
    global _de_en_tokenizer, _de_en_model
    if _de_en_model is None:
        with _translator_lock:
            if _de_en_model is None:
                if not os.path.isdir(DE_EN_TRANSLATION_MODEL_PATH):
                    raise FileNotFoundError(
                        f"DE→EN model directory not found: {DE_EN_TRANSLATION_MODEL_PATH}"
                    )
                print("🔧 Loading DE→EN translation model...")
                _de_en_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    DE_EN_TRANSLATION_MODEL_PATH, local_files_only=True
                )
                _de_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                    DE_EN_TRANSLATION_MODEL_PATH, local_files_only=True
                )
    return _de_en_tokenizer, _de_en_model


def get_en_de_translator():
    global _en_de_tokenizer, _en_de_model
    if _en_de_model is None:
        with _translator_lock:
            if _en_de_model is None:
                if not os.path.isdir(EN_DE_TRANSLATION_MODEL_PATH):
                    raise FileNotFoundError(
                        f"EN→DE model directory not found: {EN_DE_TRANSLATION_MODEL_PATH}"
                    )
                print("🔧 Loading EN→DE translation model...")
                _en_de_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    EN_DE_TRANSLATION_MODEL_PATH, local_files_only=True
                )
                _en_de_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                    EN_DE_TRANSLATION_MODEL_PATH, local_files_only=True
                )
    return _en_de_tokenizer, _en_de_model


def _translate_with_local_marian(text: str, direction: str) -> str:
    if not text.strip():
        return text
    if direction == "de_to_en":
        tokenizer, model = get_de_en_translator()
    elif direction == "en_to_de":
        tokenizer, model = get_en_de_translator()
    else:
        raise ValueError(f"Unknown translation direction: {direction}")
    chunks = _split_text_for_translation(text)
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(
            [chunk], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        output_ids = model.generate(**inputs, max_new_tokens=512)
        translated_chunks.append(
            tokenizer.decode(output_ids[0], skip_special_tokens=True)
        )
    return " ".join(translated_chunks).strip()


def detect_language(text: str) -> str:
    try:
        return _langdetect(text)
    except LangDetectException:
        pass
    german_signals = [
        'ä','ö','ü','ß',' die ',' der ',' das ',' und ',
        ' mit ',' für ',' von ',' ist ',' sind ',' werden '
    ]
    hits = sum(1 for sig in german_signals if sig in text.lower())
    return 'de' if hits >= 2 else 'en'


def translate_de_to_en(text: str) -> str:
    try:
        return _translate_with_local_marian(text, "de_to_en")
    except Exception as exc:
        print(f"DE→EN translation error: {exc}")
        return text


def translate_en_to_de(text: str) -> str:
    try:
        return _translate_with_local_marian(text, "en_to_de")
    except Exception as exc:
        print(f"EN→DE translation error: {exc}")
        return text


def translate_entries_to_german(entries: List[Dict]) -> List[Dict]:
    de_entries = []
    for entry in entries:
        e = entry.copy()
        title_src   = entry.get("title", "")
        details_src = entry.get("details", "")
        with _translation_cache_lock:
            title_cached   = _translation_cache.get(("en_to_de", title_src))
            details_cached = _translation_cache.get(("en_to_de", details_src))
        if title_cached is None:
            title_cached = translate_en_to_de(title_src)
            with _translation_cache_lock:
                _translation_cache[("en_to_de", title_src)] = title_cached
        if details_cached is None:
            details_cached = translate_en_to_de(details_src)
            with _translation_cache_lock:
                _translation_cache[("en_to_de", details_src)] = details_cached
        e["title"]   = title_cached
        e["details"] = details_cached
        de_entries.append(e)
    return de_entries


# --- STATE SCHEMA ---
@dataclass
class RequirementsState:
    user_query: str = ""
    transformed_queries: List[str] = field(default_factory=list)
    retrieved_docs: List[Document] = field(default_factory=list)
    reranked_docs: List[Document] = field(default_factory=list)
    is_context_complete: bool = False
    validation_feedback: str = ""
    generated_requirements: str = ""
    retry_count: int = 0
    # BUG FIX #5: max_retries was 1, which means retry_count reached
    # max_retries after just ONE pass, so the retry branch was never taken.
    # Setting to 2 actually allows one real retry before forcing generation.
    max_retries: int = 2
    final_output_path: str = ""


# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------
def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore
        resolved_uri = resolve_milvus_uri()
        try:
            _vectorstore = Milvus(
                embedding_function=get_embedding_model(),
                connection_args={"uri": resolved_uri},
                collection_name=COLLECTION_NAME,
                auto_id=True,
                drop_old=False
            )
            print(f"✅ Milvus connected: {resolved_uri}")
            return _vectorstore
        except Exception as exc:
            if "Open local milvus failed" in str(exc) and not _is_remote_milvus_uri(resolved_uri):
                fallback_uri = os.path.join(
                    MILVUS_DB_DIR,
                    f"milvus_automotive_recovery_{uuid.uuid4().hex[:8]}.db"
                )
                os.makedirs(os.path.dirname(fallback_uri), exist_ok=True)
                print(f"⚠️ Retrying with recovery DB: '{fallback_uri}'.")
                _vectorstore = Milvus(
                    embedding_function=get_embedding_model(),
                    connection_args={"uri": fallback_uri},
                    collection_name=COLLECTION_NAME,
                    auto_id=True,
                    drop_old=False
                )
                return _vectorstore
            raise RuntimeError(
                f"Failed to initialize Milvus. URI='{resolved_uri}', error='{exc}'"
            ) from exc


def ingest_documents(file_path: str):
    """
    Parse, semantically chunk, and ingest a document into Milvus.

    BUG FIX #9 — SemanticChunker percentile tuned from default (95th) to 85th.
    The 95th percentile breakpoint is extremely conservative for technical
    standards documents where semantic drift is gradual. Using 85th produces
    finer-grained, more topically focused chunks, which improves retrieval
    precision (the retriever fetches the right paragraph rather than a 2-page
    block that happens to contain the right paragraph somewhere inside it).
    """
    try:
        text_content = ""
        if file_path.lower().endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            text_content = "\n".join([d.page_content for d in docs])
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            text_content = pytesseract.image_to_string(image)

        if not text_content.strip():
            st.warning("No text could be extracted from the file.")
            return False

        semantic_chunker = SemanticChunker(
            get_embedding_model(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,  # FIX: was default 95th — too coarse
        )
        raw_docs = [Document(page_content=text_content, metadata={"source": file_path})]
        semantic_chunks = semantic_chunker.create_documents(
            [d.page_content for d in raw_docs]
        )
        for chunk in semantic_chunks:
            chunk.metadata["source"] = file_path

        if not semantic_chunks:
            st.warning("Semantic chunker produced 0 chunks.")
            return False

        vector_db = get_vectorstore()
        vector_db.add_documents(semantic_chunks)
        print(f"✅ Ingested {len(semantic_chunks)} semantic chunks into Milvus.")
        return True

    except Exception as e:
        st.error(f"Ingestion Error: {e}")
        return False


# ---------------------------------------------------------------------------
# HELPER: Context budget guard
# ---------------------------------------------------------------------------
def _truncate_context(context_text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    BUG FIX #7: Hard-cap the RAG context string before it is injected into
    the LLM prompt. Without this guard, large semantic chunks can push the
    combined prompt past num_ctx, causing the model to silently drop the
    tail of the context — often the most relevant part — and still only
    generate a handful of tokens because the KV cache is nearly full.
    """
    if len(context_text) <= max_chars:
        return context_text
    truncated = context_text[:max_chars]
    # Don't cut mid-sentence
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        truncated = truncated[:last_period + 1]
    return truncated + "\n[... context truncated for token budget ...]"


def _build_scored_context(docs: List[Document], query: str = "") -> str:
    """
    Build a context string with per-chunk relevance scores.

    RAG Alignment: By tagging each chunk with its cosine similarity score,
    the LLM receives an explicit signal about which chunks are most relevant.
    This compensates for the 8B model's weak implicit attention over long
    unstructured context blocks — it can now prioritise high-scoring chunks.
    """
    if not docs:
        return "No context available."

    import numpy as np

    # If scores are already in metadata (set by reranking_node), use them.
    # Otherwise, compute them on the fly against the provided query.
    need_scoring = any("relevance_score" not in d.metadata for d in docs)
    if need_scoring and query:
        embedder = get_embedding_model()
        query_arr = np.array(embedder.embed_query(query))
        for doc in docs:
            chunk_vec = np.array(embedder.embed_query(doc.page_content))
            doc.metadata["relevance_score"] = round(float(np.dot(query_arr, chunk_vec)), 3)

    # Sort by relevance (highest first) so primacy bias works in our favour
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get("relevance_score", 0.0),
        reverse=True,
    )

    parts = []
    for i, doc in enumerate(sorted_docs, 1):
        score = doc.metadata.get("relevance_score", "n/a")
        source = doc.metadata.get("source", "unknown")
        header = f"[Chunk {i} | Relevance: {score} | Source: {os.path.basename(source)}]"
        parts.append(f"{header}\n{doc.page_content}")

    full_context = "\n\n".join(parts)
    return _truncate_context(full_context)


def _retrieve_fresh_for_parent(
    parent_req: Dict,
    stale_docs: List[Document],
) -> List[Document]:
    """
    Retrieve FRESH context documents specifically relevant to a parent requirement,
    then merge with the original (stale) context docs to maximise coverage.

    This fixes the critical bug where sub-requirement generation reused chunks
    from the original broad query — chunks selected for "powertrain requirements"
    are poor context for drilling into "clutch pack slipping detection".
    """
    import hashlib as _hl

    # Build a search query from the parent requirement's content
    search_query = f"{parent_req.get('title', '')} {parent_req.get('details', '')}".strip()
    if not search_query:
        return stale_docs

    try:
        vector_db = get_vectorstore()
        fresh_results = vector_db.similarity_search(search_query, k=6)
        print(f"   • Fresh retrieval: {len(fresh_results)} docs for '{parent_req.get('req_id', '')}")
    except Exception as exc:
        print(f"   ⚠️ Fresh retrieval failed ({exc}), using stale context")
        return stale_docs

    # Merge: fresh + stale, deduplicated by content hash
    seen = set()
    merged = []
    for doc in fresh_results + (stale_docs or []):
        h = _hl.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    print(f"   • Merged context: {len(merged)} unique docs")
    return merged


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded_file):
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def parse_requirements(text: str) -> List[Dict[str, str]]:
    req_pattern = r"(REQ-[A-Za-z0-9_-]+):\s*(.*?)(?=\nREQ-[A-Za-z0-9_-]+:|\Z)"
    matches = re.findall(req_pattern, text, re.DOTALL)
    if matches:
        items = []
        for req_id, body in matches:
            lines = [l.strip() for l in body.strip().split('\n') if l.strip()]
            title = lines[0] if lines else req_id
            desc_lines = [l for l in lines[1:] if l]
            # Strip leading "Description:" label
            desc_lines = [re.sub(r'^Description:\s*', '', l, flags=re.IGNORECASE)
                          for l in desc_lines]
            desc = " ".join(desc_lines)
            items.append({
                "req_id": req_id,
                "title": title,
                "details": desc,
                "raw": f"{req_id}: {body.strip()}"
            })
        return items

    # Fallback: numbered list
    num_pattern = r"(\d+[\.\)]\s+.+?)(?=\n\d+[\.\)]|\Z)"
    num_matches = re.findall(num_pattern, text, re.DOTALL)
    if num_matches:
        items = []
        for i, block in enumerate(num_matches, 1):
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            title = lines[0]
            desc = " ".join(lines[1:])
            items.append({
                "req_id": f"REQ-{i:03d}",
                "title": title,
                "details": desc,
                "raw": block.strip()
            })
        return items

    return [{"req_id": "REQ-001", "title": "Generated Requirements",
             "details": text.strip(), "raw": text.strip()}]


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

def supervisor_node(state: RequirementsState):
    return state


def query_transformation_node(state: RequirementsState):
    """
    BUG FIX #8 — RAG Alignment via HyDE-inspired query expansion.

    The original code generated 3 generic sub-queries and fed them
    directly to the embedding model. The problem: all-MiniLM-L6-v2
    was trained on short question/answer pairs and produces very
    different embeddings for "what are the safety requirements for an
    inverter?" vs the document text "The inverter shall achieve ≥95%
    efficiency under nominal load…".

    The fix is a two-step expansion:
      1. Generate domain-specific sub-queries (as before).
      2. Generate a short HYPOTHETICAL DOCUMENT EXCERPT (HyDE).
         Embedding a hypothetical answer that looks like the target
         document text produces a query vector much closer to the
         actual chunk vectors in Milvus, dramatically improving recall.
    """
    print("🔄 Node: Query Transformation (with HyDE alignment)")

    # Step 1: sub-queries
    sub_query_prompt = (
        "You are an expert search query generator for automotive engineering documents.\n"
        "Generate 3 specific technical search queries to find relevant specifications for:\n\n"
        f"User Request: {state.user_query}\n\n"
        "Output ONLY the 3 queries, one per line. No numbering, no preamble."
    )
    sub_response = get_llm().invoke(sub_query_prompt)
    sub_queries = [q.strip() for q in sub_response.split('\n') if q.strip()][:3]

    # Step 2: HyDE — generate a short hypothetical document excerpt
    hyde_prompt = (
        "You are an ISO 26262 automotive requirements engineer.\n"
        "Write a single SHORT paragraph (3–5 sentences) that looks like an excerpt from\n"
        "a technical specification document that DIRECTLY answers this request:\n\n"
        f"{state.user_query}\n\n"
        "Output ONLY the paragraph. No headings, no preamble, no requirement IDs."
    )
    hyde_excerpt = get_llm().invoke(hyde_prompt).strip()

    # Combine: original query + sub-queries + HyDE excerpt
    all_queries = [state.user_query] + sub_queries
    if hyde_excerpt:
        all_queries.append(hyde_excerpt)

    print(f"   • Generated {len(all_queries)} queries (including HyDE)")
    return {"transformed_queries": all_queries[:5]}


def retrieval_node(state: RequirementsState):
    """
    BUG FIX #6 — k raised from 3 to 6 per query.
    With only 3 docs per query and 4 queries, deduplication often left
    fewer than 6 unique chunks — not enough context for comprehensive
    requirements generation across multiple subsystems.
    """
    print("🔍 Node: Retrieval")
    try:
        t0 = time.perf_counter()
        vector_db = get_vectorstore()
        all_docs = []
        seen_content = set()

        for q in state.transformed_queries:
            q0 = time.perf_counter()
            results = vector_db.similarity_search(q, k=6)  # FIX: was k=3
            q_ms = (time.perf_counter() - q0) * 1000
            print(f"   • Retrieved {len(results)} docs for query in {q_ms:.1f} ms")

            for doc in results:
                # Use a content hash for dedup (handles whitespace variations)
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        total_ms = (time.perf_counter() - t0) * 1000
        print(f"✅ Retrieval: {len(all_docs)} unique docs in {total_ms:.1f} ms")
        return {"retrieved_docs": all_docs}

    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"retrieved_docs": []}


def reranking_node(state: RequirementsState):
    """
    BUG FIX #4 — Real cosine similarity reranking.

    The original node did `retrieved_docs[:6]` — literally just a list
    slice with no scoring. This is NOT reranking; it's a no-op that
    discards potentially more relevant documents ranked 7–12.

    Fix: embed the original user query once, then score every retrieved
    chunk by cosine similarity, and return the top-N by score.
    This is a lightweight cross-attention-free reranker that runs
    entirely on CPU with the already-loaded embedding model.
    """
    print("⚖️ Node: Reranking (cosine similarity)")
    if not state.retrieved_docs:
        return {"reranked_docs": []}

    import numpy as np

    embedder = get_embedding_model()
    query_vec = embedder.embed_query(state.user_query)
    query_arr = np.array(query_vec)

    scored = []
    for doc in state.retrieved_docs:
        chunk_vec = np.array(embedder.embed_query(doc.page_content))
        # Cosine similarity (vectors are already L2-normalised by encode_kwargs)
        score = float(np.dot(query_arr, chunk_vec))
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = []
    for score, doc in scored[:8]:
        doc.metadata["relevance_score"] = round(score, 3)
        top_docs.append(doc)
    scores_str = [d.metadata["relevance_score"] for d in top_docs]
    print(f"   • Reranked to top {len(top_docs)} docs (scores: {scores_str})")
    return {"reranked_docs": top_docs}


def validation_node(state: RequirementsState):
    """
    BUG FIX #5 — Retry loop was broken.
    Original: retry_count started at 0, incremented to 1 here, and
    max_retries was also 1 → condition `retry_count >= max_retries`
    was True immediately, so "generate" was always chosen on the
    first pass and the retry branch was never taken.

    Fix: max_retries is now 2. The validation logic below is also
    tightened: instead of calling the full LLM (wasting ~500 tokens
    of the budget on a yes/no question), we use a lightweight
    keyword-based heuristic. The LLM call is reserved for ambiguous
    cases only, and the prompt is much shorter.
    """
    print("✔️ Node: Validation")
    context_text = _truncate_context(
        "\n\n".join([d.page_content for d in state.reranked_docs])
    )
    new_retry = state.retry_count + 1

    # Fast heuristic: if we have enough content, skip LLM call
    if len(context_text) > 500 and len(state.reranked_docs) >= 3:
        print("   • Heuristic pass: context is sufficient")
        return {
            "is_context_complete": True,
            "validation_feedback": "",
            "retry_count": new_retry
        }

    # Lightweight LLM validation for genuinely thin context
    prompt = (
        f'Does the following context contain enough technical information to generate\n'
        f'detailed requirements for: "{state.user_query}"?\n\n'
        f'Context (first 800 chars):\n{context_text[:800]}\n\n'
        f'Reply with exactly one word: YES or NO.'
    )
    response = get_llm().invoke(prompt)
    is_complete = "YES" in response.upper()
    feedback = response if not is_complete else ""

    return {
        "is_context_complete": is_complete,
        "validation_feedback": feedback,
        "retry_count": new_retry
    }


def generation_node(state: RequirementsState):
    """
    Generates top-level requirements using relevance-scored RAG context.

    Key improvements:
    - Context chunks are tagged with relevance scores (RAG alignment).
    - Prompt does NOT duplicate the Modelfile system role — it provides only
      task-specific instructions, freeing ~400 tokens for actual output.
    - Requests a MINIMUM of 15 requirements (raised from 10).
    - Explicitly instructs the model NOT to stop early.
    """
    print("📝 Node: Generation")
    context_text = _build_scored_context(state.reranked_docs, state.user_query)

    final_prompt = (
        "CONTEXT DATA (Technical Specifications — ranked by relevance to the query):\n"
        f"{context_text}\n\n"
        "USER REQUEST:\n"
        f"{state.user_query}\n\n"
        "TASK:\n"
        "Generate a MINIMUM of 15 formal requirements based on the context above.\n"
        "Cover ALL relevant subsystem domains found in the context: sensing, actuation,\n"
        "control, communication, power, diagnostics, safety, and HMI where applicable.\n\n"
        "MANDATORY FORMAT for EVERY requirement:\n\n"
        "REQ-XXX: <Concise Title>\n"
        "Description: <Detailed, testable description with specific metrics (ms, V, Nm,\n"
        "degrees-C, Hz) and standard references (ISO 26262 ASIL level, SAE J-numbers)>\n"
        "(Classification tag: Functional | Non-Functional - Safety | Non-Functional - Performance |\n"
        "Non-Functional - Electrical | Non-Functional - Robustness)\n\n"
        "Rules:\n"
        "- Use SHALL or MUST for every binding statement\n"
        "- Each requirement MUST have numeric acceptance criteria\n"
        "- No markdown, bold (**), asterisks, or bullet points\n"
        "- No preamble, apology, or conversational filler\n"
        "- Do NOT stop at 10 — continue until you have exhausted all aspects of the context\n\n"
        "OUTPUT THE REQUIREMENTS NOW:\n"
    )

    response = get_llm().invoke(final_prompt)
    return {"generated_requirements": response}


def finalize_output_node(state: RequirementsState):
    print("💾 Node: Finalize")
    os.makedirs("output", exist_ok=True)
    txt_path = os.path.join("output", "Requirements.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(state.generated_requirements)
    return {"final_output_path": txt_path}


# --- GRAPH EDGES ---
def route_validation(state: RequirementsState):
    if state.is_context_complete or state.retry_count >= state.max_retries:
        return "generate"
    return "transform"


def build_graph():
    workflow = StateGraph(RequirementsState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("transform", query_transformation_node)
    workflow.add_node("retrieve",  retrieval_node)
    workflow.add_node("rerank",    reranking_node)
    workflow.add_node("validate",  validation_node)
    workflow.add_node("generate",  generation_node)
    workflow.add_node("finalize",  finalize_output_node)

    workflow.add_edge(START,       "supervisor")
    workflow.add_edge("supervisor","transform")
    workflow.add_edge("transform", "retrieve")
    workflow.add_edge("retrieve",  "rerank")
    workflow.add_edge("rerank",    "validate")

    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {"generate": "generate", "transform": "transform"}
    )

    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# HIERARCHICAL SUB-REQUIREMENT GENERATION
# ---------------------------------------------------------------------------
def generate_sub_requirements(
    parent_req: Dict,
    additional_instructions: str,
    context_docs: List[Document],
    num_subreqs: int = 8
) -> str:
    """
    Generate sub-requirements for a parent requirement.

    Key improvements:
    - Uses relevance-scored context (RAG alignment).
    - Uses the creative LLM (temperature=0.45) for diverse decomposition.
    - Requests a MINIMUM of 8 sub-requirements (raised from 5).
    - Prompt includes 3 example IDs to prime the model for longer output.
    """
    context_text = _build_scored_context(
        context_docs,
        query=f"{parent_req.get('title', '')} {parent_req.get('details', '')}",
    )

    pid = parent_req['req_id']
    prompt = (
        "Output ONLY plain text — absolutely NO markdown, NO bold (**), NO asterisks.\n\n"
        "PARENT REQUIREMENT:\n"
        f"{pid}: {parent_req['title']}\n"
        f"{parent_req.get('details', '')}\n\n"
        "CONTEXT DATA (ranked by relevance to the parent requirement):\n"
        f"{context_text}\n\n"
        "ADDITIONAL INSTRUCTIONS:\n"
        f"{additional_instructions.strip() if additional_instructions.strip() else 'None provided.'}\n\n"
        f"TASK: Generate a MINIMUM of {num_subreqs} detailed child/sub-requirements for {pid}.\n"
        "Each child requirement MUST:\n"
        f"- Have ID format: {pid}-SUB-N (N = 1, 2, 3, ... up to at least {num_subreqs})\n"
        "- Decompose a specific functional or non-functional aspect of the parent\n"
        "- Include testable numeric acceptance criteria (ms, V, Nm, degrees-C, Hz)\n"
        "- Use SHALL or MUST keywords, never 'should'\n"
        "- Be traceable back to the parent requirement\n\n"
        "Output format — plain text only, one blank line between entries:\n\n"
        f"{pid}-SUB-1: <Title>\n"
        "Description: <Detailed, testable description>\n\n"
        f"{pid}-SUB-2: <Title>\n"
        "Description: <Detailed, testable description>\n\n"
        f"{pid}-SUB-3: <Title>\n"
        "Description: <Detailed, testable description>\n\n"
        "... continue for all sub-requirements ...\n\n"
        f"Generate at least {num_subreqs} sub-requirements. Do NOT stop early.\n\n"
        "OUTPUT THE SUB-REQUIREMENTS NOW:\n"
    )
    return get_llm_creative().invoke(prompt)


def parse_sub_requirements(text: str, parent_id: str) -> List[Dict]:
    clean = re.sub(r'\*+', '', text)
    clean = re.sub(r'_{2}', '', clean)
    clean = clean.strip()

    pid = re.escape(parent_id)
    pattern = rf"({pid}-SUB-\d+):\s*(.*?)(?=\n\s*{pid}-SUB-\d+:|\Z)"
    matches = re.findall(pattern, clean, re.DOTALL)

    if matches:
        items = []
        for req_id, body in matches:
            lines = [l.strip() for l in body.strip().split('\n') if l.strip()]
            title = lines[0].lstrip('*_ ') if lines else req_id
            desc_lines = []
            for l in lines[1:]:
                l = re.sub(r'^Description:\s*', '', l, flags=re.IGNORECASE)
                if l:
                    desc_lines.append(l)
            desc = " ".join(desc_lines)
            items.append({
                "req_id": req_id,
                "title": title,
                "details": desc,
                "raw": f"{req_id}: {body.strip()}",
                "parent_id": parent_id,
                "level": "child",
                "source_query": f"Hierarchical expansion of {parent_id}"
            })
        return items

    # Fallback: numbered list
    num_pattern = r"(\d+[\.\)]\s+.+?)(?=\n\d+[\.\)]|\Z)"
    num_matches = re.findall(num_pattern, clean, re.DOTALL)
    if num_matches:
        items = []
        for i, block in enumerate(num_matches, 1):
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            title = lines[0]
            desc = " ".join(lines[1:])
            items.append({
                "req_id": f"{parent_id}-SUB-{i}",
                "title": title,
                "details": desc,
                "raw": block.strip(),
                "parent_id": parent_id,
                "level": "child",
                "source_query": f"Hierarchical expansion of {parent_id}"
            })
        return items

    return [{
        "req_id": f"{parent_id}-SUB-1",
        "title": "Sub-Requirement",
        "details": clean,
        "raw": clean,
        "parent_id": parent_id,
        "level": "child",
        "source_query": f"Hierarchical expansion of {parent_id}"
    }]


# ---------------------------------------------------------------------------
# COMPILE & EXPORT
# ---------------------------------------------------------------------------
def generate_excel_from_all(all_entries: List[Dict], output_path: str):
    rows = []
    for entry in all_entries:
        rows.append({
            "ID": entry.get("req_id", ""),
            "Title": entry.get("title", ""),
            "Details": entry.get("details", ""),
            "Parent ID": entry.get("parent_id", ""),
            "Level": entry.get("level", "top"),
            "Source Query": entry.get("source_query", "")
        })
    pd.DataFrame(rows).to_excel(output_path, index=False)


def generate_txt_from_all(all_entries: List[Dict], output_path: str):
    lines = ["=" * 70, "COMPILED REQUIREMENTS", "=" * 70, ""]
    for entry in all_entries:
        indent = "    " if entry.get("level") == "child" else ""
        parent_note = (f"  [Sub-requirement of {entry['parent_id']}]"
                       if entry.get("parent_id") else "")
        lines.append(f"{indent}{entry.get('req_id','')}: {entry.get('title','')}{parent_note}")
        if entry.get("details"):
            for line in entry["details"].split('\n'):
                lines.append(f"{indent}  {line}")
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_pdf_from_all(
    all_entries: List[Dict],
    output_path: str,
    doc_title: str = "ISO 29148 Requirements Document",
    doc_subtitle: str = "Auto-generated by ISO Requirements Agent (Milvus + LangGraph)",
    section_heading: str = "Requirements",
):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=1*inch, bottomMargin=0.75*inch
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#1a3c5e"), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#555555"), spaceAfter=20
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading1"],
        fontSize=13, textColor=colors.HexColor("#1a3c5e"),
        spaceBefore=14, spaceAfter=6, borderPad=4
    )
    req_id_style = ParagraphStyle(
        "ReqID", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#0d6efd"),
        spaceBefore=10, spaceAfter=2
    )
    req_title_style = ParagraphStyle(
        "ReqTitle", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=2
    )
    req_detail_style = ParagraphStyle(
        "ReqDetail", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#333333"),
        leftIndent=12, spaceAfter=4
    )
    child_id_style = ParagraphStyle(
        "ChildID", parent=req_id_style,
        fontSize=10, textColor=colors.HexColor("#198754"), leftIndent=20
    )
    child_title_style = ParagraphStyle(
        "ChildTitle", parent=req_title_style, leftIndent=20
    )
    child_detail_style = ParagraphStyle(
        "ChildDetail", parent=req_detail_style, leftIndent=32
    )

    story = []
    story.append(Paragraph(doc_title, title_style))
    story.append(Paragraph(doc_subtitle, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a3c5e"), spaceAfter=16))

    top_count   = sum(1 for e in all_entries if e.get("level") != "child")
    child_count = sum(1 for e in all_entries if e.get("level") == "child")
    summary_data = [
        ["Metric", "Count"],
        ["Top-Level Requirements", str(top_count)],
        ["Sub-Requirements", str(child_count)],
        ["Total", str(len(all_entries))]
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4f8"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    top_entries = [e for e in all_entries if e.get("level") != "child"]
    child_map: Dict[str, List] = {}
    for e in all_entries:
        if e.get("level") == "child" and e.get("parent_id"):
            child_map.setdefault(e["parent_id"], []).append(e)

    story.append(Paragraph(section_heading, section_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#cccccc"), spaceAfter=8))

    for entry in top_entries:
        req_id  = entry.get("req_id", "")
        title   = entry.get("title", "")
        details = entry.get("details", "")

        story.append(Paragraph(req_id, req_id_style))
        story.append(Paragraph(title, req_title_style))
        if details:
            safe = details.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, req_detail_style))

        if req_id in child_map:
            story.append(Spacer(1, 4))
            for child in child_map[req_id]:
                c_id     = child.get("req_id", "")
                c_title  = child.get("title", "")
                c_detail = child.get("details", "")
                story.append(Paragraph(f"↳ {c_id}", child_id_style))
                story.append(Paragraph(c_title, child_title_style))
                if c_detail:
                    safe_c = c_detail.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    story.append(Paragraph(safe_c, child_detail_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"✅ PDF compiled: {output_path}")


def compile_and_export(all_entries, pdf_path, xlsx_path, txt_path):
    generate_pdf_from_all(all_entries, pdf_path)
    generate_excel_from_all(all_entries, xlsx_path)
    generate_txt_from_all(all_entries, txt_path)


def compile_and_export_german(all_entries, pdf_path, xlsx_path, txt_path):
    de_entries = translate_entries_to_german(all_entries)
    generate_pdf_from_all(
        de_entries, pdf_path,
        doc_title="ISO 29148 Anforderungsdokument",
        doc_subtitle="Automatisch generiert vom ISO Anforderungs-Agenten (Milvus + LangGraph)",
        section_heading="Anforderungen",
    )
    generate_excel_from_all(de_entries, xlsx_path)
    generate_txt_from_all(de_entries, txt_path)


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
def run_app():
    st.set_page_config(
        page_title="ISO 29148 Requirements Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    defaults = {
        "all_requirements": [],
        "last_context_docs": [],
        "last_generated_text": "",
        "compiled_files_ready": False,
        "german_files_ready": False,
        "input_language": "en",
        "last_ingested_upload_fingerprint": "",
        "iteration_target_ids": [],
        "user_selected_no_subreqs": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📂 Upload Specification")
        uploaded_file = st.file_uploader(
            "PDF or Image (auto-ingest)", type=["pdf","png","jpg","jpeg"]
        )
        if uploaded_file:
            upload_fingerprint = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            if upload_fingerprint != st.session_state.last_ingested_upload_fingerprint:
                with st.spinner("Running semantic chunking & indexing..."):
                    path    = save_uploaded_file(uploaded_file)
                    success = ingest_documents(path)
                if success:
                    st.session_state.last_ingested_upload_fingerprint = upload_fingerprint
                    st.success("✅ Indexed to Milvus with Semantic Chunker!")
                else:
                    st.error("Ingestion failed. Please re-upload the file to retry.")
            else:
                st.caption(f"Already indexed: {uploaded_file.name}")

        st.markdown("---")
        st.subheader("📋 Session Summary")
        top_count   = sum(1 for e in st.session_state.all_requirements if e.get("level") != "child")
        child_count = len(st.session_state.all_requirements) - top_count
        st.metric("Top-Level Requirements", top_count)
        st.metric("Sub-Requirements", child_count)
        st.metric("Total", len(st.session_state.all_requirements))

        if st.session_state.all_requirements:
            if st.button("🗑️ Clear All Requirements", type="secondary"):
                st.session_state.all_requirements    = []
                st.session_state.last_generated_text = ""
                st.session_state.last_context_docs   = []
                st.session_state.compiled_files_ready = False
                st.rerun()

    # --- MAIN ---
    st.title("🚙 ISO 29148 Requirements Agent")
    st.caption("Milvus · SemanticChunker · LangGraph · HyDE Query Alignment · Cosine Reranking")

    tab_gen, tab_drill, tab_export = st.tabs([
        "1️⃣  Generate Requirements",
        "2️⃣  Drill Down (Hierarchical)",
        "3️⃣  Compile & Export"
    ])

    # ===========================================================
    # TAB 1
    # ===========================================================
    with tab_gen:
        st.subheader("Describe the subsystem or feature")
        user_query = st.text_area(
            "Requirement query",
            placeholder="e.g. Safety requirements for an autonomous braking system",
            height=130,
            label_visibility="collapsed"
        )
        run_btn = st.button("🚀 Generate Requirements", type="primary",
                            use_container_width=True)

        if run_btn:
            if not user_query.strip():
                st.warning("Please enter a description.")
            else:
                detected_lang = detect_language(user_query)
                st.session_state.input_language = detected_lang
                if detected_lang == "de":
                    with st.spinner("🇩🇪 German detected — translating to English..."):
                        english_query = translate_de_to_en(user_query)
                    st.info(f"🇩🇪 Translated query: *{english_query}*")
                else:
                    english_query = user_query

                with st.spinner("Preparing embedding model and vector store..."):
                    try:
                        get_embedding_model()
                        get_vectorstore()
                    except Exception as exc:
                        st.error(f"Milvus preparation failed: {exc}")
                        return

                app       = build_graph()
                thread_id = str(uuid.uuid4())
                config    = {"configurable": {"thread_id": thread_id}}
                initial_state = RequirementsState(user_query=english_query)
                status_box    = st.empty()

                try:
                    final_state = {}
                    with st.spinner("Running RAG Workflow…"):
                        for event in app.stream(initial_state, config):
                            for node, state_update in event.items():
                                status_box.info(f"⚡ Processing Node: **{node.upper()}**")
                                final_state.update(state_update)

                    reqs_text = final_state.get("generated_requirements", "")
                    reranked  = final_state.get("reranked_docs", [])
                    status_box.empty()

                    if reqs_text:
                        st.success("✅ Generation Complete!")
                        parsed = parse_requirements(reqs_text)
                        for p in parsed:
                            if "level" not in p:
                                p["level"] = "top"
                            p["source_query"] = user_query

                        st.session_state.all_requirements.extend(parsed)
                        st.session_state.last_generated_text = reqs_text
                        st.session_state.last_context_docs   = reranked
                        st.session_state.compiled_files_ready = False
                        st.session_state.iteration_target_ids = [p["req_id"] for p in parsed]
                        st.session_state.user_selected_no_subreqs = False

                        st.subheader("Generated Requirements")
                        st.text_area("Output", reqs_text, height=500,
                                     label_visibility="collapsed")
                        st.info(
                            f"✔️ Added **{len(parsed)}** requirement(s) to the session. "
                            "Go to **Tab 2** to drill down, or **Tab 3** to export."
                        )
                    else:
                        st.warning("⚠️ No requirements were generated. Try rephrasing your query.")

                except Exception as exc:
                    import traceback
                    st.error(f"Workflow error: {exc}")
                    st.text(traceback.format_exc())

        if getattr(st.session_state, "iteration_target_ids", []) and not getattr(st.session_state, "user_selected_no_subreqs", False):
            st.markdown("---")
            st.markdown("### 🔄 Do you want to generate more requirements?")
            
            c1, c2, c3 = st.columns([1, 1, 4])
            with c1:
                auto_yes = st.button("✅ Yes", use_container_width=True)
            with c2:
                auto_no = st.button("❌ No", use_container_width=True)
            
            if auto_no:
                st.session_state.user_selected_no_subreqs = True
                st.rerun()
            
            if auto_yes:
                st.session_state.run_auto_subreqs = True
                st.rerun()

        if getattr(st.session_state, "run_auto_subreqs", False):
            st.session_state.run_auto_subreqs = False
            
            target_ids = st.session_state.iteration_target_ids
            all_reqs_dict = {r["req_id"]: r for r in st.session_state.all_requirements}
            
            progress_text = "Iteratively generating sub-requirements..."
            progress_bar = st.progress(0, text=progress_text)
            
            new_subreq_ids = []
            
            for i, parent_id in enumerate(target_ids):
                progress_bar.progress((i) / len(target_ids), 
                                     text=f"Generating sub-requirements for {parent_id} ({i+1}/{len(target_ids)})")
                
                parent_req = all_reqs_dict.get(parent_id)
                if not parent_req:
                    continue
                
                # Fetch context again
                stale_docs = st.session_state.last_context_docs or []
                context_docs = _retrieve_fresh_for_parent(parent_req, stale_docs)
                
                # User's specific prompt
                custom_prompt = (
                    f"Please read the selected \"{parent_req['req_id']}: {parent_req['title']} {parent_req.get('details', '')}\" "
                    f"requirement in-depth technically and critically and generate more in-depth and in-detail functional "
                    f"and non-functional (technical) requirements based on ISO 26262 and automotive SAE standards"
                )
                
                try:
                    raw_output = generate_sub_requirements(parent_req, custom_prompt, context_docs, num_subreqs=10)
                    sub_reqs = parse_sub_requirements(raw_output, parent_id)
                    
                    # Avoid duplicates
                    existing_ids = {e["req_id"] for e in st.session_state.all_requirements}
                    new_subs = [s for s in sub_reqs if s["req_id"] not in existing_ids]
                    
                    st.session_state.all_requirements.extend(new_subs)
                    new_subreq_ids.extend([s["req_id"] for s in new_subs])
                except Exception as exc:
                    st.error(f"Failed to generate sub-requirements for {parent_id}: {exc}")
            
            progress_bar.empty()
            
            if new_subreq_ids:
                st.session_state.iteration_target_ids = new_subreq_ids
                st.success(f"✅ Generated {len(new_subreq_ids)} new sub-requirements! You can continue generating deeper requirements.")
            else:
                st.session_state.iteration_target_ids = []
                st.info("No new sub-requirements were generated.")
            
            st.session_state.compiled_files_ready = False
            st.rerun()

    # ===========================================================
    # TAB 2
    # ===========================================================
    with tab_drill:
        st.subheader("🔍 Drill Down — Generate Sub-Requirements")

        if not st.session_state.all_requirements:
            st.info("Generate top-level requirements in Tab 1 first.")
        else:
            top_reqs = [e for e in st.session_state.all_requirements
                        if e.get("level") != "child"]
            options  = {f"{e['req_id']}: {e['title']}": e for e in top_reqs}
            selected_label = st.selectbox("Select a requirement to expand:", list(options.keys()))
            selected_req   = options[selected_label]

            additional_instructions = st.text_area(
                "Additional instructions (optional):",
                placeholder="e.g. Focus on hardware interfaces only, or add ASIL D constraints",
                height=80
            )

            drill_btn = st.button("🔬 Generate Sub-Requirements", type="primary",
                                  use_container_width=True)

            if drill_btn:
                with st.spinner(f"Generating sub-requirements for {selected_req['req_id']}…"):
                    try:
                        # Fresh retrieval: fetch chunks specifically relevant to
                        # the selected parent requirement, then merge with the
                        # original context docs for maximum coverage.
                        stale_docs = st.session_state.last_context_docs or []
                        context_docs = _retrieve_fresh_for_parent(
                            selected_req, stale_docs
                        )
                        raw_output = generate_sub_requirements(
                            selected_req, additional_instructions, context_docs
                        )
                        sub_reqs = parse_sub_requirements(raw_output, selected_req["req_id"])

                        # Avoid duplicate sub-requirements
                        existing_ids = {e["req_id"] for e in st.session_state.all_requirements}
                        new_subs     = [s for s in sub_reqs if s["req_id"] not in existing_ids]
                        st.session_state.all_requirements.extend(new_subs)
                        st.session_state.compiled_files_ready = False

                        st.success(f"✅ Added {len(new_subs)} sub-requirements!")
                        st.text_area("Raw Output", raw_output, height=350,
                                     label_visibility="visible")
                    except Exception as exc:
                        import traceback
                        st.error(f"Sub-requirement generation failed: {exc}")
                        st.text(traceback.format_exc())

            st.markdown("---")
            children = [e for e in st.session_state.all_requirements
                        if e.get("parent_id") == selected_req["req_id"]]
            if children:
                st.subheader(f"Current Sub-Requirements for `{selected_req['req_id']}`")
                for c in children:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"<span style='color:#198754;font-weight:600'>"
                        f"↳ {c['req_id']}</span>: {c['title']}",
                        unsafe_allow_html=True
                    )

    # ===========================================================
    # TAB 3
    # ===========================================================
    with tab_export:
        st.subheader("📦 Compile All Requirements & Export")

        if not st.session_state.all_requirements:
            st.info("No requirements to export yet.")
        else:
            st.markdown(
                f"Ready to compile **{len(st.session_state.all_requirements)}** "
                "requirement(s) into downloadable files."
            )
            preview_data = [{"ID": e.get("req_id",""), "Title": e.get("title","")[:60],
                             "Level": e.get("level","top"), "Parent": e.get("parent_id","—")}
                            for e in st.session_state.all_requirements]
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

            compile_btn = st.button("⚙️ Compile All Files (PDF · Excel · TXT)",
                                    type="primary", use_container_width=True)

            if compile_btn or st.session_state.compiled_files_ready:
                if compile_btn:
                    os.makedirs("output", exist_ok=True)
                    pdf_path  = os.path.join("output","Requirements_Compiled.pdf")
                    xlsx_path = os.path.join("output","Requirements_Compiled.xlsx")
                    txt_path  = os.path.join("output","Requirements_Compiled.txt")

                    with st.spinner("Compiling output files..."):
                        try:
                            compile_and_export(
                                st.session_state.all_requirements,
                                pdf_path, xlsx_path, txt_path
                            )
                            st.session_state.compiled_pdf   = pdf_path
                            st.session_state.compiled_xlsx  = xlsx_path
                            st.session_state.compiled_txt   = txt_path
                            st.session_state.compiled_files_ready = True
                            st.success("✅ All files compiled successfully!")
                        except Exception as e:
                            import traceback
                            st.error(f"Compilation failed: {e}")
                            st.text(traceback.format_exc())

                if st.session_state.compiled_files_ready:
                    st.markdown("---")
                    st.subheader("⬇️ Download — English")
                    dl1, dl2, dl3 = st.columns(3)

                    for col, path_key, label, fname, mime in [
                        (dl1, "compiled_pdf",  "📄 PDF (EN)",   "Requirements_EN.pdf",
                         "application/pdf"),
                        (dl2, "compiled_xlsx", "📊 Excel (EN)", "Requirements_EN.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                        (dl3, "compiled_txt",  "📝 TXT (EN)",   "Requirements_EN.txt",
                         "text/plain"),
                    ]:
                        p = st.session_state.get(path_key, "")
                        if p and os.path.exists(p):
                            mode = "rb" if path_key != "compiled_txt" else "r"
                            with open(p, mode, **({"encoding":"utf-8"} if mode=="r" else {})) as f:
                                col.download_button(label=label, data=f,
                                                    file_name=fname, mime=mime,
                                                    use_container_width=True)

                    st.markdown("---")
                    st.subheader("🇩🇪 Download — Deutsch (German)")
                    st.caption(
                        "Click below to translate all requirements to German and download. "
                        "Translation is done on-device using the local MarianMT model."
                    )
                    compile_de_btn = st.button("🔄 Translate & Compile German Version",
                                               use_container_width=True)
                    if compile_de_btn:
                        os.makedirs("output", exist_ok=True)
                        pdf_de  = os.path.join("output","Requirements_DE.pdf")
                        xlsx_de = os.path.join("output","Requirements_DE.xlsx")
                        txt_de  = os.path.join("output","Requirements_DE.txt")
                        try:
                            with st.spinner("Translating requirements to German…"):
                                compile_and_export_german(
                                    st.session_state.all_requirements,
                                    pdf_de, xlsx_de, txt_de
                                )
                            st.session_state.compiled_pdf_de  = pdf_de
                            st.session_state.compiled_xlsx_de = xlsx_de
                            st.session_state.compiled_txt_de  = txt_de
                            st.session_state.german_files_ready = True
                            st.success("✅ German version compiled!")
                        except Exception as exc:
                            import traceback
                            st.error(f"German compilation failed: {exc}")
                            st.text(traceback.format_exc())

                    if st.session_state.german_files_ready:
                        dde1, dde2, dde3 = st.columns(3)
                        for col, path_key, label, fname, mime in [
                            (dde1, "compiled_pdf_de",  "📄 PDF (DE)",   "Anforderungen_DE.pdf",
                             "application/pdf"),
                            (dde2, "compiled_xlsx_de", "📊 Excel (DE)", "Anforderungen_DE.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                            (dde3, "compiled_txt_de",  "📝 TXT (DE)",   "Anforderungen_DE.txt",
                             "text/plain"),
                        ]:
                            p = st.session_state.get(path_key, "")
                            if p and os.path.exists(p):
                                mode = "rb" if path_key != "compiled_txt_de" else "r"
                                with open(p, mode, **({"encoding":"utf-8"} if mode=="r" else {})) as f:
                                    col.download_button(label=label, data=f,
                                                        file_name=fname, mime=mime,
                                                        use_container_width=True)


if __name__ == "__main__":
    run_app()