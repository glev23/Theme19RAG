"""
Вспомогательные функции Тема 19: LangChain + BERTA + FAISS + Qwen3 (CPU).
Пути задаются только из CONFIG (передаётся из ноутбука).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BertaEmbeddings(Embeddings):
    """BERTA: префиксы search_document / search_query (карточка модели HF)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device or "cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        emb = self._model.encode(
            texts,
            prompt="search_document: ",
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
        )
        return emb.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self._model.encode(
            text,
            prompt="search_query: ",
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if emb.ndim == 1:
            return emb.tolist()
        return emb[0].tolist()


def load_corpus_documents(corpus_dir: Path) -> list[Document]:
    """Загрузка PDF и текстов из каталога (LangChain loaders)."""
    docs: list[Document] = []
    for ext, loader_cls in (("**/*.pdf", PyPDFLoader), ("**/*.txt", TextLoader), ("**/*.md", TextLoader)):
        for path in corpus_dir.glob(ext):
            if path.name.startswith(".") or path.name.lower() == "readme.md":
                continue
            try:
                if loader_cls is PyPDFLoader:
                    loader = PyPDFLoader(str(path))
                else:
                    loader = TextLoader(str(path), encoding="utf-8")
                part = loader.load()
                for d in part:
                    d.metadata.setdefault("source", str(path.name))
                docs.extend(part)
            except Exception:
                continue
    return docs


def split_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def build_qwen_llm(
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[HuggingFacePipeline, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "clean_up_tokenization_spaces"):
        tokenizer.clean_up_tokenization_spaces = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    # Только max_new_tokens, без max_length — иначе transformers предупреждает при каждом вызове
    model.generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={"max_new_tokens": max_new_tokens},
    )
    return llm, tokenizer


def format_chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_json_block(text: str) -> dict[str, Any]:
    """Пытается вытащить JSON из ответа LLM."""
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSON не найден в ответе модели")
    return json.loads(m.group(0))


def run_llm_json(
    llm: HuggingFacePipeline,
    tokenizer: Any,
    system: str,
    user: str,
) -> tuple[str, dict[str, Any]]:
    prompt = format_chat_prompt(tokenizer, system, user)
    raw = llm.invoke(prompt)
    if isinstance(raw, str):
        raw_text = raw
    else:
        raw_text = str(raw)
    try:
        data = extract_json_block(raw_text)
    except Exception:
        data = {
            "answer": raw_text[:4000],
            "citations": [],
            "confidence": 0.0,
            "notes": "parse_error",
        }
    return raw_text, data


def format_context_with_citations(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] Источник (файл): {src}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def retrieve_docs(
    vectorstore: FAISS,
    question: str,
    k: int,
    fetch_k: int,
    use_mmr: bool,
) -> list[Document]:
    if use_mmr:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return list(retriever.invoke(question))


def ensure_rag_citations(data: dict[str, Any], docs: list[Document]) -> None:
    """Если в JSON пустые/битые citations — добавить цитату из топ-1 retrieval (KPI отчёта, цитаты обязательны)."""
    raw = data.get("citations")
    cits = raw if isinstance(raw, list) else []
    valid: list[dict[str, Any]] = []
    for c in cits:
        if not isinstance(c, dict):
            continue
        fn = str(c.get("file") or c.get("source") or "").strip()
        if fn:
            ex = str(c.get("excerpt", "")).strip()
            valid.append({"file": fn, "excerpt": ex})
    if valid:
        data["citations"] = valid
        return
    if not docs:
        data["citations"] = []
        return
    top = docs[0]
    fn = str(top.metadata.get("source", "unknown"))
    excerpt = (top.page_content or "").strip().replace("\n", " ")[:280]
    prev = (data.get("notes") or "").strip()
    note = "citation_fallback: топ-1 retrieval (в ответе JSON не было валидного списка citations)."
    data["citations"] = [{"file": fn, "excerpt": excerpt}]
    data["notes"] = f"{prev}; {note}" if prev else note


def rag_answer(
    vectorstore: FAISS,
    llm: HuggingFacePipeline,
    tokenizer: Any,
    question: str,
    system_template: str,
    user_suffix: str,
    k: int,
    fetch_k: int,
    use_mmr: bool,
) -> dict[str, Any]:
    docs = retrieve_docs(vectorstore, question, k=k, fetch_k=fetch_k, use_mmr=use_mmr)
    context = format_context_with_citations(docs)
    system = system_template.format(context=context)
    user = f"Вопрос:\n{question}\n\n{user_suffix}"
    raw_text, data = run_llm_json(llm, tokenizer, system, user)
    data["_retrieved_sources"] = [d.metadata.get("source", "") for d in docs]
    data["_raw"] = raw_text
    ensure_rag_citations(data, docs)
    return data


def no_rag_answer(
    llm: HuggingFacePipeline,
    tokenizer: Any,
    question: str,
    system_no_context: str,
    user_suffix: str,
) -> dict[str, Any]:
    system = system_no_context
    user = f"Вопрос:\n{question}\n\n{user_suffix}"
    raw_text, data = run_llm_json(llm, tokenizer, system, user)
    data["_retrieved_sources"] = []
    data["_raw"] = raw_text
    return data


def documents_table_from_chunks(corpus_dir: Path, chunks: list[Document]) -> pd.DataFrame:
    rows = []
    seen = set()
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        if src in seen:
            continue
        seen.add(src)
        name = Path(src).name
        fp = corpus_dir / name
        size = fp.stat().st_size if fp.is_file() else 0
        rows.append(
            {
                "doc_id": Path(src).stem,
                "title": name,
                "source": name,
                "size_bytes": int(size),
            }
        )
    return pd.DataFrame(rows)


# Имя как в ноутбуке / шаблоне build_*
build_documents_table_from_chunks = documents_table_from_chunks


def chunks_table(chunks: list[Document]) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(chunks):
        src = d.metadata.get("source", "unknown")
        doc_id = Path(src).stem
        t = d.page_content
        rows.append(
            {
                "chunk_id": f"c_{i:05d}",
                "doc_id": doc_id,
                "text": t[:2000],
                "length": len(t),
            }
        )
    return pd.DataFrame(rows)


def heuristic_grounded(
    answer: str,
    citations: list,
    retrieved: list[str],
) -> float:
    """Прокси groundedness (0/1): citations согласованы с retrieval или имя файла встречается в ответе."""
    if not answer:
        return 0.0
    ret_set = {str(x).lower() for x in retrieved if x}
    al = answer.lower()
    if ret_set:
        for r in ret_set:
            name = Path(r).name.lower()
            stem = Path(r).stem.lower()
            if name and (name in al or stem in al):
                return 1.0
    if not citations:
        return 0.0
    if not ret_set:
        return 1.0
    for c in citations:
        if isinstance(c, dict):
            fn = str(c.get("file") or c.get("source") or "").lower()
        else:
            fn = str(c).lower()
        if fn and any(fn == r or fn in r or r in fn for r in ret_set):
            return 1.0
    return 0.5
