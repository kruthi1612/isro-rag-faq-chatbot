import argparse
import os
import re
import sys
import time
import shutil
from pathlib import Path
import pymupdf4llm
from dotenv import load_dotenv

# Ensure module import resolution from any working directory
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_function

load_dotenv()

DEFAULT_PDF_ADVT_MAP = {
    "Advertisement_IPRC_Careers_24092026.pdf": "ISRO:IPRC:24092026",
    "Annexure_A_24092026.pdf": "ISRO:IPRC:ANNEXURE_A:24092026",
    "Bilingual_Advertisement_27082026.pdf": "ISRO:ADVT:27082026",
    "FAQs_2026__27082026.pdf": "ISRO:FAQS:27082026",
    "Final_Advt_No_03_2026_19102026.pdf": "ISRO:ICRB:03:2026",
}


def clean_markdown(md_text: str) -> str:
    """Clean and normalize markdown extracted from ISRO PDFs:

    1. Replace <br> and <br/> tags with spaces to prevent word concatenation (e.g. BE001<br>Electronics).
    2. Strip Devanagari script blocks to maximize English token density and embedding quality.
    3. Remove unicode artifacts and replacement characters.
    4. Normalize excess whitespace and newlines.
    """
    # Replace line break tags with spaces so words/tokens remain separated
    md_text = re.sub(r"<br\s*/?>", " ", md_text, flags=re.IGNORECASE)
    # Strip Devanagari script
    md_text = re.sub(r"[\u0900-\u097F]+", "", md_text)
    # Remove unicode replacement and zero-width characters
    md_text = re.sub(r"[\uFFFD\u200B\u200C\u200D]+", " ", md_text)
    # Clean orphaned pipes or spaces
    md_text = re.sub(r"[ \t]+", " ", md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    return md_text.strip()


def parse_and_chunk_pdf(pdf_path: str, advt_no: str):
    """Parse an ISRO notification PDF to Markdown and split into table-aware chunks."""
    raw_md = pymupdf4llm.to_markdown(pdf_path)
    cleaned_md = clean_markdown(raw_md)

    headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    md_splits = MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(cleaned_md)

    # 1500 chunk size with 300 overlap keeps large qualification tables & multi-part FAQs intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " ", ""],
    )
    chunks = splitter.split_documents(md_splits)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{advt_no}_chunk_{idx}"
        chunk.metadata["advt_no"] = advt_no
        chunk.metadata["source"] = os.path.basename(pdf_path)

    return chunks


def ingest_directory(
    pdf_dir: str = "data/raw_pdfs",
    persist_dir: str = "./isro_vectordb",
    provider: str = "huggingface",
    model_name: str = "all-MiniLM-L6-v2",
):
    """Ingest all ISRO recruitment PDFs in a directory into ChromaDB."""
    start_time = time.time()
    all_chunks = []
    print(f"Scanning directory: {pdf_dir}...")
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            advt_no = DEFAULT_PDF_ADVT_MAP.get(filename, os.path.splitext(filename)[0])
            chunks = parse_and_chunk_pdf(pdf_path, advt_no)
            print(f" - Parsed {len(chunks)} chunks from {filename} ({advt_no})")
            all_chunks.extend(chunks)

    if not all_chunks:
        print(f"No PDF files found in {pdf_dir}.")
        return

    print(f"\nTotal parsed chunks: {len(all_chunks)}")
    print(f"Initializing embedding provider: {provider} (model: {model_name})...")
    embeddings = get_embedding_function(provider=provider, model_name=model_name)

    # Clean existing persist_dir to avoid duplicate chunk accumulation
    if os.path.exists(persist_dir):
        print(f"Refreshing existing ChromaDB directory at '{persist_dir}'...")
        shutil.rmtree(persist_dir)

    print(f"Embedding and persisting chunks into ChromaDB at '{persist_dir}'...")
    Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)
    elapsed = time.time() - start_time
    print(f"Done! Successfully indexed {len(all_chunks)} chunks in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ISRO recruitment PDFs into ChromaDB.")
    parser.add_argument("--pdf_dir", default="data/raw_pdfs", help="Directory containing ISRO PDFs")
    parser.add_argument("--pdf", default=None, help="Optional single PDF path")
    parser.add_argument("--advt", default=None, help="Advertisement number for single PDF")
    parser.add_argument("--db", default="./isro_vectordb", help="ChromaDB persistence directory")
    parser.add_argument(
        "--provider",
        default=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        choices=["huggingface", "local", "openai"],
        help="Embedding provider (huggingface: zero tokens/cost, openai: API)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="Embedding model name",
    )
    args = parser.parse_args()

    if args.pdf:
        advt = args.advt or os.path.splitext(os.path.basename(args.pdf))[0]
        chunks = parse_and_chunk_pdf(args.pdf, advt)
        embeddings = get_embedding_function(provider=args.provider, model_name=args.model)
        Chroma.from_documents(chunks, embeddings, persist_directory=args.db)
        print(f"Indexed {len(chunks)} chunks from {args.pdf} into {args.db}.")
    else:
        ingest_directory(args.pdf_dir, args.db, provider=args.provider, model_name=args.model)
