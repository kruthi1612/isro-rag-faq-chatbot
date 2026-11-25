# 🚀 ISRO Recruitment RAG-Based FAQ Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-purple.svg)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Groq](https://img.shields.io/badge/LLM-Groq%20High%20Speed-orange.svg)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/Fallback-Google%20Gemini-blue.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An institutional-grade, production-resilient **Retrieval-Augmented Generation (RAG) FAQ Chatbot** engineered specifically for **Indian Space Research Organisation (ISRO)** and **ISRO Centralised Recruitment Board (ICRB)** recruitment advertisements.

The architecture is built from the ground up to operate reliably on zero-cost tiers: combining **free local CPU embeddings** (`all-MiniLM-L6-v2`), table-preserving layout parsing (`PyMuPDF4LLM`), **Hybrid Ensemble Retrieval** (Regex-tokenized `BM25` + `ChromaDB`), **Multi-Provider LLM Fallback** (**Groq** → **Google Gemini** → **OpenAI**), and an automated **SQLite FAQ response cache**.

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Architecture Overview](#-architecture-overview)
3. [Core Technical Innovations](#-core-technical-innovations)
   - [Zero-Token Local Embeddings](#1-zero-token-local-embeddings-all-minilm-l6-v2)
   - [Multi-Provider LLM Hierarchy & 429 Failover](#2-multi-provider-llm-hierarchy--429-failover)
   - [Persistent SQLite FAQ Cache](#3-persistent-sqlite-faq-cache-6080-token-reduction)
   - [Table-Preserving Markdown Ingestion](#4-table-preserving-markdown-ingestion)
   - [Strict Grounding & Abstention System](#5-strict-grounding--abstention-system)
4. [Official ISRO Knowledge Base](#-official-isro-knowledge-base)
5. [Repository Structure](#-repository-structure)
6. [Setup & Installation](#-setup--installation)
7. [Environment Configuration (.env)](#-environment-configuration-env)
8. [Ingestion & Indexing Pipeline](#-ingestion--indexing-pipeline)
9. [Running the Interactive Web Chatbot](#-running-the-interactive-web-chatbot)
10. [Automated Benchmark Evaluation](#-automated-benchmark-evaluation)
11. [Production Best Practices for Free Tiers](#-production-best-practices-for-free-tiers)
12. [Troubleshooting & FAQ](#-troubleshooting--faq)
13. [Authors & Institutional Attribution](#-authors--institutional-attribution)

---

## 🎯 Executive Summary

Government recruitment notices (such as ISRO/ICRB advertisements) feature intricate qualification matrices, reservation categories, and strict cut-off dates formatted in multi-column tables. Standard RAG architectures frequently fail on these documents due to:
1. **API Rate Limits & Ingestion Quotas**: Heavy reliance on cloud embedding APIs leads to `429 Too Many Requests` errors during ingestion or user spikes.
2. **Table Chunk Fragmentation**: Naive text splitters break rows and columns, losing the link between Post Codes (e.g., `BE001`) and minimum CGPA criteria.
3. **Lexical vs. Semantic Mismatch**: Vector-only search often fails on alphanumeric identifiers (e.g., `BE-003`, `Post Code 17`), while sparse-only search fails on natural language questions (*"What is the upper age limit for reserved categories?"*).

**This system solves all three challenges**:
- **0 external API calls** for embeddings using local HuggingFace models.
- **Hybrid Ensemble Retrieval** (`BM25` lexical search + `ChromaDB` dense vector search).
- **Multi-Provider LLM Failover**: Primary generation on Groq, automatically failing over to Google Gemini upon hitting rate limits.
- **Automated Q&A Caching**: Serves repeated recruitment queries in $<2\text{ ms}$ with **0 LLM tokens**.

---

## 🏛 Architecture Overview

```
                               ┌─────────────────────────┐
                               │  Candidate User Query   │
                               └────────────┬────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │   1. SQLite FAQ Cache Lookup    │
                           │   (Normalized Query Matching)   │
                           └────────┬───────────────┬────────┘
                                    │               │
                 [Cache Hit: 0 Tokens Used]   [Cache Miss]
                                    │               │
                                    ▼               ▼
                      ┌──────────────────┐ ┌───────────────────────────────────┐
                      │  Instant Return  │ │ 2. Hybrid Ensemble Retrieval      │
                      │  (< 2ms Latency) │ │    • Sparse Lexical: BM25         │
                      └──────────────────┘ │      (Regex tokenized for codes)  │
                                           │    • Dense Vector: ChromaDB       │
                                           │    • Embeddings: all-MiniLM-L6-v2 │
                                           │      (Local CPU, 0 tokens)        │
                                           └────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                           ┌───────────────────────────────────┐
                                           │ 3. Multi-Tier LLM Fallback Chain  │
                                           │    • Primary: Groq                │
                                           │      (High speed / 14.4k req/day) │
                                           │    • Automatic 429 Failover:      │
                                           │      Google Gemini (3.5-flash)    │
                                           │    • Secondary Safety Net:        │
                                           │      OpenAI (gpt-4o-mini)         │
                                           └────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                           ┌───────────────────────────────────┐
                                           │ 4. Cache Update & Persistence     │
                                           │    Stores answer + citations in   │
                                           │    isro_faq_cache.sqlite          │
                                           └────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                           ┌───────────────────────────────────┐
                                           │ Strictly Grounded Candidate Resp. │
                                           │ (Advt No. + Section Citations)    │
                                           └───────────────────────────────────┘
```

---

## 💡 Core Technical Innovations

### 1. Zero-Token Local Embeddings (`all-MiniLM-L6-v2`)
- **Zero API Quota Consumption**: Runs `sentence-transformers/all-MiniLM-L6-v2` locally inside the Python backend via `langchain-huggingface`.
- **Ultra-Compact Footprint**: Weighs only **~80 MB** and runs comfortably on standard CPU hosting (Render, Railway, Hugging Face Spaces, or a 1-vCPU VPS).
- **Fast CPU Indexing**: Embeds all **164 notification chunks in ~24 seconds** on standard CPU, completely immune to external API rate limits or billing interruptions.

### 2. Multi-Provider LLM Hierarchy & 429 Failover
The system implements a resilient multi-tier LLM dispatcher (`src/llm_manager.py`):
1. **Primary LLM — Groq**: Routes queries to Groq (`openai/gpt-oss-20b`, `llama-3.1-8b-instant`, or `llama-3.3-70b-versatile`) offering up to 14,400 free requests per day.
2. **Automatic 429 Fallback — Google Gemini (`gemini-3.5-flash-lite`)**: When Groq returns an HTTP 429 Token Rate Limit (`Limit 8000 TPM exceeded`), the engine intercepts the exception, reroutes the prompt with retrieved context, and returns the response from Google Gemini without candidate interruption.
3. **Secondary Safety Net — OpenAI (`gpt-4o-mini`)**: Available as an optional tertiary fallback.

### 3. Persistent SQLite FAQ Cache (60–80% Token Reduction)
Candidate queries for public recruitment notices follow a heavy-tailed power-law distribution. Most queries revolve around the same 20 topics (*"What is the application fee?"*, *"What is the age limit?"*, *"Is B.Tech in AI&ML eligible?"*).
- **Normalization Algorithm**: Converts queries to lowercase, strips punctuation, collapses multi-spaces, and normalizes unicode.
- **SQLite Storage**: Stores normalized queries, original queries, answers, serialized source chunk citations, and access timestamps in `isro_faq_cache.sqlite`.
- **Instant Response**: Repeated inquiries return in **$<2\text{ ms}$** without consuming a single LLM token.
- **Metrics Tracking**: Records total lookups, cache hits, misses, and live hit rate percentage.

### 4. Table-Preserving Markdown Ingestion
- Multi-column tables in ISRO PDFs contain vital post codes and eligibility rules. `PyMuPDF4LLM` extracts raw layout into structured Markdown tables.
- **Tag Normalization**: Unspaced `<br>` tags generated inside table cells (e.g. `|/<br>BE001|`) are converted to clean whitespace, ensuring alphanumeric tokens remain isolated and indexable.
- **Bilingual Cleaning**: Strips Devanagari blocks and replacement characters, maximizing token density for English retrieval.
- **Cohesive Chunking**: Uses `chunk_size=1500` with `chunk_overlap=300` across markdown header boundaries (`#`, `##`, `###`, `|`), keeping complex qualification tables intact.
- **Regex BM25 Tokenizer**: `re.findall(r"[a-zA-Z0-9_\-]+", text.lower())` enables exact lexical retrieval of post codes (e.g., `BE001`, `BE-003`, `Post Code 17`).

### 5. Strict Grounding & Abstention System
- Generation temperature is locked to `0.0`.
- System prompts require explicit citations of the Advertisement Number (`advt_no`) and Section.
- **Hallucination Prevention**: For out-of-scope or unstated facts (such as negative marking in interviews or unlisted exam dates), the bot strictly abstains:
  > *"This information is not specified in the uploaded ISRO recruitment notification."*

---

## 📚 Official ISRO Knowledge Base

The repository indexes 5 official recruitment notifications located in `data/raw_pdfs/`:

| File Name | Assigned Advt ID | Chunks | Key Content Covered |
| :--- | :--- | :---: | :--- |
| `Final_Advt_No_03_2026_19102026.pdf` | `ISRO:ICRB:03:2026` | 79 | Centralized Scientist/Engineer 'SC' & Technical Assistant recruitment notices |
| `Advertisement_IPRC_Careers_24092026.pdf` | `ISRO:IPRC:24092026` | 37 | ISRO Propulsion Complex (IPRC) Mahendragiri technical post notifications |
| `Bilingual_Advertisement_27082026.pdf` | `ISRO:ADVT:27082026` | 26 | Bilingual advertisement: BE001, BE002, BE003 posts, vacancies, qualifications |
| `FAQs_2026__27082026.pdf` | `ISRO:FAQS:27082026` | 17 | Clarification FAQs: degree equivalency (AI&ML), CGPA rules, age criteria |
| `Annexure_A_24092026.pdf` | `ISRO:IPRC:ANNEXURE_A:24092026` | 5 | Physical fitness, endurance standards, and medical criteria |
| **Total Knowledge Base** | **5 Official Documents** | **164** | **100% Ingested & Verified** |

---

## 📁 Repository Structure

```text
isro-rag-faq-chatbot/
├── docs/
│   └── ISRO_RAG_FAQ_Chatbot_Case_Study_Report.docx
├── data/
│   ├── raw_pdfs/
│   │   ├── Advertisement_IPRC_Careers_24092026.pdf
│   │   ├── Annexure_A_24092026.pdf
│   │   ├── Bilingual_Advertisement_27082026.pdf
│   │   ├── FAQs_2026__27082026.pdf
│   │   └── Final_Advt_No_03_2026_19102026.pdf
│   └── eval/
│       ├── isro_faq_eval_dataset.json   # 120 curated benchmark QA items
│       └── isro_faq_eval_dataset.csv
├── src/
│   ├── ingest.py          # PDF-to-Markdown parser, cleaner, chunker, & ChromaDB indexer
│   ├── rag_engine.py      # ISROFAQEngine: Hybrid ensemble retrieval + cache + fallback
│   ├── embeddings.py      # Local HuggingFace / OpenAI embedding factory
│   ├── llm_manager.py     # Multi-provider LLM manager (Groq -> Gemini -> OpenAI)
│   ├── cache.py           # SQLite FAQ cache with query normalization & statistics
│   ├── app.py             # Streamlit web application with live provider & cache indicators
│   └── test_eval.py       # Automated benchmark evaluation test runner
├── isro_vectordb/         # ChromaDB persistence storage (164 chunks)
├── isro_faq_cache.sqlite  # SQLite database for persistent FAQ caching
├── .env.example           # Environment template
├── .gitignore
├── requirements.txt       # Version-locked dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone & Set Up Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/isro-rag-faq-chatbot.git
cd isro-rag-faq-chatbot

# Create virtual environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install version-locked dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration (.env)

Create a `.env` file by copying the template:

```bash
cp .env.example .env
```

Populate `.env` with your free API keys:

```env
# Multi-Provider LLM Keys (Groq -> Gemini -> OpenAI)
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional secondary fallback

# Zero-Token Local Embeddings (Recommended)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Model Selection
GROQ_MODEL=openai/gpt-oss-20b
GEMINI_MODEL=gemini-3.5-flash-lite
OPENAI_MODEL=gpt-4o-mini

# FAQ Caching (1 = enable, 0 = disable)
ENABLE_FAQ_CACHE=1
FAQ_CACHE_DB=./isro_faq_cache.sqlite
```

---

## 🔄 Ingestion & Indexing Pipeline

Run the ingestion script to parse, clean, chunk, and index all 5 PDFs into ChromaDB:

```bash
python src/ingest.py --provider huggingface --model all-MiniLM-L6-v2 --db ./isro_vectordb
```

Execution Log:
```text
Scanning directory: data/raw_pdfs...
 - Parsed 37 chunks from Advertisement_IPRC_Careers_24092026.pdf (ISRO:IPRC:24092026)
 - Parsed 5 chunks from Annexure_A_24092026.pdf (ISRO:IPRC:ANNEXURE_A:24092026)
 - Parsed 26 chunks from Bilingual_Advertisement_27082026.pdf (ISRO:ADVT:27082026)
 - Parsed 17 chunks from FAQs_2026__27082026.pdf (ISRO:FAQS:27082026)
 - Parsed 79 chunks from Final_Advt_No_03_2026_19102026.pdf (ISRO:ICRB:03:2026)

Total parsed chunks: 164
Initializing embedding provider: huggingface (model: all-MiniLM-L6-v2)...
Refreshing existing ChromaDB directory at './isro_vectordb'...
Embedding and persisting chunks into ChromaDB at './isro_vectordb'...
Done! Successfully indexed 164 chunks in 24.54 seconds.
```

---

## 💻 Running the Interactive Web Chatbot

Launch the Streamlit web application:

```bash
streamlit run src/app.py
```

The application will launch at `http://localhost:8501`.

### Key Interface Features:
- **Response Attribution**: Every assistant response is tagged with its origin — `⚡ Served from cache · 0 tokens used`, `🤖 Groq (Primary)`, or `🤖 Google Gemini (Fallback)`.
- **Source Passages Expander**: Inspect the retrieved notification passages, advertisement identifier, and section heading for every answer.
- **Quick FAQ Chips**: One-click query buttons for the most frequent candidate inquiries shown on first load.
- **Clean, focused UI**: No sidebar or configuration panel — all production defaults are hardcoded for a consistent, professional experience.

---

## 📊 Automated Benchmark Evaluation

The repository includes a dedicated benchmark evaluation script (`src/test_eval.py`) that executes live queries across all 5 categories from the 120-item dataset:

```bash
python src/test_eval.py
```

### Empirical Test Matrix:

| Query ID | Category | Candidate Question | Ground Truth / Answer | Engine Used | Grounding Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **FAQ_001** | Eligibility & Qualification | *What is the minimum percentage or CGPA required for Post Code BE001?* | Minimum 65% marks or CGPA 6.84/10 in qualifying degree. | Groq (Primary) | ✅ Grounded (`ISRO:ADVT:27082026`) |
| **FAQ_003** | Eligibility & Qualification | *Are candidates with a BE/B.Tech in AI&ML eligible for Post Code BE003?* | Yes, AI&ML is explicitly listed as eligible under Computer Science. | Groq (Primary) | ✅ Grounded (`ISRO:FAQS:27082026`) |
| **FAQ_021** | Age Limit & Relaxation | *Which ITI trades are accepted for Technician 'B' Chemical (Post Code 17)?* | Chemical, Attendant Operator, Instrument Mechanic, Maintenance Mechanic, Lab Assistant. | Groq (Primary) | ✅ Grounded (`ISRO:ICRB:03:2026`) |
| **FAQ_041** | Fees, Dates & Selection | *What happens if a suitable PwBD candidate is not found for a backlog vacancy?* | Post may be interchanged among 5 PwBD categories or filled by non-PwBD candidate. | Google Gemini (Fallback on 429) | ✅ Grounded (`ISRO:ADVT:27082026`) |
| **FAQ_071** | Vacancies & Reservation | *What is the age limit for Fireman 'A' (Post Code 71) at IPRC Mahendragiri?* | 18–25 years as of 05.10.2026 (relaxations apply if reserved). | Groq (Primary) | ✅ Grounded (`ISRO:IPRC:24092026`) |
| **FAQ_101** | Selection & Testing | *Is Travelling Allowance paid for attending the Written Test at SDSC SHAR?* | No Travelling Allowance will be paid for attending the Written Test. | Google Gemini (Fallback) | ✅ Grounded (`ISRO:ICRB:03:2026`) |
| **FAQ_103** | Out-of-Scope / Abstention | *What is the minimum GATE score required to apply for Scientist/Engineer 'SC'?* | *"This information is not specified in the uploaded ISRO recruitment notification."* | Groq (Primary) | ✅ Correct Abstention |

---

## 🛡️ Production Best Practices for Free Tiers

1. **Leverage FAQ Caching**:
   Student and candidate traffic follows a power-law distribution. Query normalization and persistent caching reduces monthly LLM API tokens by **60–80%** while reducing response time to $<2\text{ ms}$.
2. **Public Data Protection**:
   Free-tier providers (Google AI Studio, Groq) may log prompts for abuse prevention. This is acceptable for public ISRO recruitment notifications, but the application avoids logging candidate personally identifiable information (PII).
3. **Respect Project-Level Quotas**:
   Do not generate multiple API keys under a single account to bypass limits (rate limits are enforced at the project level). The built-in multi-provider failover mechanism (**Groq** → **Gemini** → **OpenAI**) provides resilient uptime.

---

## 🔧 Troubleshooting & FAQ

### Q1: What happens if Groq returns a 429 rate limit error?
The application's `MultiProviderLLMManager` automatically catches HTTP 429 exceptions and transparently retries the prompt with the retrieved context using the Google Gemini free tier (`gemini-3.5-flash-lite`).

### Q2: Why are local HuggingFace embeddings preferred over OpenAI embeddings?
The ISRO recruitment dataset consists of 164 chunks. Local `all-MiniLM-L6-v2` embeddings run in-memory on CPU, take only ~80 MB of RAM, execute in $<25\text{ ms}$, cost \$0, and never hit an API rate limit during ingestion or user searches.

### Q3: How do I clear the FAQ cache?
Delete the SQLite cache file and restart the application:
```bash
rm isro_faq_cache.sqlite
```

---
