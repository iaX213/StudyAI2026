
## 🏗️ Итоговая production-архитектура

```
                    ┌───────────────────────────────────────┐
                    │          OFFLINE (INDEXING)           │
                    │                                       │
PDF/DOCX/HTML ──────► Docling Parser                        │
                    │       │                               │
                    │       ▼                               │
                    │  clean_text()                         │
                    │       │                               │
                    │       ▼                               │
                    │  contextual_chunk() + RecursiveSplit  │
                    │ (LLM summary + prefix к каждому чанку)│
                    │       │                               │
                    │       ▼                               │
                    │  BGE-M3: Dense(1024) + Sparse         │
                    │  за один forward pass                 │
                    │       │                               │
                    │       ▼                               │
                    │  Milvus Lite / 2.5                    │
                    │  HNSW(M=16, ef=200) + SparseInverted  │
                    └───────────────────────────────────────┘

                    ┌───────────────────────────────────────┐
                    │           ONLINE (QUERY)              │
                    │                                       │
 USER QUERY ─────────► [Query Router]                       │
                    │  простой / сложный (Multi-Query)      │
                    │       │                               │
                    │       ▼                               │
                    │  Milvus Hybrid Search                 │
                    │  AnnSearchRequest × 2 + RRFRanker(60) │
                    │  top-20 кандидатов                    │
                    │       │                               │
                    │       ▼                               │
                    │  BGE-Reranker-v2-m3                   │
                    │  CrossEncoderReranker: 20 → top-5     │
                    │       │                               │
                    │       ▼                               │
                    │  LangGraph CRAG                       │
                    │  grade → [generate | rewrite→retrieve]│
                    │       │                               │
                    │       ▼                               │
                    │  LLM Generation                       │
                    │  T-lite SFT+DPO @ vLLM :8080          │
                    │  temp=0, streaming output             │
                    │       │                               │
                    │       ▼                               │
                    │  ANSWER + CITATIONS                   │
                    └───────────────────────────────────────┘

                    ┌───────────────────────────────────────┐
                    │         PRODUCTION СЛОИ               │
                    │                                       │
                    │  Langfuse / LangSmith                 │
                    │  → трассировка каждого запроса        │
                    │                                       │
                    │  RAGAS Eval Pipeline                  │
                    │  → Context Recall, Faithfulness       │
                    │  → CI/CD alert on drop                │
                    │                                       │
                    │  Prometheus + Grafana                 │
                    │  → p50/p95 latency per stage          │
                    │  → TTFT, tokens/sec, error rate       │
                    └───────────────────────────────────────┘
```

---

## 📋 Итоговый стек: copy-paste для нового проекта

```bash
# Core RAG stack 2026
pip install \
  docling \
  langchain langchain-community langchain-openai \
  langgraph \
  pymilvus "pymilvus[model]" \
  FlagEmbedding \
  sentence-transformers \
  ragas \
  python-dotenv
```

```
КОМПОНЕНТ      ВЫБОР (2026 default)           АЛЬТЕРНАТИВА (РФ закрытый контур)
───────────    ─────────────────────────────  ────────────────────────────────────
Парсер PDF     Docling (MIT, self-hosted)      Marker-PDF (GPL-3.0, self-hosted)
Чанкинг        RecursiveSplitter 512tok        SemanticChunker (после eval-теста)
Contextual     +LLM summary prefix             кешировать summary на документ
Эмбеддинги     BGE-M3 (MIT, HuggingFace)       multilingual-e5-large (MIT)
Векторная БД   Milvus Lite → 2.5 (Apache 2.0)  Qdrant self-hosted (Apache 2.0)
Hybrid Search  Dense+Sparse → RRFRanker(k=60)  обязательно — no exceptions
Reranker       BGE-Reranker-v2-m3 (MIT)         FlashRank на CPU если нет GPU
LLM            vLLM endpoint (OpenAI API)       любой OpenAI-совместимый API
Граф / агент   LangGraph (CRAG pattern)         LlamaIndex Workflow
Evaluation     RAGAS (open-source)              DeepEval (open-source)
Observability  Langfuse self-hosted (FSL)       MLflow + кастомные метрики
```

---
