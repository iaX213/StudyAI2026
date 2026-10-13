
**Алгоритм HNSW** (Hierarchical Navigable Small World) — основа большинства современных векторных баз. Строит многоуровневый граф: верхние слои — разреженные длинные связи для быстрого перемещения, нижние — плотные короткие для точного поиска. Балансирует точность и скорость через параметры `ef_construction` (качество при индексации) и `ef` (качество при поиске).

#### HNSW: алгоритм под капотом

**HNSW** (Hierarchical Navigable Small World) — многоуровневый граф над векторами:

```
Layer 2 (sparse):  A ──────────── B ──── E   ← длинные "шоссе"
Layer 1:           A ── C ── B ── D ── E
Layer 0 (dense):   A-c1-c2-C-c3-c4-B-c5-D   ← все векторы

ПОИСК: войти в верхний слой → двигаться к цели → спуститься ниже → локализовать
Сложность: O(log N) vs O(N) для полного перебора
```

**Параметры HNSW:**

```
M (максимум рёбер на узел):
  M=16  → production default. Recall 0.95+, разумный размер.
  M=24  → высокий recall, больше памяти, медленнее build.
  M=12  → write-heavy, быстрее вставка, ниже recall.
  ⚠️ Устанавливается ДО вставки данных — после изменить нельзя!

efConstruction (глубина поиска при построении):
  efConstruction=200 → production default. Высокое качество графа.
  efConstruction=400 → максимальное качество, в 2x медленнее build.
  ⚠️ Влияет только на build, не на query.

efSearch (глубина при запросе):
  ef=100 → production default.
  ef=200 → выше recall, выше latency.
  ✅ Можно изменять динамически без перестройки индекса.
```

#### ✏️ Код: создание коллекции с гибридным поиском

```python
from pymilvus import MilvusClient, DataType

DB_PATH    = "/workspace/db/rag_enterprise.db"  # Milvus Lite
COLLECTION = "enterprise_docs"

client = MilvusClient(DB_PATH)

# Пересоздаём при повторном запуске
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# Схема: два векторных поля + метаданные
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True  # произвольные поля без миграции схемы
)
schema.add_field("id",            DataType.INT64,              is_primary=True)
schema.add_field("text",          DataType.VARCHAR,            max_length=8192)
schema.add_field("source",        DataType.VARCHAR,            max_length=512)
schema.add_field("chunk_id",      DataType.INT64)
schema.add_field("dense_vector",  DataType.FLOAT_VECTOR,       dim=dense_dim)  # 1024
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # переменная длина

# Два индекса под два типа векторов
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",
    metric_type="IP",   # Inner Product = Cosine для нормализованных
    params={"M": 16, "efConstruction": 200}
)
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",  # аналог инвертированного индекса
    metric_type="IP"
)

client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    index_params=index_params
)
```

#### ✏️ Код: батчевая загрузка чанков

```python
BATCH_SIZE = 32  # оптимально для 24GB GPU с BGE-M3 fp16

texts   = [c["text"]     for c in all_chunks]
sources = [c["source"]   for c in all_chunks]
ids     = [c["chunk_id"] for c in all_chunks]

total_inserted = 0
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts   = texts[i:i+BATCH_SIZE]
    batch_sources = sources[i:i+BATCH_SIZE]
    batch_ids     = ids[i:i+BATCH_SIZE]

    # ОДИН вызов BGE-M3 → и dense, и sparse одновременно
    embeddings = ef(batch_texts)

    data = [
        {
            "text":          batch_texts[j],
            "source":        batch_sources[j],
            "chunk_id":      batch_ids[j],
            "dense_vector":  embeddings["dense"][j].tolist(),
            "sparse_vector": embeddings["sparse"][j],  # dict {token_id: weight}
        }
        for j in range(len(batch_texts))
    ]

    client.insert(collection_name=COLLECTION, data=data)
    total_inserted += len(data)
    print(f"  Батч {i//BATCH_SIZE + 1}: {total_inserted} чанков проиндексировано")
```

**Почему `BATCH_SIZE=32`:** слишком большой → OOM. Слишком маленький → GPU большую часть времени ждёт transfer данных. 32 — эмпирический оптимум для 24GB GPU с BGE-M3 fp16.

---

### 4.2 Гибридный поиск: сочетание векторного и полнотекстового

### 3.6 Гибридный поиск и алгоритм RRF

Семантический поиск по dense-вектору плохо справляется с точными совпадениями: запрос "astra-freeipa-server" ищет документы с "системой аутентификации" вместо документов с буквальным именем пакета. BM25 справляется с этим, но не понимает синонимы.

**BM25** (Best Match 25) — улучшенная TF-IDF формула, учитывающая насыщение частоты термина и нормализацию по длине документа:

```
Score(q, d) = Σ IDF(qi) × [ f(qi, d) × (k1 + 1) ]
                            / [ f(qi, d) + k1 × (1 - b + b × |d|/avgdl) ]
```

Где `f(qi, d)` — частота термина в документе, `|d|` — длина документа, `k1=1.2` и `b=0.75` — стандартные параметры.

**RRF** (Reciprocal Rank Fusion) — алгоритм слияния результатов нескольких ранжированных списков:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

Где суммирование идёт по всем системам поиска, `rank_i(d)` — позиция документа `d` в i-м списке, `k=60` — сглаживающая константа (стандарт).

Документ, попавший в топ-5 обоих списков, получает суммарный вес ≈ `2/(60+5) ≈ 0.031` — значительно выше, чем документ топ-5 только одного списка. Это и есть "усиление совпавших" без нормализации шкал.

**EnsembleRetriever** в LangChain реализует RRF: принимает список ретриверов и веса, запускает их параллельно, объединяет через RRF.

Индустриальные бенчмарки (например, на датасетах BEIR) показывают, что гибридный поиск (dense + BM25) превосходит чисто векторный в среднем на 10-15% по метрике Recall@10, особенно на корпоративных базах с большим количеством аббревиатур и специфического сленга.

#### BM25: математика точного поиска

```
BM25 SCORE(q, d) = Σ IDF(qi) × [ f(qi,d)×(k1+1) / (f(qi,d) + k1×(1-b+b×|d|/avgdl)) ]

  f(qi, d)  = частота термина qi в документе d
  IDF(qi)   = редкость термина в корпусе (уникальный → высокий IDF)
  k1 = 1.2  = насыщение частоты (diminishing returns)
  b  = 0.75 = нормализация по длине документа
```

BM25 находит то, где semantic-поиск слепой:

```
"ГОСТ Р 59383-2021"     → BM25 находит точно, Dense промахивается
"CVE-2024-12345"        → BM25 по артикулу, Dense по "уязвимость"
"astra-freeipa-server"  → BM25 по имени пакета
"API v3.7.2"            → версия не имеет семантики
```

#### RRF: объединение без нормализации шкал

```
ФОРМУЛА:
  RRF_score(d) = Σ_i  1 / (k + rank_i(d))

  k = 60      — константа сглаживания (производственный стандарт)
  rank_i(d)   — позиция документа d в i-м списке

НАГЛЯДНЫЙ ПРИМЕР (k=60):
  Dense rank:  [DocA=1, DocB=2, DocC=3]
  Sparse rank: [DocB=1, DocC=2, DocA=3]

  DocA: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
  DocB: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252 ← победитель
  DocC: 1/(60+3) + 1/(60+2) = 0.01563 + 0.01613 = 0.03176

  DocB побеждает: он в top-2 обоих поисков
```

**Почему не Weighted Linear Combination:**

```
ПЛОХОЙ ВАРИАНТ:
  combined = α × cosine_score + (1-α) × normalize(bm25_score)
  
  Проблема: BM25 scores неограничены и зависят от корпуса.
  Cosine живёт в [-1, 1]. На разных корпусах нужно перетюнивать α.

ХОРОШИЙ ВАРИАНТ — RRF:
  Работает только с позициями (rank), не со scores.
  k=60 — универсальная константа, не требует тюнинга.
  Работает на любом корпусе из коробки.
```

**Метрики:** бенчмарки BEIR (2025): гибридный поиск превосходит чисто векторный на **10–15% по Recall@10**.

---

#### Hybrid Search с RRF: полный код

```python
from pymilvus import AnnSearchRequest, RRFRanker

def hybrid_search(query: str, top_k: int = 5, rerank_k: int = 20):
    """
    Hybrid search: Dense + Sparse → RRF слияние.
    rerank_k: запрашиваем с запасом для RRF, возвращаем top_k.
    k=60 в RRFRanker — например.
    """
    query_embeddings = ef([query])

    # Запрос к dense (семантический поиск)
    dense_req = AnnSearchRequest(
        data=[query_embeddings["dense"][0].tolist()],
        anns_field="dense_vector",
        param={"metric_type": "IP", "ef": 100},
        limit=rerank_k
    )
    # Запрос к sparse (лексический поиск)
    sparse_req = AnnSearchRequest(
        data=[query_embeddings["sparse"][0]],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=rerank_k
    )
    # RRF слияние: k=60 → сглаживание, не требует нормализации шкал
    results = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["text", "source"]
    )
    return results[0]
```

**Почему `k=60` в RRFRanker:** при k=60 разрыв между позицией 1 и позицией 5 сглаживается — каждый поиск "голосует", а не доминирует. Это позволяет не нормализовать несовместимые шкалы (cosine 0.8 и BM25 24.3 не суммируются напрямую).


---
## 🎯 Модуль 4: Оптимизация поиска

### 4.1 Re-ranking результатов

#### Архитектурная проблема bi-encoder

Bi-encoder (BGE-M3 в Milvus) кодирует запрос и документ **независимо**. Это его главный компромисс:

```
BI-ENCODER (retrieval):

  [Запрос]  ──embed──→ Вектор_Q ──┐
                                   ├── cosine_sim → score
  [Документ]──embed──→ Вектор_D ──┘

  ПЛЮС: O(1) поиск через ANN-индекс, миллионы документов
  МИНУС: нет joint attention — модель не "видит" запрос
         при кодировании документа и наоборот
```

#### Cross-Encoder: совместный анализ

```
CROSS-ENCODER (reranker):

  [Запрос + " [SEP] " + Документ]
              │
              ▼
  [BERT/Transformer: Self-Attention видит ОБА текста]
              │
              ▼
  [Score: 0.0 – 1.0]  ← точная оценка релевантности

  ПЛЮС: joint attention → принципиально более точная оценка
  МИНУС: O(N) на N кандидатов → только на топ-20-100 от retriever
```

**Двухэтапный поиск (production standard):**

```
STAGE 1 — Fast Retrieval (bi-encoder + ANN):
  База: N миллионов чанков
  Алгоритм: HNSW ANN → O(log N)
  Время: 10-50ms
  Результат: top-20 кандидатов

STAGE 2 — Precise Reranking (cross-encoder):
  Вход: 20 кандидатов
  Алгоритм: cross-encoder для каждой пары (query, doc)
  Время: 80-200ms (на GPU)
  Результат: top-3-5 документов

ИТОГО: 100-300ms latency, качество как у cross-encoder
```

### 3.7 Реранкер: второй проход высокой точности

Ретривер работает по принципу "высокий recall, умеренная precision": возвращает top-20 кандидатов с запасом. Реранкер — cross-encoder: он смотрит на пару (запрос, документ) совместно, давая значительно более точную оценку релевантности.

Разница архитектур:

```
Bi-encoder (retriever):  Query → embed_q    ]
                         Doc   → embed_d    ] → cosine_sim(embed_q, embed_d)
                         Независимые вычисления, быстрый поиск

Cross-encoder (reranker): [Query + Doc] → LLM/classifier → relevance_score
                          Совместный анализ, медленно, очень точно
```

Реранкер применяется только к top-K от ретривера (обычно top-20), так как cross-encoder нельзя применить ко всей базе знаний (O(n) вызовов). После реранкинга берут top-3-5 для передачи в генератор.

BGE Reranker v2 (open-source, работает локально) и Cohere Rerank 3.5 — два стандартных выбора в 2026. Для on-premise на том же RunPod поде можно запустить BGE Reranker через `sentence-transformers` без дополнительного GPU (он меньше по размеру).

---

#### BGE-Reranker-v2-m3: характеристики

- **278M параметров** — оптимум quality/latency (v2-large 568M: +2 nDCG@10 но 2x latency)
- **51.8 nDCG@10** на BEIR benchmark (июль 2025)
- MIT-лицензия, BAAI, self-hosted
- GPU latency: **80–200ms** @ top-100 docs @ 512 токенов
- CPU latency: **~130ms** @ batch 16 пар — viable только для offline rerank

По production кейсам: cross-encoder reranker добавляет **10–25% точности** поверх hybrid retrieval — самый высокий ROI среди всех оптимизаций.

#### ✏️ Код: Cross-Encoder Reranker

```python
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

# Загружается один раз, используется в каждом запросе
reranker_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3"  # лучший self-hosted на июль 2026
)
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
```

**Интеграция в retrieval pipeline:**

```python
# ContextualCompressionRetriever = base_retriever + compressor
# Для цепочки выглядит как один retriever, но внутри — двухэтапный pipeline
smart_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=milvus_retriever  # top-20 от Milvus
)
```

**Standalone rerank функция (для CRAG-узлов):**

```python
def rerank_results(query: str, results: list, top_n: int = 5) -> list:
    """Milvus результаты → LangChain Documents → rerank → top_n."""
    docs = [
        Document(
            page_content=r["entity"]["text"],
            metadata={"source": r["entity"]["source"],
                      "milvus_score": r["distance"]}
        )
        for r in results
    ]
    reranked = compressor.compress_documents(docs, query)
    return reranked
```

#### Параметр top_n: как выбрать

```
rerank_k (к reranker):  20-50 кандидатов  ← запрашиваем с запасом
top_n (после rerank):   3-5               ← передаём в LLM

ПОЧЕМУ 3-5:
  "Lost in the middle" — LLM хуже использует факты из середины
  длинного контекста. 5 хорошо отобранных чанков → лучше, чем 20 смешанных.
  Контекст 3000-5000 токенов → нет деградации внимания.

КОГДА top_n = 7-10:
  Аналитические запросы, синтез из разных источников.
  Корпус с низкой плотностью информации.
```

---
