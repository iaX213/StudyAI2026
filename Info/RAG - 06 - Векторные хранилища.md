
## 🗄️ Модуль 2: Векторное хранилище

### 3.5 Векторные базы данных: от прототипа к масштабу

Векторная база данных хранит пары (вектор, метаданные) и отвечает на запросы вида "найди K ближайших векторов к query-вектору". Это задача ANN (Approximate Nearest Neighbor) — точный NN в высокоразмерном пространстве вычислительно неосуществим при больших объёмах.

**Алгоритм HNSW** (Hierarchical Navigable Small World) — основа большинства современных векторных баз. Строит многоуровневый граф: верхние слои — разреженные длинные связи для быстрого перемещения, нижние — плотные короткие для точного поиска. Балансирует точность и скорость через параметры `ef_construction` (качество при индексации) и `ef` (качество при поиске).

**ChromaDB** — Python-библиотека с опциональным сервером. `chromadb.PersistentClient(path=...)` создаёт локальное хранилище на базе SQLite, никаких сетевых зависимостей. Использует HNSW-индекс через `hnswlib`. Практический потолок — 500 000-1 000 000 векторов при разумной латентности. Выбор для прототипа и небольших production-инсталляций.

**Qdrant** — self-hosted сервер на Rust. Архитектурные преимущества: сегментированное хранение на диске (данные переживают рестарт), гибкий scoring pipeline (dense + sparse + ColBERT в одном запросе), фильтрация по метаданным (payload-фильтры). `QdrantClient(path=...)` — local mode без сервера, работает как embedded БД, аналогично SQLite. Qdrant Local Mode имеет примерно тот же практический потолок, что и локальная Chroma (до 1-2 миллионов векторов), но выигрывает за счет того, что код, написанный для Local Mode, бесшовно переносится на серверный кластер простой сменой инициализации клиента. Практический потолок Qdrant Local ограничен исключительно объемом RAM и SSD локальной машины

**Milvus** — distributed система, designed for billion-scale. Архитектура: отдельные поды для хранения, индексации, запросов и координации. Требует Kubernetes и DevOps-зрелости. Обоснован при > 100 миллионов векторов, multi-tenancy с изоляцией коллекций, высоком throughput (>10 000 QPS).

Практическое правило выбора:

```
- Прототип / до 1M векторов: ChromaDB или Qdrant Local.
- Production / от 1M до 100M+ векторов: Qdrant (self-hosted кластер / Cloud) — отличный баланс мощности и простоты DevOps (один бинарник).
- Enterprise / сотни миллионов векторов / сложные команды: Milvus (требует K8s, выделенных команд DevOps, но дает гранулярный скейлинг отдельных микросервисов БД).
```

---

### 2.2 Развёртывание Milvus и загрузка чанков

#### Почему Milvus, а не FAISS или Chroma

```
FAISS:   библиотека, не БД. Нет персистентности, метаданных, фильтрации.
         → только для исследований и встроенных сценариев

ChromaDB: хорошо для прототипов. Потолок ~500k-1M векторов.
         → прототип, команды до 10 человек

Qdrant:  production-grade, Rust, отличная фильтрация.
         Local Mode → Kubernetes без изменения кода.
         → production до 50M-100M векторов

Milvus 2.5: billion-scale. Нативный hybrid search (30x быстрее решений 2024).
         HNSW + Sparse Inverted Index. GPU-индексы, multi-tenancy.
         → enterprise от 10M векторов, высокий throughput
```

**Milvus 2.5** (декабрь 2024) — нативная интеграция BM25 в коллекцию. Одна коллекция, один запрос, RRF встроен. Не нужен отдельный Elasticsearch.

**Milvus Lite** — для МК и разработки: работает как файл `.db` без Docker. Тот же API — код переносится в production без изменений.

```python
# Milvus Lite (разработка, МК):
client = MilvusClient("./milvus_rag.db")

# Milvus production standalone:
# client = MilvusClient("http://localhost:19530")
```

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

### 2.3 Настройка индексов и метаданных для фильтрации

#### Scalar index: без него фильтрация = full scan

В Milvus scalar fields (не векторы) без индекса требуют full scan при фильтрации. На миллионах чанков это катастрофа:

```
БЕЗ SCALAR INDEX:
  ANN по вектору → 1000 кандидатов → full scan для filter → O(N)

СО SCALAR INDEX:
  ANN по вектору ∩ Bitmap index → O(log N) ∩ O(1)
```

```python
# Создаём scalar indexes для полей фильтрации
client.create_index(
    collection_name=COLLECTION,
    field_name="doc_type",
    index_params={"index_type": "STL_SORT"}
)
```

#### Практические паттерны фильтрации

```
ПАТТЕРН 1: Временная фильтрация
  filter: f"created_at >= {int(datetime(2025,1,1).timestamp())}"
  → int64 timestamp быстрее VARCHAR-даты

ПАТТЕРН 2: Источниковая фильтрация
  filter: 'doc_type == "security_policy"'
  → только документы нужного типа

ПАТТЕРН 3: Версионирование
  filter: "version_num == 3"
  → только актуальная версия документа

ПАТТЕРН 4: Partition Key для multi-tenant
  schema.add_field("tenant_id", DataType.VARCHAR, is_partition_key=True)
  → физическая изоляция данных разных отделов/проектов
```

#### Hybrid Search с RRF: полный код

```python
from pymilvus import AnnSearchRequest, RRFRanker

def hybrid_search(query: str, top_k: int = 5, rerank_k: int = 20):
    """
    Hybrid search: Dense + Sparse → RRF слияние.
    rerank_k: запрашиваем с запасом для RRF, возвращаем top_k.
    k=60 в RRFRanker — отраслевой стандарт.
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

