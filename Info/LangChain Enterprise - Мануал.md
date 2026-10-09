
## LangChain Enterprise — от локальной модели до агентных систем

---

## ⚙️ ЧАСТЬ 0: АККАУНТЫ

```
Зарегистрировать (Либо ЛЮБЫЕ аналоги, обсуждали в Теории):
1. runpod.io       → пополнить $10 картой
(1. cloud.yandex.ru → подать заявку на GPU-квоту (обрабатывается 1-3 дня))
2. wandb.ai        → бесплатный аккаунт
3. huggingface.co  → бесплатный, получить Read и Write Access Token (Settings → Tokens)
4. github.com      → для финального repo
5. hub.docker.com  → для разворачивания image
```

---

## 📦 ЧАСТЬ 0: ИНФРАСТРУКТУРА RUNPOD

### ШАГ 0.1 — Настроить Pod

```
RunPod → Pods → Deploy a Pod

═══════════════════════════════════════════
ФИЛЬТРЫ (жать Filter GPUs перед выбором):
═══════════════════════════════════════════
  Cloud type:        Secure        ← не Community!
  CUDA versions:     12.8, 12.9, 13.0
  Minimum VRAM/GPU:  24 GB
  Disk type:         NVME быстрее SSD
  vCPUs / GPU:       чем больше, тем быстрее (например 8)
  Minimum RAM / GPU: чем больше, тем лучше (например 50 GB)
  Всё остальное:     дефолт / 0

═══════════════════════════════════════════
GPU И ШАБЛОН:
═══════════════════════════════════════════
  GPU:      НАПРИМЕР RTX 4090, 24GB (~$0.74/ч Secure Cloud)
  Template: Runpod Pytorch 2.8.0
            runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

═══════════════════════════════════════════
STORAGE:
═══════════════════════════════════════════
  Container disk:     40 GB
  Persistent storage: Network Volume
    Первый запуск:    Create Network Volume
                      Name: masterclass-vol
                      Size: 100 GB → Create
    Повторный запуск: выбрать существующий masterclass-vol
                      ❗ НЕ "Automatically create"!
    Цена volume:      ~$2/мес, данные выживают при любом краше

═══════════════════════════════════════════
ЧЕКБОКСЫ:
═══════════════════════════════════════════
  ✅ Start Jupyter notebook
  ✅ SSH terminal access

```

После запуска: нажать `Connect → JupyterLab` → откроется в браузере.

В JupyterLab: `File → New → Terminal`
Либо: **Connect to your Pod using SSH**. 

**Коннектимся:**
```bash
ssh root@213.173.99.33 -p 19432 -i ~/.ssh/id_ed25519 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=10
```

Проверим, что мы выбрали подходящую GPU в SSH: `nvidia-smi`

---

## ЧАСТЬ 1: АРХИТЕКТУРА — КАК ЭТО РАБОТАЕТ

### 1.1 Где мы находимся

Когда мы подключаемся к RunPod поду по SSH или открываем JupyterLab — мы оказываемся **внутри Docker-контейнера**. Этот контейнер создаётся из нашего образа (из МК1: `statooin/llm-fine-tuning-sft-dpo:v2`). RunPod запускает его на физическом хосте с GPU.

```
Физический сервер RunPod (хост)
│
├── Docker daemon (работает на хосте, нам недоступен)
│
└── Наш контейнер (statooin/llm-langchain:v1)
    │
    ├── /pre_start.sh  ← наш startup скрипт
    ├── /start.sh      ← RunPod базовый скрипт (запускает Jupyter + SSH)
    │
    ├── Процессы внутри контейнера:
    │   ├── ollama serve    (порт 11434, background)
    │   ├── jupyter lab     (порт 8888, запускает /start.sh)
    │   ├── python vllm...  (порт 8001, запускаем вручную в Блоке 7)
    │   └── gradio app      (порт 7860, запускаем в Блоке 7)
    │
    └── Хранилище:
        ├── /workspace/          ← Volume Disk (персистентный!)
        │   ├── ollama_models/   ← модели Ollama (сохраняются)
        │   ├── qdrant_data/     ← Qdrant local mode storage
        │   ├── chroma_data/     ← ChromaDB PersistentClient
        │   ├── data/docs/       ← корпоративные документы
        │   └── notebooks/       ← наши Jupyter ноутбуки
        │
        └── / (container disk)   ← эфемерный, 40 ГБ, исчезает при стопе
```

### 1.2 Почему Qdrant local mode — правильное решение

```
Qdrant имеет три режима:

:memory:  → данные только в RAM, исчезают при рестарте ячейки ❌
path=...  → данные на диске (SQLite), персистентны ✅
url=...   → подключение к Qdrant серверу (нам сервер не нужен)

Для МК используем: QdrantClient(path="/workspace/qdrant_data")

Ограничение local mode: до ~20 000 векторов, брутфорс-поиск вместо HNSW.
Для МК с несколькими сотнями корпоративных документов — более чем достаточно.
В production: переключиться на url= и полноценный Qdrant-сервер — одна строка кода.
```

---

## ЧАСТЬ 2: DOCKER-ОБРАЗ МК2 — СОЗДАЁМ ЛОКАЛЬНО

### 2.1 Структура проекта на локальном Linux

```bash
# Создаём на Kali:
mkdir -pv /root/projects/llm-langchain/{ipynb,data/docs}
cd /root/projects/llm-langchain
```

```
/root/projects/llm-langchain/
├── Dockerfile
├── pre_start.sh
├── ipynb/
│   ├── 01_infra.ipynb
│   ├── 02_langchain_basics.ipynb
│   ├── 03_rag.ipynb
│   ├── 04_knowledge_graph.ipynb
│   ├── 05_agents.ipynb
│   ├── 06_advanced.ipynb
│   └── 07_monitoring_ui.ipynb
└── data/
    └── docs/          ← корпоративные документы для RAG демо
```

### 2.2 Dockerfile для МК2

```dockerfile
# MK2 базируется на том же RunPod образе что и MK1
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Системные утилиты
RUN apt-get update -y && apt-get install -y curl sqlite3 pciutils lshw \
    && rm -rf /var/lib/apt/lists/*

# Ollama — устанавливаем бинарник
# Работает как отдельный процесс внутри контейнера, не требует Docker
RUN curl -fsSL https://ollama.com/install.sh | sh

# LangChain экосистема
RUN pip install \
    "langchain>=0.3.0" \
    "langchain-ollama" \
    "langchain-openai" \
    "langchain-community" \
    "langchain-huggingface" \
    "langchain-qdrant" \
    "langchain-experimental" \
    "langchain-chroma" \
    "langgraph>=0.3.0"

# Векторные БД (local mode, без серверов)
RUN pip install \
    "chromadb" \
    "qdrant-client[fastembed]" \
    "rank_bm25"

# Embedding и ML
RUN pip install \
    "sentence-transformers" \
    "pypdf" \
    "python-docx"

# Monitoring и UI
RUN pip install \
    "wandb" \
    "weave" \
    "langfuse" \
    "gradio" \
    "networkx" \
    "matplotlib"

# Проверка что torch не сломан
RUN python -c "
import torch
import langchain, langgraph, chromadb, qdrant_client
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('Все пакеты импортируются OK')
"

# Ноутбуки в образ
COPY ipynb/ /opt/masterclass/notebooks/
# Документы для RAG
COPY docs/ /opt/masterclass/docs/

# Стартовый скрипт
COPY pre_start.sh /pre_start.sh
RUN chmod +x /pre_start.sh
```

Проверяем, что всё установилось:
```bash
pip list | grep -E "langchain|langgraph"

python3 -c "
import importlib.metadata
import langchain_ollama, langchain_openai, langchain_huggingface, langchain_qdrant

print(f\"LangChain ver: {importlib.metadata.version('langchain')}\")
print(f\"LangGraph ver: {importlib.metadata.version('langgraph')}\")
print('✅ Все модули успешно импортированы и готовы к работе!')

import importlib, importlib.metadata

# Словарь: 'имя_в_pip' : 'имя_модуля_в_коде'
packages = {
    'chromadb': 'chromadb',
    'qdrant-client': 'qdrant_client', 
    'rank_bm25': 'rank_bm25',
    'sentence-transformers': 'sentence_transformers',
    'pypdf': 'pypdf',
    'python-docx': 'docx', # Обращаем внимание на разницу имен!
    'wandb': 'wandb',
    'weave': 'weave',
    'langfuse': 'langfuse',
    'gradio': 'gradio',
    'networkx': 'networkx',
    'matplotlib': 'matplotlib'
}

print('=== Аудит инфраструктуры (MLOps & UI) ===')

# Проходим циклом по всем пакетам
for pip_name, mod_name in packages.items():
    try:
        # 1. Пытаемся импортировать модуль
        importlib.import_module(mod_name)
        # 2. Если импорт успешен, запрашиваем версию
        ver = importlib.metadata.version(pip_name)
        # 3. Выводим красивый статус (с выравниванием текста)
        print(f'✅ {pip_name:<22} v{ver}')
    except Exception as e:
        # Если пакета нет или сломаны зависимости
        print(f'❌ {pip_name:<22} ОШИБКА: {e}')

print('=========================================')
"

```

### 2.3 pre_start.sh  

```bash
#!/bin/bash
set -e

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "=== Masterclass MK2 pre-start ==="

# ── 1. АВТОРИЗАЦИЯ ───────────────────────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "HuggingFace OK"
else
    echo "HF_TOKEN не задан"
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    echo "WandB OK"
else
    echo "WANDB_API_KEY не задан"
fi

# ── 2. ПОДГОТОВКА WORKSPACE ──────────────────────────────────────────────
mkdir -pv /workspace/ollama_models
mkdir -pv /workspace/qdrant_data
mkdir -pv /workspace/chroma_data
mkdir -pv /workspace/data/docs

# Копируем ноутбуки (не перезаписываем если уже изменены)
shopt -s nullglob
for f in /opt/masterclass/notebooks/*.ipynb; do
    fname=$(basename "$f")
    if [ ! -f "/workspace/$fname" ]; then
        cp "$f" /workspace/
        echo "Скопирован: $fname"
    fi
done
shopt -u nullglob

# ДОКУМЕНТЫ ДЛЯ RAG → /workspace ────────────────────────────────────
shopt -s nullglob
for f in /opt/masterclass/docs/*.txt; do
    fname=$(basename "$f")
    if [ ! -f "/workspace/data/docs/$fname" ]; then
        cp "$f" /workspace/data/docs/
        echo "Скопирован: $fname"
    else
        echo "$fname уже есть"
    fi
done
shopt -u nullglob

# ── 3. OLLAMA — ЗАПУСКАЕМ КАК ПРОЦЕСС ───────────────────────────────────
# OLLAMA_MODELS: храним в /workspace — выживают при рестарте пода
# OLLAMA_HOST:   слушаем на всех интерфейсах
export OLLAMA_MODELS=/workspace/ollama_models
OLLAMA_HOST=0.0.0.0 OLLAMA_MODELS=/workspace/ollama_models ollama serve &
echo "Ollama запущен (порт 11434)"

# ── 4. СКАЧИВАЕМ МОДЕЛИ В ФОНЕ ───────────────────────────────────────────
{
    # Ждём пока Ollama поднимется
    sleep 15

    # qwen3:8b (~5.2 ГБ)
    if ! ollama list | grep -q "qwen3:8b"; then
        echo "Скачиваем qwen3:8b (~5.2 ГБ)..."
        OLLAMA_MODELS=/workspace/ollama_models ollama pull qwen3:8b
        echo "qwen3:8b OK"
    else
        echo "qwen3:8b уже есть"
    fi

    # nomic-embed-text (~274 МБ)
    if ! ollama list | grep -q "nomic-embed-text"; then
        echo "Скачиваем nomic-embed-text (~274 МБ)..."
        OLLAMA_MODELS=/workspace/ollama_models ollama pull nomic-embed-text
        echo "nomic-embed-text OK"
    else
        echo "nomic-embed-text уже есть"
    fi

    # llama3.2-vision:11b (~7.9 ГБ) — качаем последней
    if ! ollama list | grep -q "llama3.2-vision:11b"; then
        echo "Скачиваем llama3.2-vision:11b (~7.9 ГБ)..."
        OLLAMA_MODELS=/workspace/ollama_models ollama pull llama3.2-vision:11b
        echo "llama3.2-vision:11b OK"
    else
        echo "llama3.2-vision:11b уже есть"
    fi

    echo "=== Все модели готовы! ==="
} &

echo "=== MK2 pre-start завершён. Jupyter запускается... ==="
```

**Проверка after_start_check.sh:**
```bash
#!/bin/bash

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

PASS=0
FAIL=0

ok()   { echo "  OK   $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL $1"; FAIL=$((FAIL+1)); }

echo ""
echo "============================================"
echo "   ПРОВЕРКА MK2 pre-start"
echo "============================================"

# ── 1. АВТОРИЗАЦИЯ ───────────────────────────────
echo ""
echo "── 1. АВТОРИЗАЦИЯ ──────────────────────────"

if python -c "from huggingface_hub import whoami; whoami()" &>/dev/null; then
    HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null)
    ok "HuggingFace авторизован (пользователь: $HF_USER)"
else
    fail "HuggingFace НЕ авторизован"
fi

if python -c "import wandb; api = wandb.Api(); print(api.viewer['username'])" &>/dev/null; then
    WB_USER=$(python -c "import wandb; api = wandb.Api(); print(api.viewer['username'])" 2>/dev/null)
    ok "WandB авторизован (пользователь: $WB_USER)"
else
    fail "WandB НЕ авторизован"
fi

# ── 2. ПАПКИ WORKSPACE ───────────────────────────
echo ""
echo "── 2. ПАПКИ /workspace ─────────────────────"

for dir in ollama_models qdrant_data chroma_data data/docs; do
    if [ -d "/workspace/$dir" ]; then
        ok "/workspace/$dir существует"
    else
        fail "/workspace/$dir ОТСУТСТВУЕТ"
    fi
done

# ── 3. НОУТБУКИ ──────────────────────────────────
echo ""
echo "── 3. НОУТБУКИ в /workspace ────────────────"

NB_COUNT=0
for nb in /workspace/*.ipynb; do
    [ -f "$nb" ] && NB_COUNT=$((NB_COUNT+1))
done

if [ "$NB_COUNT" -gt 0 ]; then
    ok "Найдено $NB_COUNT ноутбук(ов):"
    for nb in /workspace/*.ipynb; do
        [ -f "$nb" ] && echo "       $(basename $nb)"
    done
else
    fail "Ноутбуки в /workspace НЕ найдены"
fi

# ── 4. OLLAMA ПРОЦЕСС ────────────────────────────
echo ""
echo "── 4. OLLAMA ───────────────────────────────"

if pgrep -x "ollama" &>/dev/null; then
    ok "Ollama процесс запущен (PID: $(pgrep -x ollama))"
else
    fail "Ollama процесс НЕ запущен"
fi

if curl -s --max-time 3 http://127.0.0.1:11434/api/tags &>/dev/null; then
    ok "Ollama API отвечает на порту 11434"
else
    fail "Ollama API НЕ отвечает (ещё грузится или упал)"
fi

# ── 5. МОДЕЛИ ────────────────────────────────────
echo ""
echo "── 5. МОДЕЛИ OLLAMA ────────────────────────"

if curl -s --max-time 3 http://127.0.0.1:11434/api/tags &>/dev/null; then
    MODELS=$(curl -s http://127.0.0.1:11434/api/tags)

    for model in "qwen3:8b" "nomic-embed-text" "llama3.2-vision:11b"; do
        if echo "$MODELS" | grep -q "$model"; then
            ok "$model скачана"
        else
            fail "$model НЕ скачана (ещё качается или ошибка)"
        fi
    done
else
    fail "Ollama API недоступен — модели не проверить"
fi

# ── ИТОГ ─────────────────────────────────────────
echo ""
echo "============================================"
echo "   ИТОГ: OK=$PASS  FAIL=$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "   ВСЁ ГОТОВО — можно работать!"
else
    echo "   ЕСТЬ ПРОБЛЕМЫ — см. FAIL выше"
fi
echo "============================================"
echo ""
```
### 2.4 Сборка образа на Linux и публикация

```bash
# На Linux:
cd /root/projects/llm-langchain

# Сначала копируем ноутбуки в папку (из RunPod по SCP):
scp -i ~/.ssh/id_ed25519 -P ПОРТ \
    root@IP_МК2_ПОДА:/workspace/*.ipynb ./ipynb/

# Собрать образ (займёт 15-25 мин, скачивается base layer)
docker build --platform linux/amd64 -t statooin/llm-langchain:v1 .

# Проверить что образ создан:
docker images | grep llm-langchain
```

Ожидаемый вывод:
```
REPOSITORY                  TAG       SIZE
statooin/llm-langchain      v1        18.3GB
```

```bash
# Залить на Docker Hub:
docker login
docker push statooin/llm-langchain:v1
```

Важно: на Docker Hub загружается только наш слой (~3-4 ГБ). Base layer RunPod уже закеширован.

### 2.5 Проверка запуска

После старта пода (~60-120 сек):

```bash
# Подключаемся по SSH:
ssh root@IP_ПОДА -p ПОРТ -i ~/.ssh/id_ed25519

# Проверяем что Ollama запущена:
curl http://localhost:11434/api/tags
```

Ожидаемый вывод (после ~15 сек запуска):
```json
{"models":[{"name":"qwen3:8b","size":5234567890}]}
```

```bash
# Проверяем что /workspace смонтирован:
df -h /workspace
# Ожидаем: ~839T mfs#euro-3.runpod.net:9421 (сетевой том)

# Смотрим что скачивается в фоне:
tail -f /workspace/ollama_download.log 2>/dev/null || \
    watch -n 5 'ollama list'

# Проверяем Python-пакеты:
python -c "
import langchain, langgraph, chromadb, qdrant_client
from langchain_ollama import ChatOllama
print('Все пакеты OK ✅')
"
```

---

## ЧАСТЬ 3: ИНФРАСТРУКТУРА И ПЕРВЫЙ ТЕСТ (БЛОК 1)

### Теория (15 мин)

**📌 Термин: On-premise** — развёртывание программного обеспечения на собственных серверах организации. Данные не покидают контролируемый контур.

**📌 Термин: Inference** — процесс генерации ответа уже обученной моделью. Не изменяет веса. В отличие от training (обучения, МК1).

**📌 Термин: Token** — минимальная единица текста для LLM. Примерно 0.75 слова. GPT-4o: $5-15 за миллион токенов. Разработчик за день: ~1 млн токенов. Команда 50 человек: $250-750/день. RTX 4090 ($2 000) окупается за 2-3 недели.

**Почему нельзя `docker compose` внутри RunPod пода:**

Мы уже находимся внутри Docker-контейнера. Запустить Docker внутри Docker (DinD) требует привилегированного режима, который RunPod не предоставляет для GPU-подов. Поэтому Ollama запускается как фоновый Linux-процесс, а Qdrant и ChromaDB работают в Python-процессе в local mode.

**Фреймворки для запуска LLM:**

```
Ollama
  Принцип: один бинарник = менеджер моделей + inference сервер
           запускается как Linux-процесс: ollama serve &
           API: OpenAI-совместимый на localhost:11434
  Для МК:  разработка, демонстрация (Блоки 1-6)
  Минус:   один запрос за раз (нет batching)

vLLM
  Принцип: PagedAttention — KV-кеш как виртуальная память
           80-120 токенов/сек на RTX 4090
           OpenAI-совместимый API
  Для МК:  production демо в Блоке 7
  Запуск:  python -m vllm.entrypoints.openai.api_server ...

SGLang
  Принцип: RadixAttention — кеширует общие префиксы промптов
           при RAG: системный промпт + контекст одинаков → кешируется
           прирост 2-5× для RAG нагрузки
  Для МК:  теоретическое упоминание
```


### notebook_01_infra_web_api.ipynb

```python
# =============================================================================
# ЯЧЕЙКА 0: Проверка доступности LLM API
# Провайдер читается из .env → PROVIDER=yandex|deepseek|qwen3
# =============================================================================
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FIX 3.1: явный путь — CWD внутри ноутбука = /workspace/notebooks,
# без аргумента load_dotenv() не найдёт /workspace/.env
load_dotenv("/workspace/.env")


def get_llm() -> ChatOpenAI:
    """
    Фабрика LLM: читает PROVIDER из env, возвращает ChatOpenAI.
    Все три провайдера используют OpenAI-совместимый endpoint.

    Raises:
        ValueError: если PROVIDER неизвестен или отсутствуют обязательные переменные.
    """
    provider = os.getenv("PROVIDER", "yandex").lower()

    if provider == "yandex":
        api_key   = os.getenv("YANDEX_API_KEY")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        if not api_key or not folder_id:
            raise ValueError("Не заданы YANDEX_API_KEY или YANDEX_FOLDER_ID в .env")
        return ChatOpenAI(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=f"gpt://{folder_id}/yandexgpt/latest",
            # x-folder-id обязателен для YandexGPT — нестандартный OpenAI заголовок
            default_headers={"x-folder-id": folder_id},
            temperature=0.7,
            max_tokens=200,
        )

    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Не задан DEEPSEEK_API_KEY в .env")
        return ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=200,
        )

    elif provider == "qwen3":
        # Ollama на хосте → host.docker.internal работает благодаря extra_hosts в compose
        return ChatOpenAI(
            base_url="http://host.docker.internal:11434/v1",
            api_key="EMPTY",  # Ollama/vLLM требуют непустую строку
            model="qwen3:8b",
            temperature=0.7,
            max_tokens=200,
        )

    else:
        raise ValueError(
            f"Неизвестный PROVIDER='{provider}'. Допустимо: yandex | deepseek | qwen3"
        )


# Создаём один раз — llm доступен во всех следующих ячейках
llm = get_llm()

response = llm.invoke([HumanMessage(content="Кто ты? Ответь одним предложением.")])

print(f"✅ API доступен")
print(f"   Провайдер : {os.getenv('PROVIDER', 'yandex').upper()}")
print(f"   Ответ     : {response.content}")
```

---

### notebook_01_infra_local.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Проверяем GPU, пакеты, диск
# ⚠️ Если GPU не видно — проверить что Pod запущен с GPU шаблоном
# ⚠️ Если BF16 = False — использовать float16 в следующих ячейках

import subprocess, torch

result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA OK:       {torch.cuda.is_available()}")
print(f"GPU:           {torch.cuda.get_device_name(0)}")
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
used  = torch.cuda.memory_reserved(0) / 1024**3
print(f"VRAM:               {total:.1f} GB всего, {total-used:.1f} GB свободно")
print(f"BF16 support:  {torch.cuda.is_bf16_supported()}")

# Проверяем диск
disk_info = subprocess.run(['df','-h','/workspace'], capture_output=True, text=True)
print(f"\nДиск /workspace:    {disk_info.stdout.split()[-4]} свободно")
```

Ожидаемый вывод:
```
+----------------------------------------------------------+
| NVIDIA GeForce RTX 4090                  24 GB           |
| CUDA 12.8                               0% Util          |
+----------------------------------------------------------+

PyTorch:       2.8.0+cu128
CUDA OK:       True
GPU:           NVIDIA GeForce RTX 4090
VRAM:          24.0 GB
BF16 support:  True
```

```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Проверяем что Ollama работает
# ⚠️ Ollama запущен в /pre_start.sh как background process

import requests, time

def wait_for_ollama(max_attempts=10):
    for i in range(max_attempts):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                models = r.json().get("models", [])
                print(f"Ollama OK ✅ — доступных моделей: {len(models)}")
                for m in models:
                    size_gb = m.get('size', 0) / 1024**3
                    print(f"  - {m['name']} ({size_gb:.1f} ГБ)")
                return True
        except Exception as e:
            print(f"Попытка {i+1}/{max_attempts}: Ollama ещё стартует...")
            time.sleep(3)
    return False

if not wait_for_ollama():
    print("⚠️ Ollama не ответила. Запустим вручную:")
    print("! OLLAMA_MODELS=/workspace/ollama_models ollama serve &")
```

Ожидаемый вывод:
```
Ollama OK ✅ — доступных моделей: 3
  - qwen3:8b (5.2 ГБ)
  - nomic-embed-text (0.3 ГБ)
  - llama3.2-vision:11b (7.9 ГБ)
```

```python
# ── ЯЧЕЙКА 3 ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Первый запрос к локальной LLM через OpenAI API
# ⚠️ Показать: тот же API что у OpenAI, только base_url другой

import requests, json

response = requests.post(
    "http://localhost:11434/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "Отвечай кратко и по делу."},
            {"role": "user", "content": "Ты локальная или облачная модель? Откуда ты работаешь?"}
        ]
    }
)

data = response.json()
print(f"Ответ: {data['choices'][0]['message']['content']}")
print(f"Токены: input={data['usage']['prompt_tokens']}, output={data['usage']['completion_tokens']}")
```

Ожидаемый вывод:
```
Ответ: Я локальная языковая модель Qwen3, запущенная через Ollama
прямо на вашем GPU-сервере. Никакие данные не покидают ваш контур.
Токены: input=28, output=35
```

---

```python
# ── ЯЧЕЙКА 4 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Структура /workspace — где что лежит

import os, subprocess

def show_workspace():
    for item in sorted(os.listdir("/workspace")):
        path = f"/workspace/{item}"
        if os.path.isdir(path):
            size = subprocess.run(['du', '-sh', path],
                capture_output=True, text=True).stdout.split()[0]
            print(f"  📁 {item}/  ({size})")
        else:
            size = os.path.getsize(path) / 1024**2
            print(f"  📄 {item}  ({size:.1f} МБ)")

print("/workspace:")
show_workspace()
```

Ожидаемый вывод:
```
/workspace:
  📁 chroma_data/  (0)
  📁 data/  (45М)
  📁 ollama_models/  (13.4G)
  📁 qdrant_data/  (0)
  📄 01_infra.ipynb  (0.1 МБ)
  📄 02_langchain_basics.ipynb  ...
```

---

## ЧАСТЬ 4: LANGCHAIN + МОДЕЛИ (БЛОК 2)

**📌 Термин: LangChain** — фреймворк для построения AI-приложений. Стандартизирует интерфейс к LLM, векторным базам, инструментам. Единый код — любой провайдер.

**📌 Термин: Runnable** — базовый интерфейс LangChain. Любой объект с `.invoke()`, `.stream()`, `.batch()`. Промпт, LLM, парсер, ретривер — всё Runnable. Соединяются оператором `|`.

**📌 Термин: LCEL** (LangChain Expression Language) — `prompt | llm | parser`. Автоматически добавляет streaming, async, retry, параллелизм.

**📌 Термин: Structured Output** — режим при котором LLM гарантированно возвращает валидный JSON по заданной Pydantic-схеме.

**📌 Термин: Chat Template** — специальная разметка диалога для модели. Qwen3 использует `<|im_start|>` / `<|im_end|>` токены. LangChain применяет её автоматически.

**📌 Термин: Thinking mode (режим рассуждений)** — функция Qwen3: генерирует цепочку `<think>...</think>` перед ответом. Для RAG и агентов отключаем: `reasoning=False`.

**Связка:**

```python
# Ollama (разработка и демо):
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False)

# vLLM с T-lite-dpo:
from langchain_openai import ChatOpenAI
llm_prod = ChatOpenAI(base_url="http://localhost:8001/v1", model="tlite-dpo")

# web API YandexAPI
llm = ChatOpenAI(
    base_url="https://llm.api.cloud.yandex.net/v1",
    api_key=_api_key,
    model=f"gpt://{_folder_id}/yandexgpt/latest",
    default_headers={"x-folder-id": _folder_id},
    temperature=0.1,
    max_tokens=512,
)

# Меняем один объект llm — весь остальной код не трогаем
```

### notebook_02_langchain_basics.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Базовое подключение к локальной LLM

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    reasoning=False,  # выключаем <think> теги для продуктивных задач
)

# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Базовое подключение к YandexGPT

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv("/workspace/.env")

_api_key   = os.getenv("YANDEX_API_KEY")
_folder_id = os.getenv("YANDEX_FOLDER_ID")

llm = ChatOpenAI(
    base_url="https://llm.api.cloud.yandex.net/v1",
    api_key=_api_key,
    model=f"gpt://{_folder_id}/yandexgpt/latest",
    default_headers={"x-folder-id": _folder_id},
    temperature=0.1,
    max_tokens=512,
)

response = llm.invoke("Привет! Ты локальная модель?")
print(response.content)
print(f"Использовано токенов: {response.usage_metadata}")
```

```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Первая LCEL цепочка: промпт → LLM → парсер

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты корпоративный ассистент службы поддержки. Отвечай чётко и по делу."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"question": "Как сбросить пароль от корпоративной почты?"})
print(result)
```

Ожидаемый вывод:
```
Для сброса пароля корпоративной почты:
1. Перейдите на портал IT-поддержки: it.company.local
2. Нажмите "Забыли пароль?"
3. Введите корпоративный email и следуйте инструкциям

При отсутствии доступа к почте — обратитесь в IT: ext. 1234
```

---

Теперь мы обращаемся к пайплайну через метод `.stream()` вместо стандартного `.invoke()`. Через цикл `for` мы перехватываем каждый отдельный токен (кусок текста) и моментально выводим его на экран через `print(flush=True)`:

1. **Паттерн Streaming:** Мы показываем, как забирать ответ у модели "на лету", по мере его формирования.
2. **Метрику TTFT (Time To First Token):** Мы наглядно видим золотой стандарт UX в AI. Без стриминга пользователь смотрел бы на зависший интерфейс 10–30 секунд и думал, что всё сломалось. Со стримингом мы отдаем первый символ меньше чем за 0.5 секунды — система кажется мгновенной и отзывчивой.

```python
# ── ЯЧЕЙКА 3a ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Streaming — токены приходят по одному в реальном времени
# ⚠️ Ключевой момент для UX: TTFT < 0.5 сек вместо ожидания 10-30 сек

print("Streaming (токены появляются по мере генерации):")
print("─" * 50)
for chunk in chain.stream({"question": "Объясни разницу между SQL и NoSQL за 3 предложения."}):
    print(chunk, end="", flush=True)
print("\n" + "─" * 50)
```

```python
# ── ЯЧЕЙКА 3b ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Продвинутый LCEL конвейер: генерация + сайд-эффекты (запись файла)

import os
from typing import Iterator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def write_to_disk_stream(stream: Iterator[str]) -> Iterator[str]:
    """
    Правильный перехватчик (Middleware) для стриминга.
    Принимает поток чанков, пишет каждый на диск и прокидывает (yield) дальше,
    не блокируя работу LCEL-цепочки и не заставляя ждать конца генерации.
    """
    for chunk in stream:
        with open("/workspace/data/postmortem.md", "a", encoding="utf-8") as f:
            f.write(chunk)
        yield chunk  # Отдаем токен дальше по конвейеру (в консоль)

# Промпт: задаем строгую структуру ответа для системного инженера
prompt = ChatPromptTemplate.from_template("""
Ты — Senior SRE инженер. Составь шаблон Post-Mortem отчета для следующего инцидента.
Инцидент: {incident}

Требования:
- Строгий Markdown-формат.
- Обязательные секции: "Влияние", "Хронология", "Корневая причина", "Action Items".
- Без лишних приветствий, сразу к делу.
""")

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False)

# Подготовка: создаем директорию и очищаем файл перед запуском
os.makedirs("/workspace/data", exist_ok=True)
with open("/workspace/data/postmortem.md", "w", encoding="utf-8") as f:
    f.write("<!-- АВТОГЕНЕРАЦИЯ POST-MORTEM -->\n\n")

# ── LCEL Pipeline: 5 компонентов конвейера ─────────────────────────────
# 1. RunnablePassthrough — берет входящую строку и оборачивает её в словарь
# 2. Prompt              — вставляет инцидент в шаблон
# 3. LLM                 — генерирует ответ
# 4. StrOutputParser     — срезает служебные теги моделей, возвращает чистый текст
# 5. write_to_disk_stream— пишет токены в файл, не прерывая стрим

postmortem_chain = (
    {"incident": RunnablePassthrough()}     # Шаг 1: str -> dict
    | prompt                                # Шаг 2: dict -> PromptValue
    | llm                                   # Шаг 3: PromptValue -> AIMessage
    | StrOutputParser()                     # Шаг 4: AIMessage -> str (поток)
    | RunnableLambda(write_to_disk_stream)  # Шаг 5: str -> str (с записью на диск)
)

print("ГЕНЕРАЦИЯ POST-MORTEM ОТЧЕТА (параллельно пишется в /workspace/data/postmortem.md):\n")

# Запускаем стриминг. Передаем просто строку, RunnablePassthrough сам положит её в ключ "incident"
test_incident = "Упал production-кластер Kubernetes из-за истекшего TLS сертификата etcd"

for chunk in postmortem_chain.stream(test_incident):
    print(chunk, end="", flush=True)

print("\n\n✅ Отчет успешно сохранен на диск!")
```

```python
# ── ЯЧЕЙКА 3c ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Продвинутый LCEL конвейер с глубокой телеметрией (astream_events)

import os
from typing import AsyncIterator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Асинхронный перехватчик (Middleware) для записи на диск
async def write_to_disk_stream(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Генератор перехватывает токены на лету, пишет на диск и отдает дальше.
    """
    async for chunk in stream:
        with open("/workspace/data/postmortem.md", "a", encoding="utf-8") as f:
            f.write(chunk)
        yield chunk

# 2. Строгий промпт
prompt = ChatPromptTemplate.from_template("""
Ты — Senior SRE инженер. Составь шаблон Post-Mortem отчета для следующего инцидента.
Инцидент: {incident}

Требования:
- Строгий Markdown-формат.
- Обязательные секции: "Влияние", "Хронология", "Корневая причина", "Action Items".
- Без лишних приветствий, сразу к делу.
""")

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False)

# 3. Подготовка файловой системы
os.makedirs("/workspace/data", exist_ok=True)
with open("/workspace/data/postmortem.md", "w", encoding="utf-8") as f:
    f.write("<!-- АВТОГЕНЕРАЦИЯ POST-MORTEM -->\n\n")

# 4. Сборка пайплайна
postmortem_chain = (
    {"incident": RunnablePassthrough()}          
    | prompt                                     
    | llm                                        
    | StrOutputParser()                          
    | write_to_disk_stream
)

print("ГЕНЕРАЦИЯ POST-MORTEM ОТЧЕТА (Глубокая телеметрия через astream_events):\n")

test_incident = "Упал production-кластер Kubernetes из-за истекшего TLS сертификата etcd"

# 5. ДЕМОНСТРАЦИЯ: Асинхронный перехват событий графа
async for event in postmortem_chain.astream_events(
    test_incident,
    version="v2"
):
    kind = event["event"]
    name = event.get("name", "")

    # Событие 1: Старт всей цепочки
    if kind == "on_chain_start" and name == "RunnableSequence":
        print("[⚙️ Инициализация пайплайна и подготовка промпта...]")
        
    # Событие 2: Модель начала "думать"
    elif kind == "on_chat_model_start":
        print("[▶ LLM начала генерацию...]\n")
        print("─" * 60)

    # Событие 3: Прилетел новый токен
    elif kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)

    # Событие 4: Завершение работы нейросети
    elif kind == "on_chat_model_end":
        print("\n" + "─" * 60)
        print("[⏹ LLM успешно завершила генерацию]")

print("\n[✓ Файл /workspace/data/postmortem.md сохранен и содержит полный лог инцидента!]")
```

---

Мы описываем жесткую структуру данных через Pydantic (класс `SupportTicket`) и привязываем её к LLM с помощью метода `.with_structured_output()`. Затем передаем "человеческий" текст жалобы, а модель сама парсит его, классифицирует проблему и возвращает готовый типизированный Python-объект:

1. **Жесткие гарантии (Contract Testing):** Мы показываем, что модель теперь возвращает не непредсказуемый текст, а 100% валидный JSON. Схема физически ограничивает генерацию (через `tool_choice`).
2. **Бесшовная интеграция с бэкендом:** Мы наглядно объясняем, как легко встраивать LLM в классическую архитектуру. Нам больше не нужно парсить ответы регулярками — мы получаем готовый объект для записи в БД (или отправки в Jira/ServiceDesk).

```python
# ── ЯЧЕЙКА 4a ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Structured Output — гарантированный JSON
# ⚠️ .with_structured_output() — современный API (2025+)
# ⚠️ Модель физически НЕ МОЖЕТ вернуть невалидный JSON

from pydantic import BaseModel, Field
from typing import Literal

class SupportTicket(BaseModel):
    """Тикет поддержки — модель возвращает ИМЕННО ЭТУ структуру"""
    category: Literal["hardware", "software", "network", "access", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(description="Краткое описание, 1 предложение")
    requires_callback: bool = Field(description="Нужен ли обратный звонок")

# Биндим Pydantic-схему к LLM
structured_llm = llm.with_structured_output(SupportTicket)

ticket = structured_llm.invoke(
    "У меня не работает VPN с утра. Завтра важная встреча с клиентом, нужно срочно. "
    "Телефон для связи оставил у секретаря."
)

print(f"Категория:      {ticket.category}")
print(f"Приоритет:      {ticket.priority}")
print(f"Описание:       {ticket.summary}")
print(f"Нужен звонок:   {ticket.requires_callback}")
print(f"\nJSON:\n{ticket.model_dump_json(indent=2)}")
```

Ожидаемый вывод:
```
Категория:      network
Приоритет:      high
Описание:       VPN недоступен, срочное восстановление доступа необходимо
Нужен звонок:   True

JSON:
{
  "category": "network",
  "priority": "high",
  "summary": "VPN недоступен, срочное восстановление доступа необходимо",
  "requires_callback": true
}
```

```python
# ── ЯЧЕЙКА 4b ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Structured Output + Семантическая маршрутизация (Router)
# Извлекаем строгий JSON из текста, показываем его и на лету маршрутизируем.

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
import datetime

# 1. Строгая схема данных (Контракт)
class SupportTicket(BaseModel):
    """Тикет поддержки — модель возвращает ИМЕННО ЭТУ структуру"""
    category: Literal["hardware", "software", "network", "access", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(description="Краткое описание проблемы, 1 предложение")
    requires_callback: bool = Field(description="Нужен ли обратный звонок (если есть номер)")

# Оборачиваем LLM: теперь она гарантированно возвращает объект SupportTicket
structured_llm = llm.with_structured_output(SupportTicket)

# 2. ПРОКСИ-ПЕРЕХВАТЧИК ДЛЯ ДЕБАГА
def debug_and_pass(ticket: SupportTicket) -> SupportTicket:
    print(f"📦 [DEBUG] Извлечённый JSON:\n{ticket.model_dump_json(indent=2)}\n")
    return ticket

# 3. Специализированные узлы-обработчики (Handlers)
def handle_critical_network(ticket: SupportTicket) -> str:
    return f"🚨 [PAGERDUTY] СЕТЕВОЙ ИНЦИДЕНТ! Звоним дежурному NOC-инженеру.\n   Детали: {ticket.summary}"

def handle_access(ticket: SupportTicket) -> str:
    return f"🔑 [IAM AUTO-APPROVE] Запрос прав доступа. Отправлено на ревью тимлиду.\n   Описание: {ticket.summary}"

def handle_general(ticket: SupportTicket) -> str:
    return f"📝 [JIRA BACKLOG] Создан стандартный тикет (Приоритет: {ticket.priority.upper()}).\n   Категория: {ticket.category}"

# 4. Семантический маршрутизатор (RunnableBranch)
router = RunnableBranch(
    (lambda ticket: ticket.category == "network" and ticket.priority in ["high", "critical"],
     RunnableLambda(handle_critical_network)),
    (lambda ticket: ticket.category == "access",
     RunnableLambda(handle_access)),
    RunnableLambda(handle_general)
)

# 5. Супер-пайплайн с дебаг-узлом посередине
ticket_pipeline = structured_llm | RunnableLambda(debug_and_pass) | router

print("=== СЕМАНТИЧЕСКАЯ МАРШРУТИЗАЦИЯ ИНЦИДЕНТОВ ===\n")

# Тест 1: Завуалированная проблема с сетью
q1 = "У меня не работает VPN с утра. Завтра важная встреча с клиентом, нужно срочно. Звоните на мобильный."
print("◯ ЗАПРОС 1:", q1)
print("🔘 РЕЗУЛЬТАТ:\n" + ticket_pipeline.invoke(q1))
print("-" * 60)

# Тест 2: Запрос доступов
q2 = "Добавьте меня в группу разработчиков GitLab для проекта frontend-app."
print("◯ ЗАПРОС 2:", q2)
print("🔘 РЕЗУЛЬТАТ:\n" + ticket_pipeline.invoke(q2))
print("-" * 60)

# Тест 3: Бытовая проблема
q3 = "У меня мышка заедает, колесико не крутится. Замените при случае, не горит."
print("◯ ЗАПРОС 3:", q3)
print("🔘 РЕЗУЛЬТАТ:\n" + ticket_pipeline.invoke(q3))
print("-" * 60)

# ─── СВЯЗКА С ЯЧЕЙКОЙ 6c ─────────────────────────────────────────────────────
# Structured Output → файл на диске → read_ticket_file в агенте
#
# ЯЧЕЙКА 6c ожидает файл /workspace/ticket_TK2025001847.txt
# Здесь мы берём реальный запрос про FreeIPA/EMP042,
# прогоняем через structured_llm для классификации,
# и сохраняем на диск в формате который read_ticket_file прочитает.

print("\n" + "="*60)
print("🔗 СВЯЗКА С ЯЧЕЙКОЙ 6c")
print("   Structured Output → файл на диске → агент")
print("="*60)

# Исходный запрос — именно тот сценарий который агент будет обрабатывать
raw_query_for_agent = (
    "Необходимо срочно установить FreeIPA client на хост "
    "server-prod-app-04.corp.local для нового сотрудника "
    "Петрова Александра (EMP042, Senior DevOps Engineer). "
    "ОС: Astra Linux 1.8. Сотрудник выходит на работу завтра. "
    "Менеджер Сидоров Иван, звоните на мобильный."
)

print(f"\n📥 Входящий запрос:\n   {raw_query_for_agent}\n")

# structured_llm классифицирует запрос → SupportTicket объект
# Это демонстрирует как Structured Output питает downstream-системы
ticket_obj: SupportTicket = structured_llm.invoke(raw_query_for_agent)

print(f"📦 Structured Output (то что вернула LLM):")
print(ticket_obj.model_dump_json(indent=2))

# Форматируем в человекочитаемый тикет — именно этот формат
# будет читать read_ticket_file в ЯЧЕЙКЕ 6c
ticket_text = f"""ТИКЕТ #TK-2025-001847
Дата создания: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Приоритет: {ticket_obj.priority.upper()}
Категория: {ticket_obj.category}
Статус: ОТКРЫТ
Обратный звонок: {'Да' if ticket_obj.requires_callback else 'Нет'}

═══════════════════════════════════════════════════════════════════════════════

ТЕМА: {ticket_obj.summary}

ДЕТАЛИ ЗАПРОСА:
──────────────────────────────────────────────────────────────────────────────
- Имя сотрудника: Петров Александр (EMP042)
- Должность: Senior DevOps Engineer
- Отдел: Инфраструктура и облачные технологии
- Дата начала работы: завтра
- ОС хоста: Astra Linux 1.8
- Хост для подключения: server-prod-app-04.corp.local
- IP адрес хоста: 10.2.15.48

ТРЕБУЕМЫЕ ДЕЙСТВИЯ:
──────────────────────────────────────────────────────────────────────────────
1. Установить клиентское ПО FreeIPA на хост server-prod-app-04
2. Настроить аутентификацию Kerberos для домена corp.astra
3. Добавить EMP042 в группы: devops-team, infrastructure-admins, sudo-users

Менеджер по развитию: Сидоров Иван
Контакт: i.sidorov@corp.local
═══════════════════════════════════════════════════════════════════════════════
"""

# Сохраняем на диск — именно по пути который ожидает read_ticket_file в 6c
ticket_path = "/workspace/ticket_TK2025001847.txt"
with open(ticket_path, "w", encoding="utf-8") as f:
    f.write(ticket_text)

print(f"\n💾 Тикет сохранён на диск: {ticket_path}")
print(f"📏 Размер файла: {len(ticket_text)} символов")
print(f"\n📋 Содержимое файла:")
print("─" * 60)
print(ticket_text)
print("─" * 60)
print("\n✅ ГОТОВО — ЯЧЕЙКА 6c можно запускать")
print("   read_ticket_file найдёт файл по пути:")
print(f"   {ticket_path}")
```

---

1. **`.with_config(run_name="...")`** — супер фича LangChain. Маркируем узлы графа своими именами. Без этого в событиях был бы безликий `RunnableLambda`. Теперь в логах мы чётко видим, кто сломался (`Cloud_GPT_Primary`) и кто пришёл на помощь (`Local_Qwen3_Backup`).
2. **Перехват `on_chain_error**` — извлекаем текст ошибки прямо из ивента `event["data"].get("error")`. Инженеры видят, что LangChain ничего не скрывает: ошибка логируется внутри пайплайна, но снаружи приложение продолжает работать.

```python
# ── ЯЧЕЙКА 5 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Отказоустойчивость (Fallbacks) + Телеметрия + Стриминг
# Graceful Degradation: бесшовное переключение на резервную модель с отладкой событий.

import asyncio
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Эмулируем "упавший" облачный/внешний API через асинхронную функцию.
# Это гарантирует, что мы не зависнем на уровне сетевых сокетов ОС,
# а упадем ровно через 5 секунд (наш искусственный таймаут).
async def simulate_broken_api(prompt_value):
    await asyncio.sleep(5)
    raise ConnectionError("Timeout 504: Cloud LLM Gateway is down!")

# Оборачиваем функцию и даем ей имя для красивых логов в телеметрии
broken_primary_llm = RunnableLambda(simulate_broken_api).with_config(run_name="Cloud_GPT_Primary")

# 2. Наша рабочая резервная модель (Backup).
llm = llm
# llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False)
backup_llm = llm.with_config(run_name="Local_Qwen3_Backup")

# 3. ДЕМОНСТРАЦИЯ WITH_FALLBACKS:
# Привязываем резерв. LangChain сам перехватит ошибку из Cloud_GPT 
# и прозрачно перенаправит промпт в Local_Qwen3.
resilient_llm = broken_primary_llm.with_fallbacks([backup_llm])

# 4. Собираем стандартную LCEL-цепочку
prompt = ChatPromptTemplate.from_template(
    "Ты Senior DevOps. Объясни коротко (1-2 предложения): что такое {concept} в Kubernetes?"
)

resilient_chain = prompt | resilient_llm | StrOutputParser()

print("=== Демонстрация Fallback Mechanism + Телеметрия ===")
print("🌐 Запрос отправлен в пайплайн...\n")

# 5. ЗАПУСК СО СТРИМИНГОМ И ПЕРЕХВАТОМ СОБЫТИЙ
async for event in resilient_chain.astream_events(
    {"concept": "Pod Disruption Budget (PDB)"},
    version="v2"
):
    kind = event["event"]
    name = event.get("name", "")
    
    # Событие: Началось обращение к основной "облачной" модели
    if kind == "on_chain_start" and name == "Cloud_GPT_Primary":
        print(f"[{name}] ⏳ Ожидание ответа от облака (таймаут 5 сек)...", flush=True)
        
    # Событие: Основная модель УПАЛА С ОШИБКОЙ
    elif kind == "on_chain_error" and name == "Cloud_GPT_Primary":
        err_msg = event["data"].get("error")
        print(f"[{name}] 💥 УПАЛ С ОШИБКОЙ: {err_msg}")
        print("[LangChain] 🔄 Автоматически переключаю на резервную модель...")
        
    # Событие: Резервная локальная модель начала работу
    elif kind == "on_chat_model_start" and name == "Local_Qwen3_Backup":
        print(f"\n[{name}] ▶ Начала генерацию ответа:\n")
        print("─" * 60)
        
    # Событие: Стриминг токенов от резервной модели
    elif kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
        
    # Событие: Резервная модель закончила работу
    elif kind == "on_chat_model_end" and name == "Local_Qwen3_Backup":
        print("\n" + "─" * 60)
        print(f"[{name}] ⏹ Успешно завершила генерацию")

print("\n✅ Пайплайн отработал без остановки скрипта (Graceful Degradation)!")
```

---

Программно генерируем тестовый скриншот с классической инфраструктурной проблемой (ошибка авторизации LDAP). Модель "читает" текст прямо с пикселей (OCR), анализирует суть сбоя и возвращает текстом причину и решение:

1. **Мультимодальность (AIOps):** Мы показываем, что современные локальные LLM вышли за рамки чистого текста и могут "видеть" системы глазами пользователя, заменяя устаревший LLaVA.
2. **Автоматизация L1-поддержки:** Мы демонстрируем готовый паттерн корпоративного хелпдеска. Пользователь просто кидает скриншот непонятной ошибки в Service Desk или Slack, а система автоматически распознает проблему и выдает инструкцию еще до подключения живого Ops-инженера.

```bash
# в bash
OLLAMA_MODELS=/workspace/ollama_models ollama pull gemma3:4b
```

```python
# или в Ячейке
import subprocess
subprocess.run(["ollama", "pull", "gemma3:4b"], check=True)
```

```python
# ── ЯЧЕЙКА 6 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Мультимодальность: анализ скриншота ошибки
# ⚠️ gemma3:4b — работает с Ollama 0.32+, ~3.5 GB VRAM

import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

vision_llm = ChatOllama(
    model="gemma3:4b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

def analyze_image(image_path: str, question: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img_b64}",
        },
        {"type": "text", "text": question}
    ])
    return vision_llm.invoke([message]).content

# Создаём тестовый скриншот
import subprocess
subprocess.run(["python", "-c", """
from PIL import Image, ImageDraw
img = Image.new('RGB', (600, 200), color='white')
draw = ImageDraw.Draw(img)
draw.text((20, 20),  'Error: Connection refused', fill='red')
draw.text((20, 60),  'Host: ipa.company.local:389', fill='black')
draw.text((20, 100), 'LDAP authentication failed', fill='red')
img.save('/workspace/data/error_screenshot.png')
"""], capture_output=True)

result = analyze_image(
    "/workspace/data/error_screenshot.png",
    "Read all text in this image exactly as written."
)
print(result)
```

```
Error: Connection refused
Host: ipa.company.local:389
LDAP authentication failed
```

---

## ЧАСТЬ 5: RAG — ПОИСКОВЫЕ СИСТЕМЫ (БЛОК 3)

### Теория (20 мин)

**📌 Термин: RAG** (Retrieval-Augmented Generation) — паттерн где LLM получает не только запрос пользователя, но и релевантные фрагменты из внешней базы знаний. Решает проблему "модель не знает наши документы" которую мы увидели на МК1.

**📌 Термин: Embedding** — преобразование текста в вектор чисел. Похожие по смыслу тексты → близкие векторы. Математическое свойство, выученное при обучении embedding-модели.

**📌 Термин: Чанк** — фрагмент документа фиксированного размера (~400-800 токенов). Документ разбивается на чанки перед индексацией чтобы каждый чанк нёс конкретный смысл.

**📌 Термин: BM25** — алгоритм полнотекстового поиска по частоте слов. Находит точные совпадения: артикулы, имена пакетов, ID. Не понимает смысл.

**📌 Термин: Гибридный поиск** — комбинация векторного (семантика) и BM25 (точные совпадения) через алгоритм RRF (Reciprocal Rank Fusion).

**📌 Термин: nomic-embed-text** — лёгкая embedding-модель (274 МБ) через Ollama. Быстрый старт.

**📌 Термин: BGE-M3** (BAAI General Embedding Multilingual 3) — мультиязычная embedding-модель от Beijing Academy of AI. В одной модели: dense + sparse + ColBERT векторы. Превосходит nomic-embed-text на русскоязычных корпусах.

**📌 Термин: Qdrant local mode** — режим где Qdrant работает как Python-библиотека без отдельного сервера. `QdrantClient(path="/workspace/qdrant_data")` — данные сохраняются на диск, выживают при рестарте ячейки. Ограничение: до ~20 000 векторов.

**Связка с МК1:** T-lite-dpo не знал пакет `astra-freeipa-server` — это фактическое знание, которое fine-tuning не добавляет. RAG добавляет. После этого блока корпоративный ассистент знает документацию.

### notebook_03_rag.ipynb

```python
# ── ЯЧЕЙКА 0 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 БЕЗ RAG: задаём вопрос qwen3 напрямую
# ⚠️ Смотрим как модель галлюцинирует или признаётся в незнании
# ⚠️ Сохраняем ответ — сравним с RAG-ответом в конце ноутбука

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    reasoning=False,
)

DEMO_QUESTION = "Как в Astral Linux 1.8 установить FreeIPA Server?"

print("=" * 60)
print(f"ВОПРОС: {DEMO_QUESTION}")
print("=" * 60)
print("ОТВЕТ БЕЗ RAG (голая модель):\n")

# Стримим токены — не сидим смотрим на пустой экран
response_without_rag = ""
for chunk in llm.stream(DEMO_QUESTION):
    token = chunk.content
    print(token, end="", flush=True)
    response_without_rag += token  # копим полный ответ для сравнения в конце

print("\n" + "=" * 60)
print("ВЫВОД: модель либо галлюцинирует, либо говорит что не знает.")
print("Сейчас загрузим документацию и покажем разницу.")
print("=" * 60)
```

Видим ПОЛНЫЙ бред:

```text
Установка **FreeIPA Server** в **Astral Linux 1.8** (основан на **CentOS 8** или **RHEL 8**) может быть выполнена следующим образом. Ниже приведены шаги для установки FreeIPA Server с использованием официальных инструментов и пакетов.

Перед установкой рекомендуется обновить систему:
sudo dnf update -y

FreeIPA Server можно установить через официальный репозиторий FreeIPA. Для этого нужно добавить репозиторий и установить пакет.

### 2.1. Добавление репозитория FreeIPA
sudo dnf install -y https://downloads.pagure.org/freeipa/freeipa-server/1.8.0/RPMS/x86_64/freeipa-server-1.8.0-1.el8.x86_64.rpm

```

---

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Загрузка документов

from pathlib import Path
from langchain_core.documents import Document

docs_dir = Path("/workspace/data/docs/")
raw_docs = []

for txt_file in sorted(docs_dir.glob("**/*.txt")):
    text = txt_file.read_text(encoding="utf-8")
    raw_docs.append(Document(
        page_content=text,
        metadata={"source": str(txt_file), "filename": txt_file.name}
    ))

print(f"Загружено документов: {len(raw_docs)}")
for doc in raw_docs:
    print(f"  - {doc.metadata['filename']} ({len(doc.page_content)} символов)")
print(f"\nПример (первые 200 символов):\n{raw_docs[0].page_content[:200]}")
```

---

Мы берем длинный сырой документ и аккуратно нарезаем его на небольшие фрагменты (чанки) максимум по 600 символов. Мы специально делаем нахлест (`chunk_overlap=60`) в 60 символов между соседними кусками, чтобы не разорвать смысл на полуслове:

1. **Подготовка данных для RAG (Chunking):** Обязательный инженерный шаг перед сохранением текста в векторную базу. Векторный поиск ищет точно только по небольшим абзацам, а не по целым книгам или мануалам.
2. **Защита контекста:** Суть параметра `chunk_overlap` — это страховка от того, что важный факт, IP-адрес или строчка лога не «разрежется» пополам между двумя разными чанками, став невидимой для поиска.

```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~1 мин | 📌 Чанкинг
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"Чанков: {len(chunks)}")
```

---

Запускаем локальную модель векторизации (`nomic-embed-text`) и подключаемся к легковесной векторной базе ChromaDB. Проверяем базу на пустоту (чтобы не дублировать данные), сохраняем в нее наши нарезанные чанки и сразу делаем тестовый поиск по смыслу:

1. **Полностью локальный RAG (Air-gapped):** Мы показываем, что для работы векторного конвейера не нужны внешние API-ключи от OpenAI или HuggingFace. Вся приватная корпоративная документация векторизуется и хранится строго внутри нашего контура.
2. **Персистентность и Идемпотентность (SRE паттерны):** База сохраняет данные на жесткий диск (`PersistentClient`), а сам скрипт проверяет наличие записей перед стартом. При падении или перезапуске контейнера мы не потеряем данные и не будем заново тратить время на "прогрев" индекса.

```python
# ── ЯЧЕЙКА 3а ─────────────────────────────────────────────────────────────
# ⏱ ~5 мин | 📌 ВАРИАНТ 1: ChromaDB — самый простой
# ⚠️ PersistentClient: данные сохраняются в /workspace, выживают при рестарте

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Embedding через Ollama — не нужен HuggingFace токен
embeddings_nomic = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# PersistentClient — данные в /workspace/chroma_data
chroma_client = chromadb.PersistentClient(path="/workspace/chroma_data")

vectorstore_chroma = Chroma(
    client=chroma_client,
    collection_name="enterprise_docs",
    embedding_function=embeddings_nomic,
)

# Добавляем документы если коллекция пуста
if vectorstore_chroma._collection.count() == 0:
    print("Индексируем документы в ChromaDB...")
    vectorstore_chroma.add_documents(chunks)
    print(f"Проиндексировано: {vectorstore_chroma._collection.count()} чанков")
else:
    print(f"ChromaDB уже содержит {vectorstore_chroma._collection.count()} чанков ✅")

# Тест
results = vectorstore_chroma.similarity_search("Как в Astral Linux 1.8 установить FreeIPA Server?", k=3)
print(f"\nРезультаты поиска (top-3):")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] {doc.page_content[:150]}...")
```

---

Инициализируем мощную мультиязычную модель `BGE-M3` (BAAI General Embedding M3) для векторизации текстов. Затем запускаем базу данных Qdrant в "локальном" режиме — она работает как обычная Python-библиотека, сохраняя данные на диск. Скрипт проверяет наличие коллекции: если её нет — создает с правильной размерностью (1024) и загружает чанки, если есть — просто подключается:

1. **Production-качество эмбеддингов:** Показываем переход на `BGE-M3`. Это энтерпрайз-стандарт, который, в отличие от базовых моделей, отлично понимает русский язык, технический сленг и куски кода.
2. **Легковесная архитектура (Serverless Vector DB):** Для старта или локальной разработки нам не обязательно поднимать отдельный Docker-контейнер с тяжелым сервером БД. Qdrant может работать прямо внутри Python-процесса, экономя ресурсы (RAM/CPU), сохраняя при этом все данные на жестком диске.
3. **Идемпотентность (SRE-подход):** Снова делаем акцент на безопасности выполнения кода — скрипт сам проверяет состояние базы и никогда не задвоит данные при случайном повторном запуске ячейки.

```python
# ── ЯЧЕЙКА 3б ─────────────────────────────────────────────────────────────
# ⏱ ~8 мин | 📌 ВАРИАНТ 2: Qdrant local mode — production подход
# ⚠️ Никакого сервера! Qdrant работает как Python-библиотека
# ⚠️ path= → данные в /workspace/qdrant_data, персистентны

# клиент для работы с Qdrant
from qdrant_client import QdrantClient       
from langchain_huggingface import HuggingFaceEmbeddings    
from langchain_qdrant import QdrantVectorStore

# ─── 1. ЗАГРУЗКА EMBEDDING-МОДЕЛИ ───────────────────────────────────────────

# BGE-M3 от BAAI — мультиязычная модель, хорошо работает с русским текстом.
# Альтернатива: nomic-embed-text (274 МБ, только английский).
# При первом запуске скачивает ~570 МБ в кэш HuggingFace (~/.cache/huggingface).
print("Загружаем BGE-M3 embedding модель...")

embeddings_bge = HuggingFaceEmbeddings(
	# имя модели на HuggingFace Hub
    model_name="BAAI/bge-m3",          
    # запускаем на CPU; замените на "cuda" если есть GPU
    model_kwargs={"device": "cpu"},     
    encode_kwargs={
	    # L2-нормализация вектора → косинусное сходство
	    # становится эквивалентно скалярному произведению
        "normalize_embeddings": True    
    }
)

# Быстрый smoke-test: убеждаемся что модель загружена и возвращает вектор нужной размерности
test_vec = embeddings_bge.embed_query("тест")
print(f"BGE-M3 OK ✅ — размерность: {len(test_vec)}")  # BGE-M3 всегда даёт 1024 измерения

# ─── 2. ПОДКЛЮЧЕНИЕ К QDRANT ────────────────────────────────────────────────

# path= означает LOCAL MODE — Qdrant работает как встроенная библиотека,
# без отдельного сервера и Docker. Данные персистируются на диск по указанному пути.
# В RunPod /workspace — это Network Volume: переживает рестарт пода.
# Альтернатива для production: QdrantClient(url="http://qdrant-server:6333")

# Переиспользуем существующий клиент если он есть в globals
# иначе создаём новый (с защитой от stale lock)
if "qdrant_client" not in globals():
    try:
        qdrant_client = QdrantClient(path="/workspace/qdrant_data")
    except RuntimeError:
        # Stale lock от предыдущей сессии — удаляем и повторяем
        import os
        lock_path = "/workspace/qdrant_data/.lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print(f"🔓 Удалён stale lock: {lock_path}")
        qdrant_client = QdrantClient(path="/workspace/qdrant_data")

# ─── 3. СОЗДАНИЕ КОЛЛЕКЦИИ (если не существует) ─────────────────────────────

from qdrant_client.models import Distance, VectorParams



# Получаем список всех существующих коллекций и проверяем наличие нашей.
# Это идемпотентная логика: скрипт безопасно перезапускать — 
# повторной индексации одних и тех же документов не произойдёт.
if "enterprise_docs_bge" not in [c.name for c in qdrant_client.get_collections().collections]:

    print("Создаём коллекцию Qdrant...")
    qdrant_client.create_collection(
        collection_name="enterprise_docs_bge",
        vectors_config=VectorParams(
	        # размерность вектора — должна совпадать с BGE-M3
            size=1024,              
            # метрика сходства; COSINE работает с нормализованными векторами
            # другие варианты: DOT (скалярное произведение), EUCLID
            distance=Distance.COSINE  
        )
    )

    # QdrantVectorStore — LangChain-обёртка поверх голого QdrantClient.
    # Именно через неё работают .add_documents(), .as_retriever(), similarity_search().
    vectorstore_qdrant = QdrantVectorStore(
        client=qdrant_client,
        collection_name="enterprise_docs_bge",
        embedding=embeddings_bge,   # модель, которая будет векторизовать текст при запросах
    )

    # Индексирование: каждый Document из chunks превращается в вектор через BGE-M3
    # и записывается в Qdrant вместе с метаданными (source, page и т.д.)
    # chunks — список Document-объектов из предыдущего шага (RecursiveCharacterTextSplitter)
    vectorstore_qdrant.add_documents(chunks)

    # Проверяем сколько векторов реально записалось в коллекцию
    print(f"Проиндексировано: {qdrant_client.count('enterprise_docs_bge').count} чанков ✅")

else:
    # Коллекция уже существует — просто подключаемся к ней.
    # Данные на диске сохранились с прошлого запуска — индексировать заново не нужно.
    vectorstore_qdrant = QdrantVectorStore(
        client=qdrant_client,
        collection_name="enterprise_docs_bge",
        embedding=embeddings_bge,
    )

    # Выводим текущее количество векторов для подтверждения что данные на месте
    count = qdrant_client.count("enterprise_docs_bge").count
    print(f"Qdrant уже содержит {count} чанков ✅")
```

```python
# ── ЯЧЕЙКА 4 ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 BM25 — полнотекстовый поиск

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
results = bm25_retriever.invoke("astra-freeipa-server")
print("BM25 нашёл (точное совпадение по имени пакета):")
for i, doc in enumerate(results[:2]):
    print(f"\n[{i+1}] {doc.page_content[:200]}")
```

Объединяем два независимых механизма поиска: смысловой (векторная база Qdrant) и лексический по ключевым словам (классический алгоритм BM25). Скрипт пытается использовать встроенный `EnsembleRetriever`, но если нужный пакет недоступен, плавно переключается на нашу собственную математическую реализацию слияния списков — алгоритм RRF (Reciprocal Rank Fusion):

1. **Превосходство гибридного поиска:** Показываем, как закрыть главную уязвимость векторного RAG. Нейросеть отлично ищет по "смыслу", но часто теряет специфические термины, артикулы или точные имена пакетов (например, `astra-freeipa-server`). Добавление BM25 гарантирует, что точные совпадения не потеряются.
2. **Паттерн отказоустойчивости (Graceful Degradation):** Мы наглядно демонстрируем инженерный SRE-подход к написанию кода. Если зависимость LangChain изменилась или сломалась, наш конвейер не падает с красным трейсбеком, а бесшовно переходит на резервную ручную логику ранжирования.

```python
# ── ЯЧЕЙКА 5 ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Гибридный поиск — лучший из двух миров (Векторный поиск + Ключевые слова)

# Импортируем BM25 — классический алгоритм полнотекстового поиска (TF-IDF на стероидах).
# Он отлично находит точные совпадения (версии ПО, логи, команды в терминале), 
# с которыми часто не справляется семантический векторный поиск.
from langchain_community.retrievers import BM25Retriever

# Пытаемся импортировать готовый "объединитель" ретриверов из LangChain.
try:
    # EnsembleRetriever берет результаты разных поисков и сливает их воедино.
    from langchain_classic.retrievers import EnsembleRetriever
    
    # Сообщаем об успехе импорта.
    print("EnsembleRetriever: langchain_classic OK")
    
    # Ставим флаг, чтобы использовать "коробочное" решение.
    USE_ENSEMBLE = True
except ImportError:
    # Защита от проблем с зависимостями: если пакета нет, мы не падаем с ошибкой.
    print("langchain_classic не установлен — используем ручной RRF")
    
    # Ставим флаг для активации нашего собственного алгоритма слияния.
    USE_ENSEMBLE = False

# Создаем лексический индекс (BM25) прямо в оперативной памяти из наших чанков.
bm25_retriever = BM25Retriever.from_documents(chunks)

# Настраиваем BM25 так, чтобы он возвращал топ-5 совпадений по ключевым словам.
bm25_retriever.k = 5

# Создаем векторный ретривер из уже существующей базы данных Qdrant.
vector_retriever = vectorstore_qdrant.as_retriever(
    # Используем косинусное расстояние (similarity) для поиска по смыслу.
    search_type="similarity",
    
    # Также просим вернуть топ-5 документов, но уже по семантической близости.
    search_kwargs={"k": 5}
)

# Задаем тестовый запрос пользователя, в котором есть и смысл, и точные термины.
QUERY = "Как установить astra-freeipa-server на Astra Linux 1.8?"

# Если встроенный инструмент LangChain доступен:
if USE_ENSEMBLE:
    # Создаем гибридный ретривер, передавая ему оба наших поисковика.
    hybrid_retriever = EnsembleRetriever(
        # Список ретриверов, которые будут опрашиваться параллельно.
        retrievers=[vector_retriever, bm25_retriever],
        
        # Задаем веса: 60% значимости векторному поиску (смысл), 40% — лексическому (точные слова).
        weights=[0.6, 0.4]
    )
    
    # Запускаем конвейер поиска одним вызовом invoke.
    hybrid_results = hybrid_retriever.invoke(QUERY)

# Если пакета нет, реализуем алгоритм RRF (Reciprocal Rank Fusion) своими руками:
else:
    # Функция ручного слияния результатов RRF.
    # Константа k=60 — золотой стандарт из академических статей для сглаживания весов.
    def reciprocal_rank_fusion(results_list, k=60):
        
        # Словарь для накопления итоговых баллов каждого документа.
        scores = {}
        
        # Перебираем списки результатов от каждого ретривера (векторного и BM25).
        for docs in results_list:
            
            # Перебираем документы в списке. rank — это позиция документа в выдаче (0, 1, 2...).
            for rank, doc in enumerate(docs):
                
                # Создаем уникальный ключ документа по первым 80 символам.
                # Это нужно, чтобы понять, что оба ретривера нашли один и тот же чанк.
                key = doc.page_content[:80]
                
                # Если мы видим этот документ впервые, добавляем его в словарь с нулевым счетом.
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                
                # Математика RRF: прибавляем балл, обратно пропорциональный позиции в выдаче.
                # Чем ближе к первому месту (rank=0), тем больше баллов получит документ.
                scores[key]["score"] += 1.0 / (rank + k)
        
        # Возвращаем список документов.
        return [
            # Достаем сам объект документа.
            v["doc"] for v in sorted(
                # Сортируем словарь по итоговому значению "score" по убыванию (от большего к меньшему).
                scores.values(), key=lambda x: x["score"], reverse=True
            )
        ]

    # Шаг 1: Получаем топ-5 результатов только по смыслу (вектор).
    vector_results = vector_retriever.invoke(QUERY)
    
    # Шаг 2: Получаем топ-5 результатов только по точным совпадениям (BM25).
    bm25_results   = bm25_retriever.invoke(QUERY)
    
    # Шаг 3: Сливаем обе выдачи через наш RRF-алгоритм и получаем итоговый ранжированный топ.
    hybrid_results = reciprocal_rank_fusion([vector_results, bm25_results])

# Выводим общее количество уникальных документов после слияния.
print(f"Гибридный поиск: {len(hybrid_results)} результатов")

# Проходимся по первым двум лучшим документам из итоговой гибридной выдачи.
for i, doc in enumerate(hybrid_results[:2]):
    
    # Печатаем первые 200 символов документа, чтобы визуально оценить релевантность.
    print(f"\n[{i+1}] {doc.page_content[:200]}")
```

```python
# ── ЯЧЕЙКА 6 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Полная RAG-цепочка
# ⚠️ КЛЮЧЕВОЙ МОМЕНТ МК: тот же вопрос что провалился на МК1!

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Источник {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)

rag_prompt = ChatPromptTemplate.from_template("""
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если ответа нет в контексте — скажи "Информация отсутствует".
Всегда указывай источник.

Контекст:
{context}

Вопрос: {question}
""")

rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ДЕМОНСТРАЦИЯ: вопрос который провалился на МК1
question = "Покажи только команды на сервере Astra Linux 1.8 чтобы установить FreeIPA server?"
print(f"Вопрос: {question}\n")
print("RAG-ответ (streaming):")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
```

Ожидаемый вывод (сравнение с МК1):
```
МК1 без RAG:  sudo apt-get install freeipa-server  ← НЕВЕРНО
МК2 с RAG:   sudo apt install astra-freeipa-server  ← ВЕРНО
[Источник 3: /workspace/data/docs/Astra_Linux_1.8_FreeIPA_Server_Install.txt]
```

---

**Оценка RAG-системы: декомпозиция метрик и LLM-as-a-judge**

Оценивать RAG-систему как монолитный "чёрный ящик" ошибочно — для точной локализации сбоев её необходимо разделять на независимые подсистемы поиска контекста (Retrieval) и синтеза ответа (Generation).

Для диагностики этих этапов применяются три стандарта: **Faithfulness** (отсутствие галлюцинаций генератора), **Context Recall** (полнота найденной ретривером информации) и **Context Precision** (качество ранжирования документов).

Поскольку ручной подсчет результатов на сотнях тестов неэффективен, индустрия использует фреймворки RAGAS и DeepEval для автоматизации процесса.

Они работают в парадигме **LLM-as-a-judge**, где сильная языковая модель выступает в роли независимого судьи, который сверяет факты и выдает детерминированные числовые оценки качества.


```python
# ── ЯЧЕЙКА 7 ──────────────────────────────────────────────────────────────
# ⏱ ~5-7 мин | 📌 Оценка RAG: LLM-as-a-judge без новых пакетов
# Реализуем три метрики из раздела 3.8 через Qwen-3, который уже в стеке.
# hybrid_retriever, llm, chunks — всё берём из предыдущих ячеек.

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── 1. ТЕСТОВЫЙ ДАТАСЕТ ────────────────────────────────────────────────────

# Ground Truth — эталонные ответы, написанные человеком заранее.
# В реальном проекте: резолюции тикетов L2/L3, официальная документация,
# ответы, подтверждённые экспертом.
# Замените на вопросы из ВАШЕЙ предметной области.
test_dataset = [
    {
        "question": "Как установить FreeIPA сервер на Astra Linux 1.8?",
        "ground_truth": "Для установки FreeIPA сервера на Astra Linux необходимо выполнить команду: sudo apt install astra-freeipa-server"
    },
    {
        "question": "Как проверить статус службы в Astra Linux 1.8?",
        "ground_truth": "Для проверки статуса службы используется команда: sudo systemctl status <имя_службы>"
    },
    {
        "question": "Какой порт использует LDAP по умолчанию?",
        "ground_truth": "LDAP по умолчанию использует порт 389 для незащищённых соединений и порт 636 для LDAPS"
    },
]

# ─── 2. ДВЕ РОЛИ — ДВА РЕЖИМА QWEN-3 ───────────────────────────────────────

# Генератор ответов: reasoning=False — нам не нужны рассуждения в RAG-ответе,
# только быстрый синтез из контекста
llm_answerer = ChatOllama(model="qwen3:8b", reasoning=False)

# Судья: reasoning=True — судья должен думать тщательно перед выставлением оценки.
# thinking-блок уйдёт в additional_kwargs и не сломает JSON-парсинг
llm_judge_base = ChatOllama(model="qwen3:8b", reasoning=True)

# ─── 3. PYDANTIC-СХЕМА ДЛЯ СУДЬИ ────────────────────────────────────────────

class JudgeScore(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0,
        description="Оценка качества от 0.0 до 1.0"
    )
    reasoning: str = Field(
        description="Краткое обоснование оценки в 1-2 предложениях"
    )

# with_structured_output гарантирует валидный JSON от судьи.
# reasoning=True в llm_judge_base изолирует <think>-теги от JSON-парсинга
judge = llm_judge_base.with_structured_output(JudgeScore)

# ─── 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────────────────────

def format_context(docs: list, max_chars: int = 3000) -> str:
    """Превращает список Document-объектов в единую строку для промпта.
    Обрезаем по max_chars чтобы не выйти за контекстное окно Qwen-3 (32k)."""
    return "\n\n---\n\n".join(
        f"[Чанк {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )[:max_chars]


def generate_rag_answer(question: str, context: str) -> str:
    """Шаг Generation в RAG-пайплайне: LLM синтезирует ответ из контекста."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты ассистент технической поддержки. "
         "Отвечай СТРОГО на основе предоставленного контекста. "
         "Если ответа в контексте нет — честно скажи об этом.\n\n"
         "Контекст:\n{context}"),
        ("human", "{question}")
    ])
    return (prompt | llm_answerer | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )


def safe_judge(chain, inputs: dict) -> JudgeScore:
    """Обёртка с защитой от сбоя судьи.
    Если structured output упал — возвращаем нейтральный Score 0.5."""
    try:
        return chain.invoke(inputs)
    except Exception as e:
        print(f"    ⚠️  Судья вернул ошибку: {e} → Score 0.5")
        return JudgeScore(score=0.5, reasoning="Ошибка оценки")

# ─── 5. ТРИ МЕТРИКИ — ТРИ ПРОМПТА ───────────────────────────────────────────

# МЕТРИКА 1: FAITHFULNESS
# Вопрос: «Основан ли ответ строго на фактах из контекста?»
# Оценивает генератор. Нужны: context + answer. Ground Truth НЕ нужен.
faith_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты строгий аудитор AI-ответов. "
     "Оцени Faithfulness — насколько ОТВЕТ основан на КОНТЕКСТЕ. "
     "Score 1.0: каждый факт в ответе подтверждён контекстом. "
     "Score 0.0: ответ содержит факты, которых нет в контексте (галлюцинации). "
     "Галлюцинация = любое утверждение без опоры на предоставленный текст."),
    ("human",
     "КОНТЕКСТ:\n{context}\n\n"
     "ОТВЕТ:\n{answer}\n\n"
     "Оцени Faithfulness от 0.0 до 1.0.")
])
faith_chain = faith_prompt | judge


# МЕТРИКА 2: CONTEXT RECALL
# Вопрос: «Нашёл ли ретривер всю информацию, нужную для идеального ответа?»
# Оценивает ретривер. Нужны: context + ground_truth. Ответ LLM НЕ нужен.
recall_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты строгий аудитор качества поиска. "
     "Оцени Context Recall — насколько КОНТЕКСТ покрывает факты из ЭТАЛОННОГО ОТВЕТА. "
     "Score 1.0: все ключевые факты эталона присутствуют в контексте. "
     "Score 0.0: контекст не содержит критически важных фактов эталона. "
     "Низкий Recall = ретривер промахнулся и не достал нужные документы."),
    ("human",
     "ЭТАЛОННЫЙ ОТВЕТ:\n{ground_truth}\n\n"
     "НАЙДЕННЫЙ КОНТЕКСТ:\n{context}\n\n"
     "Оцени Context Recall от 0.0 до 1.0.")
])
recall_chain = recall_prompt | judge


# МЕТРИКА 3: CONTEXT PRECISION
# Вопрос: «Полезные чанки стоят на верхних позициях выдачи?»
# Оценивает ранжирование ретривера. Важен порядок, не только наличие.
precision_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты строгий аудитор ранжирования поиска. "
     "Оцени Context Precision — насколько полезные чанки стоят СВЕРХУ списка. "
     "Score 1.0: самые релевантные чанки на позициях 1-2. "
     "Score 0.5: полезный чанк есть, но глубоко в списке. "
     "Score 0.0: полезных чанков нет или они на последних позициях. "
     "Важно: LLM теряет внимание к середине длинного контекста (Lost-in-the-Middle). "
     "Правильный чанк на 8-й позиции хуже, чем на 1-й."),
    ("human",
     "ВОПРОС:\n{question}\n\n"
     "ЭТАЛОННЫЙ ОТВЕТ:\n{ground_truth}\n\n"
     "РАНЖИРОВАННЫЕ ЧАНКИ:\n{chunks}\n\n"
     "Оцени Context Precision от 0.0 до 1.0.")
])
precision_chain = precision_prompt | judge

# ─── 6. ГЛАВНЫЙ ЦИКЛ ОЦЕНКИ ─────────────────────────────────────────────────

results = []

print("=" * 65)
print("🔍  Запускаем оценку RAG-системы (LLM-as-a-judge / Qwen-3)")
print("=" * 65)

for i, case in enumerate(test_dataset):
    question    = case["question"]
    ground_truth = case["ground_truth"]

    print(f"\n[{i+1}/{len(test_dataset)}] ❓ {question}")

    # ШАГ 1: Ретривал — hybrid_retriever из Ячейки 5
    retrieved_docs = hybrid_retriever.invoke(question)
    context = format_context(retrieved_docs)

    # ШАГ 2: Генерация ответа через RAG
    print("  → Генерируем ответ...", end=" ", flush=True)
    answer = generate_rag_answer(question, context)
    print("✓")

    # ШАГ 3а: Faithfulness — оцениваем генератор
    print("  → Faithfulness...",      end=" ", flush=True)
    faith = safe_judge(faith_chain, {"context": context, "answer": answer})
    print(f"{faith.score:.2f}")

    # ШАГ 3б: Context Recall — оцениваем полноту ретривала
    print("  → Context Recall...",    end=" ", flush=True)
    recall = safe_judge(recall_chain, {"ground_truth": ground_truth, "context": context})
    print(f"{recall.score:.2f}")

    # ШАГ 3в: Context Precision — оцениваем качество ранжирования
    # Формируем пронумерованный список чанков — судье важны позиции
    ranked_chunks = "\n\n".join(
        f"[Позиция {j+1}]:\n{doc.page_content[:300]}"
        for j, doc in enumerate(retrieved_docs)
    )
    print("  → Context Precision...", end=" ", flush=True)
    precision = safe_judge(precision_chain, {
        "question":    question,
        "ground_truth": ground_truth,
        "chunks":       ranked_chunks
    })
    print(f"{precision.score:.2f}")

    results.append({
        "question":          question,
        "answer":            answer,
        "faithfulness":      faith.score,
        "context_recall":    recall.score,
        "context_precision": precision.score,
        "faith_why":         faith.reasoning,
        "recall_why":        recall.reasoning,
        "precision_why":     precision.reasoning,
    })

# ─── 7. ИТОГОВАЯ ТАБЛИЦА ────────────────────────────────────────────────────

avg_f = sum(r["faithfulness"]      for r in results) / len(results)
avg_r = sum(r["context_recall"]    for r in results) / len(results)
avg_p = sum(r["context_precision"] for r in results) / len(results)

def verdict(score: float) -> str:
    if score >= 0.8: return "✅ Хорошо"
    if score >= 0.6: return "⚠️  Приемлемо"
    return "❌ Требует доработки"

print("\n" + "=" * 65)
print("📊  ИТОГОВЫЕ МЕТРИКИ RAG-СИСТЕМЫ")
print("=" * 65)
print(f"{'Метрика':<25} {'Avg Score':>10}   Вердикт")
print("-" * 65)
print(f"{'Faithfulness':<25} {avg_f:>10.2f}   {verdict(avg_f)}")
print(f"  └ Низкий → галлюцинации генератора; проверь промпт и reasoning=False")
print(f"{'Context Recall':<25} {avg_r:>10.2f}   {verdict(avg_r)}")
print(f"  └ Низкий → ретривер промахивается; попробуй гибридный поиск / BGE-M3")
print(f"{'Context Precision':<25} {avg_p:>10.2f}   {verdict(avg_p)}")
print(f"  └ Низкий → нужен Reranker (Cross-Encoder) поверх ретривера")

print("\n📋  Детальный разбор по тест-кейсам:")
print("-" * 65)
for r in results:
    print(f"\n❓  {r['question']}")
    print(f"💬  Ответ: {r['answer'][:120]}...")
    print(f"    Faithfulness  {r['faithfulness']:.2f}  — {r['faith_why'][:90]}")
    print(f"    Recall        {r['context_recall']:.2f}  — {r['recall_why'][:90]}")
    print(f"    Precision     {r['context_precision']:.2f}  — {r['precision_why'][:90]}")
```

```text
=================================================================
🔍  Запускаем оценку RAG-системы (LLM-as-a-judge / Qwen-3)
=================================================================

[1/3] ❓ Как установить FreeIPA сервер на Astra Linux?
  → Генерируем ответ... ✓
  → Faithfulness... 1.00
  → Context Recall... 1.00
  → Context Precision... 0.50
  ...
```

---

## ЧАСТЬ 6: ГРАФЫ ЗНАНИЙ (БЛОК 4)

### Теория (15 мин)

**📌 Термин: Граф знаний** — структура данных где сущности (узлы) соединены отношениями (рёбра). Позволяет задавать структурные вопросы: "кто связан с X".

**📌 Термин: Тройка** — атомарный факт: `(Субъект, Предикат, Объект)`. `(Ромашка, закупила, Microsoft Office)`.

**📌 Термин: LLMGraphTransformer** — инструмент LangChain для автоматического извлечения троек из текста с помощью LLM. Без ручного NLP-парсера.

**📌 Термин: NetworkX** — Python-библиотека для работы с графами. Для демо — идеально (zero-config). Для production (>10к документов): Microsoft GraphRAG, LightRAG, Neo4j.

**Почему граф дополняет RAG:**

```
RAG хорошо: "Что сказано в договоре N45 о возврате?"
RAG плохо:  "Какие клиенты закупают через компанию X?"

Второй вопрос требует объединения сотен документов.
Граф знаний хранит связи явно — мгновенный ответ.
```

### notebook_04_knowledge_graph.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~5 мин | 📌 Извлечение троек через LLMGraphTransformer с прогресс-баром

# Импортируем экспериментальный модуль для автоматического извлечения графов.
# Он использует LLM для задачи NER (распознавание именованных сущностей) и извлечения связей.
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Импортируем базовый класс Document, стандартный формат хранения текстов в LangChain.
from langchain_core.documents import Document

# Импортируем класс для работы с локальными моделями через Ollama.
from langchain_ollama import ChatOllama

# Импортируем библиотеку для красивых прогресс-баров (специальная версия для Jupyter Notebook).
from tqdm.notebook import tqdm

# Инициализируем локальную LLM для извлечения графа.
# reasoning=False отключает генерацию рассуждений (тегов <think>), чтобы они не ломали парсинг JSON.
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    reasoning=False,
)

# Создаем расширенный корпус из 10 неструктурированных текстов.
# Эти данные образуют единую связную корпоративную историю: от закупок ПО до расследования инцидентов.
corpus = [
    Document(page_content="Менеджер Иванов Сергей работает в компании Ромашка. Компания Техносерв сотрудничает с корпорацией Microsoft. Техносерв поставляет программное обеспечение Microsoft Office. Компания Ромашка использует Microsoft Office."),
    
    Document(page_content="Компания ИнфоТех внедряет программное обеспечение Astra Linux. Компания ИнфоТех сотрудничает с организацией ФСТЭК. Иванов Сергей использует систему Astra Linux в своей работе."),
    
    Document(page_content="Инженер Петров Алексей работает в компании ИнфоТех. Компания ИнфоТех внедряет программное обеспечение Kubernetes. Система Kubernetes использует программное обеспечение HashiCorp Vault."),
    
    Document(page_content="Произошел Сетевой Инцидент. Данный Сетевой Инцидент влияет на программное обеспечение PostgreSQL. Петров Алексей использует PostgreSQL."),
    
    Document(page_content="Организация DataCenter-X поставляет оборудование Серверы Yadro. Компания Yadro поставляет оборудование Серверы Yadro. Компания Ромашка использует Серверы Yadro."),
    
    Document(page_content="Администратор Смирнова Анна работает в компании Ромашка. Компания Ромашка использует программное обеспечение FreeIPA. Система FreeIPA использует программное обеспечение Astra Linux."),
    
    Document(page_content="Организация SecurityAudit сотрудничает с компанией Ромашка. Аудиторы из SecurityAudit используют программное обеспечение HashiCorp Vault для проверок."),
    
    Document(page_content="Смирнова Анна внедряет программное обеспечение GitLab. Инженер Петров Алексей использует систему GitLab для настройки безопасности. GitLab использует программное обеспечение Kubernetes."),
    
    Document(page_content="Произошел Аппаратный Сбой. Этот Аппаратный Сбой влияет на оборудование Серверы Yadro. Смирнова Анна использует Серверы Yadro."),
    
    Document(page_content="Компания Техносерв сотрудничает с организацией ИнфоТех. Совместно они создали компанию ТехноИнфо. Компания Ромашка сотрудничает с организацией ТехноИнфо."),
    
    # Иванов Сергей и его проекты в Ромашке
    Document(page_content="Менеджер Иванов Сергей работает в компании Ромашка. Иванов Сергей курирует проект Внедрение Astra Linux. Иванов Сергей курирует проект Модернизация Сети."),
    
    # Детализация Проекта 1 (Astra Linux)
    Document(page_content="Проект Внедрение Astra Linux использует программное обеспечение Astra Linux. Компания ИнфоТех внедряет проект Внедрение Astra Linux."),
    
    # Детализация Проекта 2 (Модернизация Сети)
    Document(page_content="Проект Модернизация Сети использует оборудование Серверы Yadro. Компания Техносерв поставляет оборудование Серверы Yadro для проекта Модернизация Сети."),
    
    # Связи людей и подрядчиков с Проектом 1
    Document(page_content="Инженер Петров Алексей работает в компании ИнфоТех. Петров Алексей поддерживает проект Внедрение Astra Linux. Проект Внедрение Astra Linux использует программное обеспечение FreeIPA."),
    
    # Связи людей и подрядчиков с Проектом 2
    Document(page_content="Администратор Смирнова Анна работает в компании Техносерв. Смирнова Анна поддерживает проект Модернизация Сети. Смирнова Анна использует оборудование Серверы Yadro."),
    
    # Партнерские замыкания графа
    Document(page_content="Компания Ромашка сотрудничает с организацией ИнфоТех. Компания Ромашка сотрудничает с организацией Техносерв. ИнфоТех сотрудничает с Техносерв.")
]

# Инициализируем трансформер, передавая ему нашу языковую модель (Qwen-3).
# По умолчанию он работает в режиме безсхемного извлечения (Schemaless), 
# то есть LLM сама придумывает типы для узлов (Person, Organization, Tool) и связей на лету.
transformer = LLMGraphTransformer(llm=llm)

# Создаем пустой список, куда будем складывать готовые графовые документы по мере их обработки.
graph_docs = []

# Информируем пользователя о начале долгого процесса.
print("Начинаем извлечение графа знаний...")

# Оборачиваем наш corpus в tqdm для отображения визуального прогресс-бара.
# desc — это текст, который будет виден рядом с ползунком прогресса.
for doc in tqdm(corpus, desc="Обработка документов"):
    
    # Передаем в трансформер строго ОДИН документ в виде списка.
    # LLM выполнит запрос только для него, а после завершения ползунок сдвинется на 1 шаг.
    result = transformer.convert_to_graph_documents([doc])
    
    # Добавляем полученный результат в наш общий список (extend, так как result — это список).
    graph_docs.extend(result)

# Проходимся циклом по каждому обработанному документу для вывода результатов в консоль.
for idx, gd in enumerate(graph_docs, 1):
    
    # Печатаем номер документа для визуального разделения.
    print(f"\n--- Документ {idx} ---")
    
    # gd.nodes содержит список найденных сущностей. Извлекаем только их имена (id).
    print(f"Узлы: {[n.id for n in gd.nodes]}")
    
    # gd.relationships содержит извлеченные тройки в формате (Субъект -> Предикат -> Объект).
    print(f"Связи: {[(r.source.id, r.type, r.target.id) for r in gd.relationships]}")

```

Ожидаемый вывод:
```
Извлекаем граф знаний...
Узлы: ['Компания Ромашка', 'Техносерв', 'Microsoft Office', 'Иванов Сергей', 'Microsoft']

Связи: [('Компания Ромашка', 'CONTRACT', 'Техносерв'), ('Техносерв', 'AUTHORIZED_PARTNER', 'Microsoft'), ('Компания Ромашка', 'RESPONSIBLE_MANAGER', 'Иванов Сергей'), ('Техносерв', 'SUPPLY', 'Microsoft Office')]

Узлы: ['Иванов Сергей', 'Проект Внедрения Astra Linux В Ромашке', 'Astra Linux', 'Ромашка', 'Инфотех', 'Фстэк']

Связи: [('Иванов Сергей', 'CURATES', 'Проект Внедрения Astra Linux В Ромашке'), ('Проект Внедрения Astra Linux В Ромашке', 'TECHNICAL_EXECUTOR', 'Инфотех'), ('Инфотех', 'CERTIFICATION_AUTHORITIES', 'Фстэк'), ('Инфотех', 'WORKS_WITH', 'Astra Linux'), ('Инфотех', 'LOCATED_IN', 'Ромашка')]
```


```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Строим и визуализируем граф в NetworkX

# Импортируем библиотеку networkx для создания, манипуляции и изучения структуры графов
import networkx as nx

# Импортируем модуль pyplot из библиотеки matplotlib для визуализации (отрисовки) графа
import matplotlib.pyplot as plt

# Создаем пустой направленный граф (Directed Graph), где связи имеют направление (от узла А к узлу Б)
G = nx.DiGraph()

# Проходим циклом по всем документам-графам (graph_docs), которые, вероятно, были извлечены парсером
for gd in graph_docs:
    
    # Перебираем все узлы (сущности) внутри текущего документа
    for node in gd.nodes:
        # Добавляем узел в наш общий граф G, используя его уникальный идентификатор (id)
        G.add_node(node.id)
        
    # Перебираем все связи (отношения) между узлами в текущем документе
    for rel in gd.relationships:
        # Добавляем ребро (связь) между исходным (source) и целевым (target) узлами,
        # одновременно сохраняя тип отношения в качестве атрибута 'label' для этого ребра
        G.add_edge(rel.source.id, rel.target.id, label=rel.type)

# Выводим в консоль сводную статистику: сколько всего уникальных узлов и связей мы собрали
print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")

# Создаем новую фигуру (окно для графика) и задаем её размер (14 дюймов в ширину, 8 в высоту)
plt.figure(figsize=(14, 8))

# Вычисляем координаты для каждого узла с помощью силового алгоритма (spring_layout).
# k=2.5 задает оптимальное расстояние между узлами (чтобы не слипались), seed=42 фиксирует расстановку для повторяемости
pos = nx.spring_layout(G, k=2.5, seed=42)

# Рисуем сами узлы графа в вычисленных позициях (pos), задаем их размер, синий цвет и небольшую прозрачность
nx.draw_networkx_nodes(G, pos, node_size=4500, node_color="#4A90D9", alpha=0.9)

# Отрисовываем текст внутри узлов (имена сущностей), настраиваем размер шрифта, белый цвет и жирность
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")

# Рисуем рёбра (линии со стрелками) между узлами, задаем размер стрелок, серый цвет и толщину линии
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color="#888", width=2)

# Извлекаем из графа словарь с метками рёбер (те самые типы отношений, которые мы положили в 'label')
edge_labels = nx.get_edge_attributes(G, 'label')

# Рисуем текстовые подписи поверх рёбер графа, чтобы было видно, как именно связаны объекты
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)

# Задаем заголовок для нашего графика
plt.title("Граф знаний корпоративных отношений")

# Отключаем отображение координатных осей (рамки и цифр), так как графу они не нужны
plt.axis('off')

# Автоматически подгоняем элементы графика, чтобы они максимально плотно уместились в области отрисовки и не обрезались
plt.tight_layout()

# Сохраняем получившийся рисунок в файл PNG с хорошим разрешением (dpi=150) и обрезкой пустых полей (bbox_inches)
plt.savefig("/workspace/knowledge_graph.png", dpi=150, bbox_inches='tight')

# Выводим сгенерированную картинку на экран (актуально для Jupyter Notebook или интерактивных сред)
plt.show()

# Печатаем сообщение в консоль, чтобы подтвердить, что файл успешно сохранен по указанному пути
print("Граф сохранён: /workspace/knowledge_graph.png")
```

---

Использование параметров `allowed_nodes` и `allowed_relationships` в `LLMGraphTransformer` переводит извлечение графа из открытой генерации в строгую классификацию по заранее заданной схеме. Это обеспечивает предсказуемость результатов, защищает граф от засорения лишними типами данных и позволяет заранее писать Cypher-запросы.

```python
# ── ЯЧЕЙКА 3a ────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Сфокусированное Schema-guided извлечение под финальный вопрос

# 1. Добавляем "Project" в разрешенные типы узлов!
allowed_nodes = [
    "Person",       # Люди (Иванов Сергей, Петров Алексей и др.)
    "Organization", # Компании (Ромашка, ИнфоТех, Техносерв)
    "Project",      # Проекты (Внедрение Astra Linux, Модернизация Сети)
    "Software",     # ПО (Astra Linux, PostgreSQL, FreeIPA)
    "Hardware"      # Оборудование (Серверы Yadro)
]

# 2. Добавляем явные связи управления и поддержки
allowed_relationships = [
    "MANAGES",      # [Person -> Project] Кто курирует/управляет проектом
    "WORKS_AT",     # [Person -> Organization] Кто где работает
    "IMPLEMENTS",   # [Organization/Person -> Project/Software] Кто внедряет
    "USES",         # [Project/Person -> Software/Hardware] Что используется в проекте
    "PARTNERS_WITH",# [Organization -> Organization] B2B партнерство
    "MAINTAINS",    # [Person/Organization -> Project/Software] Кто поддерживает
    "RUNS_ON"       # [Software -> Hardware/Software] На чем работает ПО
]

# Инициализируем трансформер со строгой сфокусированной схемой
clean_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships
)

# Извлекаем граф
clean_graph_docs = []
print("Извлекаем плотный граф связей Иванова Сергея...")

for doc in tqdm(corpus, desc="Обработка"):
    result = clean_transformer.convert_to_graph_documents([doc])
    clean_graph_docs.extend(result)

# Выводим найденные связи
for idx, gd in enumerate(clean_graph_docs, 1):
    if gd.relationships:
        print(f"\n--- Документ {idx} ---")
        for r in gd.relationships:
            print(f"  - ({r.source.id}) --[{r.type}]--> ({r.target.id})")
```




```python
# ── ЯЧЕЙКА 3b ────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Строим и визуализируем "чистый" граф (Production версия)

import networkx as nx
import matplotlib.pyplot as plt

# Создаем НОВЫЙ пустой направленный граф
G_clean = nx.DiGraph()

# Проходим циклом по всем документам
for gd in clean_graph_docs:
    
    # Добавляем узлы
    for node in gd.nodes:
        G_clean.add_node(node.id)
        
    # Добавляем связи с УМНЫМ ОБЪЕДИНЕНИЕМ
    for rel in gd.relationships:
        source = rel.source.id
        target = rel.target.id
        rel_type = rel.type
        
        # Проверяем, существует ли уже связь между этими конкретными узлами
        if G_clean.has_edge(source, target):
            existing_label = G_clean[source][target]['label']
            # Если такая связь уже есть, но тип новый — дописываем его с новой строки
            if rel_type not in existing_label:
                G_clean[source][target]['label'] = f"{existing_label}\n{rel_type}"
        else:
            # Если связи еще нет, создаем её
            G_clean.add_edge(source, target, label=rel_type)

# Выводим статистику (число рёбер здесь — это количество уникальных пар узлов)
print(f"Чистый граф: {G_clean.number_of_nodes()} узлов, {G_clean.number_of_edges()} пар связей")

# Создаем фигуру, сделаем её чуть шире для комфортного чтения
plt.figure(figsize=(16, 10))

# Вычисляем координаты. Увеличим k=3.5 (вместо 2.5), чтобы сильнее "растолкать" узлы друг от друга
pos = nx.spring_layout(G_clean, k=3.5, seed=42)

# Рисуем узлы
nx.draw_networkx_nodes(G_clean, pos, node_size=4500, node_color="#4AD989", alpha=0.9)

# Рисуем имена узлов
nx.draw_networkx_labels(G_clean, pos, font_size=8, font_color="black", font_weight="bold")

# Рисуем линии связей (добавим arc3, чтобы встречные стрелки чуть изгибались и не перекрывали друг друга)
nx.draw_networkx_edges(G_clean, pos, arrows=True, arrowsize=20, edge_color="#888", width=2, connectionstyle="arc3,rad=0.05")

# Извлекаем наши склеенные ярлыки
edge_labels = nx.get_edge_attributes(G_clean, 'label')

# Рисуем текст связей. ДОБАВЛЕНО: bbox (белая полупрозрачная подложка под текст), чтобы линии не перечеркивали слова
nx.draw_networkx_edge_labels(
    G_clean, pos, 
    edge_labels, 
    font_size=7,
    font_color="#b32400", # Сделаем текст связей темно-красным для контраста
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5) 
)

plt.title("Граф знаний (Schema-Guided Extraction)")
plt.axis('off')
plt.tight_layout()

# Сохраняем и выводим
plt.savefig("/workspace/knowledge_graph_clean.png", dpi=150, bbox_inches='tight')
plt.show()

print("Чистый граф сохранён: /workspace/knowledge_graph_clean.png")
```

---


```python
# ── ЯЧЕЙКА 4 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Глубокий запрос к графу (2-hop) + LLM синтез

def get_entity_context_deep(graph: nx.DiGraph, entity: str) -> str:
    if entity not in graph:
        return f"'{entity}' не найдена в графе"
        
    # Используем множество (set), чтобы автоматически удалять дубликаты связей
    parts = set()
    
    # Сохраняем соседей первого круга (чтобы потом пройтись по ним)
    direct_neighbors = set()
    
    # --- ШАГ 1: Ищем прямые связи (1-hop) ---
    for _, target, data in graph.edges(entity, data=True):
        parts.add(f"{entity} → [{data['label']}] → {target}")
        direct_neighbors.add(target)
        
    for source, _, data in graph.in_edges(entity, data=True):
        parts.add(f"{source} → [{data['label']}] → {entity}")
        direct_neighbors.add(source)
        
    # --- ШАГ 2: Ищем связи связей (2-hop) ---
    # Проходим по всем найденным проектам, компаниям и системам из Шага 1
    for neighbor in direct_neighbors:
        for _, target, data in graph.edges(neighbor, data=True):
            parts.add(f"{neighbor} → [{data['label']}] → {target}")
            
        for source, _, data in graph.in_edges(neighbor, data=True):
            parts.add(f"{source} → [{data['label']}] → {neighbor}")
            
    return "\n".join(sorted(parts))

# ❗️ ВАЖНО: Передаем G_clean, а не G
context = get_entity_context_deep(G_clean, "Иванов Сергей")

print(f"Глубокий контекст (2-hop) для 'Иванов Сергей':\n{context}\n")

answer = llm.invoke(
    f"На основе строго этих данных графа:\n{context}\n\n"
    f"Вопрос: Какие проекты курирует Иванов Сергей и кто с ними связан? "
    f"Отвечай подробно, перечисляя роли людей и компаний."
).content

print(f"Ответ LLM:\n{answer}")
```

---

## ЧАСТЬ 7: АГЕНТНЫЕ СИСТЕМЫ И LANGGRAPH (БЛОК 5)

### Теория (25 мин)

**📌 Термин: Chain** — фиксированная последовательность шагов. Всегда одинаково: шаг 1 → шаг 2 → ответ.

**📌 Термин: Agent** — система которая сама решает какие шаги предпринять. Получает задачу, выбирает инструмент, получает результат, решает что дальше.

**📌 Термин: Tool Calling** — способность LLM возвращать структурированный запрос на вызов функции. Qwen3 поддерживает нативно.

**📌 Термин: State** — TypedDict, словарь-контейнер данных передаваемый между узлами LangGraph. Память агента.

**📌 Термин: Node** — Python-функция принимающая State и возвращающая обновлённый State.

**📌 Термин: Edge** — связь между узлами. Обычные: A→B всегда. Conditional: A→B или A→C по содержимому State.

**📌 Термин: ToolNode** — встроенный узел LangGraph, выполняющий инструменты из tool_calls последнего сообщения.

**📌 Термин: add_messages** — аннотация для поля messages в State. Новые сообщения ДОБАВЛЯЮТСЯ к существующим, не перезаписывают.

### notebook_05_agents.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Создание инструментов

# Импортируем декоратор tool, который превращает обычные функции в инструменты для LLM
from langchain_core.tools import tool
# Импортируем класс для работы с локальными моделями через сервер Ollama
from langchain_ollama import ChatOllama

# Инициализируем объект большой языковой модели (LLM)
llm = ChatOllama(
    # Указываем конкретную модель, которую хотим использовать (должна быть скачана в Ollama)
    model="qwen3:8b",
    # Указываем локальный адрес и порт, на котором запущен сервис Ollama
    base_url="http://localhost:11434",
    # Отключаем встроенный механизм дополнительных рассуждений (reasoning)
    reasoning=False,
    # Устанавливаем температуру в 0, чтобы ответы были максимально точными и детерминированными (без креативности)
    temperature=0,
)

# Используем декоратор @tool, чтобы LangChain понял, что эту функцию модель может вызывать сама
@tool
# Объявляем функцию получения информации о сотруднике (обязательно с указанием типов аргументов)
def get_employee_info(employee_id: str) -> str:
    """Возвращает информацию о сотруднике по ID.
    Использовать когда спрашивают о конкретном сотруднике.
    """
    # Создаем словарь (Mock-базу данных), имитирующий хранилище сотрудников
    employees = {
        "EMP001": {"name": "Иванов Сергей", "dept": "ИТ", "role": "Менеджер проектов"},
        "EMP002": {"name": "Петрова Анна", "dept": "Финансы", "role": "Аналитик"},
    }
    # Проверяем, есть ли переданный ID (ключ) в нашем словаре сотрудников
    if employee_id in employees:
        # Извлекаем словарь с данными конкретного сотрудника
        e = employees[employee_id]
        # Формируем и возвращаем читаемую строку с именем, отделом и должностью
        return f"{e['name']}, отдел {e['dept']}, роль: {e['role']}"
    # Если сотрудник с таким ID не найден, возвращаем уведомление об этом
    return f"Сотрудник {employee_id} не найден"

# Превращаем функцию поиска по документации в инструмент для LLM
@tool
# Функция принимает поисковый запрос в виде строки и возвращает текстовый результат
def search_documentation(query: str) -> str:
    """Ищет ответ в корпоративной документации.
    Для вопросов о технических процедурах и настройках.
    """
    # Вызываем ранее созданный ретривер (поисковик) для поиска релевантных кусков текста по запросу
    results = hybrid_retriever.invoke(query)
    # Проверяем, вернул ли поиск хотя бы один документ
    if results:
        # Берем первые 2 результата, обрезаем текст каждого до 300 символов и склеиваем их через пустую строку
        return "\n\n".join([r.page_content[:300] for r in results[:2]])
    # Если ретривер ничего не нашел, возвращаем соответствующее сообщение
    return "Информация не найдена"

# Превращаем функцию калькулятора в инструмент для LLM
@tool
# Функция принимает количество серверов и часов (целые числа), а также стоимость часа (число с плавающей точкой по умолчанию)
def calculate_cost(servers: int, hours: int, gpu_cost_per_hour: float = 0.74) -> str:
    """Рассчитывает стоимость GPU серверов.
    Args:
        servers: количество серверов
        hours: количество часов
        gpu_cost_per_hour: стоимость одного GPU в час
    """
    # Вычисляем общую стоимость, перемножая все входные параметры
    total = servers * hours * gpu_cost_per_hour
    # Возвращаем форматированную строку с формулой расчета и итоговой суммой (округленной до 2 знаков после запятой)
    return f"{servers} серверов × {hours} ч × ${gpu_cost_per_hour}/ч = ${total:.2f}"

# Собираем все три созданные функции-инструмента в один список, чтобы потом передать их агенту
tools = [get_employee_info, search_documentation, calculate_cost]
# Выводим в консоль информацию о том, сколько инструментов было успешно загружено
print(f"Инструментов: {len(tools)}")
```

```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~5 мин | 📌 Сборка LangGraph агента

# Импортируем Annotated для добавления метаданных (правил обновления) к типам данных
from typing import Annotated
# Импортируем TypedDict для создания словаря со строгой типизацией ключей состояния
from typing_extensions import TypedDict
# Импортируем классы для построения графа состояний и специальные узлы начала и конца (START, END)
from langgraph.graph import StateGraph, START, END
# Импортируем функцию add_messages, которая отвечает за правильное склеивание (добавление) новых сообщений к истории
from langgraph.graph.message import add_messages
# Импортируем готовый узел ToolNode, который умеет автоматически запускать нужные инструменты из списка
from langgraph.prebuilt import ToolNode
# Импортируем базовые классы сообщений для правильной типизации данных, которые понимает LLM
from langchain_core.messages import BaseMessage, SystemMessage

# Объявляем класс состояния (AgentState), который будет передаваться между узлами нашего графа
class AgentState(TypedDict):
    # Указываем, что ключ messages хранит список сообщений, а add_messages гарантирует, что новые сообщения будут добавляться в конец, а не перезаписывать старые
    messages: Annotated[list[BaseMessage], add_messages]

# Привязываем список наших инструментов (созданных в предыдущей ячейке) к языковой модели
llm_with_tools = llm.bind_tools(tools)

# Определяем функцию-узел агента, которая принимает текущее состояние (включая всю историю переписки)
def agent_node(state: AgentState) -> dict:
    # Передаем историю сообщений в модель, чтобы она сгенерировала ответ (текст или вызов инструмента)
    response = llm_with_tools.invoke(state["messages"])
    # Возвращаем словарь с новым сообщением (оно автоматически добавится в общий список благодаря add_messages)
    return {"messages": [response]}

# Определяем функцию-маршрутизатор (conditional edge), которая решает, куда графу идти дальше
def should_continue(state: AgentState) -> str:
    # Извлекаем самое последнее сообщение из истории (только что сгенерированное моделью)
    last = state["messages"][-1]
    # Проверяем, содержит ли это последнее сообщение запрос на вызов инструмента (tool_calls)
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Если инструмент нужен, направляем граф в узел "tools"
        return "tools"
    # Если вызовов инструментов нет, значит модель дала финальный текстовый ответ, и мы направляем граф в конец ("end")
    return "end"

# Создаем узел для выполнения инструментов, передав ему наш список функций (tools)
tool_node = ToolNode(tools)

# Инициализируем строитель графа, передав ему структуру нашего состояния (AgentState)
builder = StateGraph(AgentState)
# Регистрируем в графе узел "agent", который выполняет функцию agent_node
builder.add_node("agent", agent_node)
# Регистрируем в графе узел "tools", который будет выполнять инструменты через tool_node
builder.add_node("tools", tool_node)
# Создаем ребро от старта (START) к узлу "agent" — именно оттуда всегда начинается выполнение графа
builder.add_edge(START, "agent")
# Добавляем условный переход из узла "agent": вызываем should_continue и идем либо в "tools", либо завершаем работу (END)
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
# Создаем ребро возврата: после выполнения инструментов ("tools") всегда возвращаемся обратно к агенту ("agent") для анализа результатов
builder.add_edge("tools", "agent")

# Компилируем (собираем) граф в готовое приложение, которое можно запускать (invoke)
agent_app = builder.compile()
# Выводим в консоль подтверждение, что сборка графа прошла успешно
print("Агент скомпилирован ✅")
```

```python
# ── ЯЧЕЙКА 3 ──────────────────────────────────────────────────────────────
# ⏱ ~2 мин | 📌 Отрисовка структуры агента (NetworkX + Matplotlib)

# Импортируем библиотеку networkx для создания и управления математическим графом
import networkx as nx
# Импортируем pyplot из matplotlib для визуализации (отрисовки) графа
import matplotlib.pyplot as plt

# Извлекаем внутренний объект графа (со всеми узлами и связями) из скомпилированного агента LangGraph
drawable_graph = agent_app.get_graph()

# Создаем НОВЫЙ пустой направленный граф (Directed Graph, так как связи имеют направление - стрелки)
G_agent = nx.DiGraph()

# Проходим циклом по всем узлам (nodes), которые есть во внутреннем графе LangGraph
for node_id in drawable_graph.nodes.keys():
    # Проверяем, является ли узел стартовым (LangGraph называет его "__start__")
    if node_id == "__start__":
        # Заменяем системное имя на красивое "START"
        display_name = "START"
    # Проверяем, является ли узел конечным (LangGraph называет его "__end__")
    elif node_id == "__end__":
        # Заменяем системное имя на "END"
        display_name = "END"
    # Для всех остальных узлов (agent, tools)
    else:
        # Оставляем имя как есть
        display_name = node_id
    
    # Добавляем узел с финальным именем в наш граф NetworkX
    G_agent.add_node(display_name)

# Проходим циклом по всем связям (переходам/edges) нашего графа LangGraph
for edge in drawable_graph.edges:
    # Определяем источник стрелки (откуда идет связь), попутно заменяя системное имя "__start__"
    source = "START" if edge.source == "__start__" else edge.source
    # Определяем цель стрелки (куда идет связь), попутно заменяя системное имя "__end__"
    target = "END" if edge.target == "__end__" else edge.target
    
    # Инициализируем пустую строку для текста над стрелкой (например, название условного перехода)
    label = ""
    # Проверяем, есть ли у текущей связи атрибут data (там LangGraph хранит имена условий)
    if hasattr(edge, 'data') and edge.data is not None:
        # Если данные есть, конвертируем их в строку (чтобы нарисовать текст)
        label = str(edge.data)
        
    # Добавляем стрелку (ребро) между источником и целью, прикрепляя к ней найденную текстовую метку
    G_agent.add_edge(source, target, label=label)

# Выводим в консоль количество получившихся узлов и связей
print(f"Граф агента: {G_agent.number_of_nodes()} узлов, {G_agent.number_of_edges()} переходов")

# Создаем фигуру (холст) для отрисовки, задаем пропорции 10 на 6 дюймов
plt.figure(figsize=(10, 6))

# Вычисляем координаты (pos) для узлов: spring_layout использует физическую модель отталкивания узлов друг от друга
pos = nx.spring_layout(G_agent, k=2.0, seed=42)

# Рисуем кружочки узлов (node_size=4000 делает их крупными, цвет задаем синий #4A90E2)
nx.draw_networkx_nodes(G_agent, pos, node_size=4000, node_color="#4A90E2", alpha=0.9)

# Рисуем текст (имена узлов) внутри кружочков, делаем шрифт белым и жирным для читаемости
nx.draw_networkx_labels(G_agent, pos, font_size=10, font_color="white", font_weight="bold")

# Рисуем стрелки переходов (arc3,rad=0.15 слегка изгибает их, чтобы встречные стрелки не сливались в одну линию)
nx.draw_networkx_edges(G_agent, pos, arrows=True, arrowsize=20, edge_color="#555", width=2, connectionstyle="arc3,rad=0.15")

# Извлекаем словарь с нашими текстовыми метками, которые мы прикрепили к связям (например, "tools" или "end")
edge_labels = nx.get_edge_attributes(G_agent, 'label')

# Рисуем текст поверх стрелок (условия перехода), добавляем белую полупрозрачную подложку (bbox), чтобы линии не зачеркивали текст
nx.draw_networkx_edge_labels(
    G_agent, pos, 
    edge_labels=edge_labels, 
    font_size=9,
    font_color="#b32400", # Цвет текста темно-красный
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3)
)

# Добавляем заголовок нашему графику
plt.title("Визуализация графа LangGraph (Agent State)", fontsize=14, fontweight="bold")
# Отключаем отображение осей координат (рамки с цифрами по краям), так как для графов они не нужны
plt.axis('off')
# Применяем функцию tight_layout, чтобы график занял максимум места на холсте без обрезки краев
plt.tight_layout()

# Сохраняем получившуюся картинку в файл PNG на диск
plt.savefig("langgraph_agent_visualization.png", dpi=150, bbox_inches='tight')
# Команда, которая выводит картинку на экран (прямо под ячейкой Jupyter)
plt.show()

# Выводим сообщение о том, что визуализация успешно завершена
print("Граф успешно визуализирован и сохранён в файл: langgraph_agent_visualization.png")
```

Ожидаемый вывод:
```
Структура агента (Mermaid-синтаксис):
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__ --> agent;
    agent --> tools;
    agent --> __end__;
    tools --> agent;

Граф успешно визуализирован и сохранён в файл: langgraph_agent_visualization.png

```

```python
# ── ЯЧЕЙКА 4 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 ПОДКЛЮЧЕНИЕ CHROMA DB, СОЗДАНИЕ ИНСТРУМЕНТОВ И СБОРКА АГЕНТА

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage

print("⏳ Подключаемся к реальной базе данных...")

# 1. ПОДКЛЮЧЕНИЕ ВАШЕГО РЕАЛЬНОГО RAG (Как в Ячейке 3а)
embeddings_nomic = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
chroma_client = chromadb.PersistentClient(path="/workspace/chroma_data")
vectorstore_chroma = Chroma(
    client=chroma_client,
    collection_name="enterprise_docs",
    embedding_function=embeddings_nomic,
)
# Используем k=3, как было в вашем успешном ручном поиске
real_retriever = vectorstore_chroma.as_retriever(search_kwargs={"k": 3})

print(f"✅ База подключена! Документов: {vectorstore_chroma._collection.count()}")


# 2. ИНСТРУМЕНТЫ (ФЕЙКОВЫЙ СЛОВАРЬ УДАЛЕН НАВСЕГДА)
@tool
def get_employee_info(employee_id: str) -> str:
    """Возвращает информацию о сотруднике по ID."""
    employees = {
        "EMP001": {"name": "Иванов Сергей", "dept": "ИТ", "role": "Менеджер проектов"},
    }
    return f"{employees[employee_id]['name']}, {employees[employee_id]['role']}" if employee_id in employees else "Не найден"

@tool
def search_documentation(query: str) -> str:
    """Ищет ответ в корпоративной документации (инструкции, Astra Linux, FreeIPA).
    ВАЖНО: Формулируй запрос подробно, включая версии ОС.
    """
    # ⚠️ ЗДЕСЬ БОЛЬШЕ НЕТ СЛОВАРЯ! ТОЛЬКО РЕАЛЬНЫЙ ПОИСК ПО CHROMA DB!
    results = real_retriever.invoke(query)
    
    if results:
        # Склеиваем найденные куски текста, как в Ячейке 3а
        return "\n\n".join([f"[{i+1}] {r.page_content}" for i, r in enumerate(results)])
    return "Информация в документации не найдена."

@tool
def calculate_cost(servers: int, hours: int, gpu_cost_per_hour: float = 0.74) -> str:
    """Рассчитывает стоимость GPU серверов."""
    return f"${servers * hours * gpu_cost_per_hour:.2f}"

tools = [get_employee_info, search_documentation, calculate_cost]


# 3. ПЕРЕСБОРКА АГЕНТА (Чтобы он "забыл" старые фейковые инструменты)
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False, temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

# Перезаписываем глобальную переменную agent_app новым агентом
agent_app = builder.compile()

print("✅ Агент пересобран с новыми инструментами!")
```

```python
# ── ЯЧЕЙКА 5a ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 ЗАПУСК АГЕНТА И КРАСИВЫЙ АНАЛИЗ ЕГО ШАГОВ (ВЫВОД РЕЗУЛЬТАТОВ)

# Импортируем SystemMessage и HumanMessage (обязательно здесь, чтобы не было ошибки NameError)
from langchain_core.messages import SystemMessage, HumanMessage

print("🚀 Инициализация комплексного запроса к агенту...\n")

# Создаем системный промпт — задаем "роль" для LLM
system_msg = SystemMessage(content="Ты корпоративный ИИ-ассистент. Отвечай на русском. Используй доступные инструменты для точных ответов.")

# Создаем запрос пользователя. В нём специально зашито 3 РАЗНЫХ задачи, 
# чтобы заставить агента вызвать все 3 инструмента (сотрудник, база знаний, калькулятор)
user_msg = HumanMessage(content=(
    "Найди информацию о сотруднике EMP001, узнай как установить FreeIPA Server на Astra Linux 1.8, "
    "и посчитай стоимость 3 GPU серверов на 8 часов."
))

print("⏳ Агент думает и планирует шаги (это может занять 10-30 секунд)...\n")

# Запускаем выполнение графа (invoke) и передаем стартовые сообщения
# В этот момент LangGraph начнет ходить по узлам (agent -> tools -> agent...)
result = agent_app.invoke({"messages": [system_msg, user_msg]})

# =====================================================================
# КРАСИВЫЙ ВЫВОД ИСТОРИИ (ЛОГОВ) РАБОТЫ АГЕНТА
# =====================================================================

print("="*70)
print(" 🕵️‍♂️ ЛОГ РАБОТЫ АГЕНТА (ШАГ ЗА ШАГОМ)")
print("="*70)

# result["messages"] содержит всю историю переписки, включая скрытые вызовы функций. 
# Проходимся циклом по каждому сообщению по порядку:
for msg in result["messages"]:
    
    # Получаем тип сообщения (HumanMessage, AIMessage или ToolMessage)
    cls = msg.__class__.__name__
    
    # 1. Если это сообщение от нас (пользователя)
    if cls == "HumanMessage":
        print(f"👤 ПОЛЬЗОВАТЕЛЬ:\n   {msg.content}")
        print("-" * 70)
        
    # 2. Если это сообщение от ИИ, и в нем есть запрос на вызов инструментов (tool_calls)
    elif cls == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
        print("🧠 АГЕНТ РЕШИЛ ИСПОЛЬЗОВАТЬ ИНСТРУМЕНТЫ:")
        # Нейросеть может вызвать несколько функций параллельно, поэтому перебираем их циклом
        for tc in msg.tool_calls:
            # Выводим имя функции и переданные в нее аргументы
            print(f"   🔧 Вызов: {tc['name']} \n   📥 Аргументы: {tc['args']}")
        print("-" * 70)
        
    # 3. Если это системное сообщение с ответом от выполненной Python-функции (инструмента)
    elif cls == "ToolMessage":
        # Если ответ от базы знаний огромный, обрезаем его до 300 символов для красоты вывода
        content_preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
        # Выводим имя инструмента, который отработал, и его результат
        print(f"📊 РЕЗУЛЬТАТ ИНСТРУМЕНТА [{msg.name}]:\n   {content_preview}")
        print("-" * 70)
        
    # 4. Если это сообщение от ИИ, НО в нем больше нет вызовов инструментов
    # Это значит, что агент собрал все данные и сгенерировал окончательный текстовый ответ
    elif cls == "AIMessage" and not (hasattr(msg, "tool_calls") and msg.tool_calls):
        print("🤖 ФИНАЛЬНЫЙ ОТВЕТ АГЕНТА:\n")
        # Печатаем финальный ответ
        print(msg.content)
        print("=" * 70)
```

```python
# ── ЯЧЕЙКА 5b (РЕЖИМ ГЛУБОКОЙ ОТЛАДКИ) ─────────────────────────────────────
# ⏱ ~3 мин | 📌 ЗАПУСК АГЕНТА И МОНИТОРИНГ ПОДКАПОТНЫХ ПРОЦЕССОВ

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

print("🚀 Запускаем агента с режимом глубокой отладки (Debug Mode)...\n")

system_msg = SystemMessage(content="Ты корпоративный ИИ-ассистент. Отвечай на русском. Обязательно используй инструменты для поиска информации.")

user_msg = HumanMessage(content=(
    "Найди информацию о сотруднике EMP001, узнай как установить FreeIPA Server на Astra Linux 1.8, "
    "и посчитай стоимость 3 GPU серверов на 8 часов."
))

# Запускаем графа
result = agent_app.invoke({"messages": [system_msg, user_msg]})

# =====================================================================
# РАСШИРЕННЫЙ АНАЛИЗАТОР ШАГОВ
# =====================================================================

print("\n" + "═"*80)
print(" 🕵️‍♂️ ДЕТАЛЬНЫЙ ЛОГ РАБОТЫ АГЕНТА (РАССЛЕДОВАНИЕ)")
print("═"*80)

for i, msg in enumerate(result["messages"]):
    print(f"\n[Шаг {i}] Тип: {msg.__class__.__name__}")
    
    # 1. Запрос пользователя
    if isinstance(msg, HumanMessage):
        print(f"👤 ПОЛЬЗОВАТЕЛЬ: {msg.content}")
        
    # 2. Мысли агента и вызов инструментов
    elif isinstance(msg, AIMessage):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("🧠 АГЕНТ ПРИНЯЛ РЕШЕНИЕ ИСКАТЬ ДАННЫЕ:")
            for tc in msg.tool_calls:
                print(f"   ➡️ ВЫБРАН ИНСТРУМЕНТ: {tc['name']}")
                print(f"   📥 СГЕНЕРИРОВАННЫЙ ЗАПРОС: {tc['args']}")
                
                # Анализируем, насколько хороший запрос придумал агент
                if tc['name'] == 'search_documentation':
                    query_text = tc['args'].get('query', '')
                    if len(query_text) < 15:
                        print("   ⚠️ ВНИМАНИЕ: Агент придумал слишком короткий запрос! RAG может не сработать.")
        elif msg.content:
            print("🤖 ФИНАЛЬНЫЙ ИЛИ ПРОМЕЖУТОЧНЫЙ ТЕКСТ АГЕНТА:")
            print(f"   💬 {msg.content}")
            
    # 3. Реальный ответ от инструментов (самое важное для RAG!)
    elif isinstance(msg, ToolMessage):
        print(f"🛠️ ОТВЕТ ОТ ИНСТРУМЕНТА '{msg.name}':")
        print(f"   📏 Длина полученных данных: {len(msg.content)} символов")
        
        # Детально проверяем ответ от нашей базы знаний
        if msg.name == "search_documentation":
            if len(msg.content) < 100 or "не найдена" in msg.content.lower():
                print("   ❌ ОШИБКА RAG: База вернула заглушку или слишком мало данных!")
                print(f"   📄 ЧТО ВЕРНУЛА БАЗА: {msg.content}")
            else:
                print("   ✅ RAG ОТРАБОТАЛ! База нашла документы. Вот первые 500 символов:")
                print(f"   📄 {msg.content[:500]}...\n   (остальной текст скрыт для читаемости)")
        else:
             print(f"   📄 Содержимое: {msg.content[:200]}")

print("\n" + "═"*80)
```

Ожидаемый вывод:
```
🔧 ВЫЗОВ: get_employee_info({'employee_id': 'EMP001'})
📊 РЕЗУЛЬТАТ [get_employee_info]: Иванов Сергей, отдел ИТ, роль: Менеджер проектов

🔧 ВЫЗОВ: search_documentation({'query': 'FreeIPA установка Astra Linux 1.8'})
📊 РЕЗУЛЬТАТ [search_documentation]: sudo apt install astra-freeipa-server...

🔧 ВЫЗОВ: calculate_cost({'servers': 3, 'hours': 8})
📊 РЕЗУЛЬТАТ [calculate_cost]: 3 серверов × 8 ч × $0.74/ч = $17.76

🤖 ОТВЕТ:
Сотрудник EMP001 — Иванов Сергей (ИТ, Менеджер проектов).
Установка FreeIPA: sudo apt install astra-freeipa-server
Стоимость 3 GPU на 8 часов: $17.76
```

---

## ЧАСТЬ 8: УПРАВЛЕНИЕ КОНТЕКСТОМ И FALLBACK (БЛОК 6)

**📌 Термин: Контекстное окно** — максимум токенов для LLM за один вызов. Qwen3:8b — 32 000 токенов.

**📌 Термин: Lost-in-the-middle** — феномен: LLM хуже учитывает информацию из середины длинного контекста.

**📌 Термин: RemoveMessage** — класс из `langchain_core.messages`. Возврат `{"messages": [RemoveMessage(id=msg.id)]}` атомарно удаляет сообщение из State.

**📌 Термин: REMOVE_ALL_MESSAGES** — sentinel из `langgraph.graph.message`. Удаляет всю историю одним вызовом.

**📌 Термин: GraphRecursionError** — исключение LangGraph при достижении `recursion_limit` (по умолчанию 25). Возникает при зацикливании агента.

**Три стратегии:**

```
Sliding Window:       оставляем последние N сообщений
                      ⚠️ ВСЕГДА include_system=True!

Явная суммаризация:   узел LangGraph сжимает историю
                      + сохраняем смысл, - доп. LLM вызов

Разделение данных:    raw данные инструментов в отдельных ключах State
                      в messages → только синтезированный текст
```

---

В Ячейке 5b использовался простой агент с ручным debug-логированием:
- LLM вызывала инструменты последовательно, а мы разбирали сырой State вручную через цикл по `messages`.
- В Ячейке 6 мы перешли на LangGraph StateGraph — формальную State Machine архитектуру, где состояние (TicketAgentState) явно типизировано, узлы (agent_node, ToolNode) взаимодействуют через именованные переходы (conditional_edges), и граф `compile()` статически проверяет топологию при запуске.
Ключевое отличие:
- поле `messages` использует специальный reducer `add_messages`, который накапливает историю диалога вместо перезаписи — критично для агентов с цепочками из 4+ инструментов.
- Вместо ручного разбора через isinstance() мы используем `astream_events(version="v2")` — встроенный API LangGraph, который транслирует события on_chat_model_start, on_tool_start, on_tool_end в реальном времени без overhead обработки всего State.
Архитектура Ячейки 6 производственная: явная топология графа, полная наблюдаемость через события, возможность масштабирования на 10+ инструментов и human-in-the-loop interrupts без переписывания кода.

```python
# ── ЯЧЕЙКА 6a: ИМПОРТЫ, STATE, ИНСТРУМЕНТЫ ────────────────────────────────
# ⏱ ~1 мин | 📌 Определяем контракт данных и четыре инструмента конвейера
# Зависимости: hybrid_retriever из guard-блока

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

# ─── STATE — центральный объект, путешествующий через граф ──────────────────
#
# Два типа полей:
#
#   Annotated[list, add_messages]  — reducer: новые сообщения ДОБАВЛЯЮТСЯ
#                                    к истории (append), а не перезаписывают.
#                                    Без этого агент «забывал» бы предыдущие
#                                    шаги на каждой итерации.
#
#   Обычные str / int              — простая перезапись при каждом обновлении.
#                                    Разные узлы могут отвечать за разные поля
#                                    независимо друг от друга.

class TicketAgentState(TypedDict):
    messages:       Annotated[list, add_messages]  # ← reducer append, не replace
    ticket_content: str   # заполняется read_ticket_file
    employee_info:  str   # заполняется get_employee_info
    rag_findings:   str   # заполняется search_documentation
    resolution:     str   # заполняется write_resolution_file
    steps_taken:    int   # счётчик итераций (демонстрационный)

# ─── ИНСТРУМЕНТЫ — четыре шага конвейера ────────────────────────────────────

@tool
def read_ticket_file(path: str) -> str:
    """Читает тикет технической поддержки из файла на диске."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ОШИБКА: файл {path} не найден на диске"


@tool
def get_employee_info(employee_id: str) -> str:
    """Возвращает ФИО, должность и отдел сотрудника по корпоративному ID."""
    # В production: запрос к LDAP / HR API
    registry = {
        "EMP001": "Иванов Сергей | Менеджер проектов | Отдел ИТ",
        "EMP042": "Петров Александр | Senior DevOps Engineer | Инфраструктура",
        "EMP100": "Сидоров Иван | Менеджер по развитию | Коммерческий отдел",
    }
    return registry.get(
        employee_id,
        f"Сотрудник {employee_id} не найден в корпоративном реестре"
    )


@tool
def search_documentation(query: str) -> str:
    """Ищет инструкции в корпоративной базе знаний (RAG: Qdrant + BGE-M3)."""
    # hybrid_retriever — из guard-блока, гибридный BM25 + векторный поиск
    docs = hybrid_retriever.invoke(query)
    if not docs:
        return "Документация по запросу не найдена в базе знаний"
    return "\n\n".join(
        f"[Документ {i+1}]:\n{doc.page_content[:600]}"
        for i, doc in enumerate(docs[:3])
    )


@tool
def write_resolution_file(path: str, content: str) -> str:
    """Записывает финальную резолюцию по тикету в Markdown-файл на диске."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Резолюция сохранена: {path} ({len(content)} символов)"


# Список инструментов — порядок влияет на то, как модель видит их в промпте
ticket_tools = [
    read_ticket_file,
    get_employee_info,
    search_documentation,
    write_resolution_file,
]

# LLM с привязанными инструментами
# reasoning=False: в RAG-агенте нужна скорость, не цепочка рассуждений
llm_agent = ChatOllama(model="qwen3:8b", reasoning=False).bind_tools(ticket_tools)

print("✅ 6a готова: State, 4 инструмента, LLM определены")
print(f"   Инструменты: {[t.name for t in ticket_tools]}")
```

---

```python
# ── ЯЧЕЙКА 6b: СБОРКА ГРАФА ────────────────────────────────────────────────
# ⏱ ~10 сек | 📌 Узлы, рёбра, компиляция, визуализация топологии
# Зависимости: всё из Ячейки 6a

# ─── УЗЛЫ ────────────────────────────────────────────────────────────────────

def agent_node(state: TicketAgentState) -> dict:
    """
    Узел агента — единственное место, где вызывается LLM.

    Принимает полный State, возвращает dict только с изменёнными полями.
    LangGraph сам мержит изменения в текущий State — узел не перезаписывает
    State целиком, только объявляет что изменилось.
    """
    response = llm_agent.invoke(state["messages"])
    return {
        "messages":    [response],                         # add_messages → append
        "steps_taken": state.get("steps_taken", 0) + 1,   # перезапись
    }


def should_continue(state: TicketAgentState) -> Literal["tools", "end"]:
    """
    Conditional Edge — функция-роутер, определяет следующий узел.

    Смотрит в State и возвращает строку с именем следующего узла.
    Именно здесь живёт логика цикла: думать дальше или завершить.

      tool_calls в ответе → агент хочет вызвать инструмент → "tools"
      tool_calls нет      → агент решил что задача выполнена → "end"
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # продолжаем цикл Thought → Action → Observation
    return "end"         # выходим из графа


# ─── СБОРКА ГРАФА ────────────────────────────────────────────────────────────
#
# Архитектура:
#
#   START
#     ↓
#   [agent_node]  ←──────────────────────┐
#     ↓                                  │
#   should_continue()                    │
#     ├── "tools" → [ToolNode] ──────────┘
#     └── "end"   → END

builder = StateGraph(TicketAgentState)

# Регистрируем узлы по имени
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(ticket_tools))   # prebuilt: параллельный запуск

# Точка входа
builder.add_edge(START, "agent")

# Conditional edge: agent сам решает куда идти
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",   # есть tool_calls → выполняем инструменты
        "end":   END,       # нет tool_calls → завершаем граф
    }
)

# После инструментов — всегда возвращаемся к агенту
# (агент анализирует результат и решает: нужен ещё инструмент или можно завершить)
builder.add_edge("tools", "agent")

# compile() проверяет топологию статически:
# достижимость узлов, висячие рёбра, наличие пути до END
ticket_agent = builder.compile()

# ─── ВИЗУАЛИЗАЦИЯ ────────────────────────────────────────────────────────────

print("📊 ТОПОЛОГИЯ ГРАФА (Mermaid):")
print("─" * 60)
print(ticket_agent.get_graph().draw_mermaid())
print("─" * 60)
print("✅ 6b готова: граф скомпилирован")
print(f"   Узлы: {list(ticket_agent.get_graph().nodes.keys())}")
```

---

```python
# ── ЯЧЕЙКА 6c: ЗАПУСК АГЕНТА (без RAG — демо-режим) ──────────────────────
# ⏱ ~5-7 мин | 📌 Визуальный конвейер, запись на диск
# RAG заменён на встроенную инструкцию — для чистого демо без зависимостей

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# ─── ИНСТРУМЕНТЫ БЕЗ RAG ─────────────────────────────────────────────────────

@tool
def read_ticket_file(path: str) -> str:
    """Читает тикет технической поддержки из файла на диске."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ОШИБКА: файл {path} не найден"


@tool
def get_employee_info(employee_id: str) -> str:
    """Возвращает ФИО, должность и отдел сотрудника по корпоративному ID."""
    registry = {
        "EMP001": "Иванов Сергей | Менеджер проектов | Отдел ИТ",
        "EMP042": "Петров Александр | Senior DevOps Engineer | Инфраструктура",
        "EMP100": "Сидоров Иван | Менеджер по развитию | Коммерческий отдел",
    }
    return registry.get(employee_id, f"Сотрудник {employee_id} не найден")


@tool
def search_documentation(query: str) -> str:
    """Ищет инструкции в корпоративной базе знаний по Astra Linux и FreeIPA."""
    # Демо-режим: встроенная инструкция вместо RAG
    # В production здесь был бы hybrid_retriever.invoke(query)
    return """
[Документ 1]: Установка FreeIPA Client на Astra Linux 1.8

Шаг 1. Установить клиентское ПО FreeIPA:
  sudo apt install astra-freeipa-client

Шаг 2. Запустить установку клиента FreeIPA:
  sudo astra-freeipa-client -d corp.astra -n server-prod-app-04

Шаг 3. Во время установки появятся два диалоговых окна:
  - Настройка аутентификации Kerberos: укажите домен в верхнем регистре (CORP.ASTRA)
  - Настройка PAM: согласитесь на автоматическое переопределение настроек

Шаг 4. Добавить пользователя в группы безопасности:
  sudo ipa group-add-member devops-team --users=petrov_a
  sudo ipa group-add-member infrastructure-admins --users=petrov_a
  sudo ipa group-add-member sudo-users --users=petrov_a

Шаг 5. Перезагрузить систему:
  sudo reboot

Шаг 6. Проверить аутентификацию:
  kinit petrov_a@CORP.ASTRA
    """


@tool
def write_resolution_file(path: str, content: str) -> str:
    """Записывает финальную резолюцию по тикету в файл на диске."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Резолюция сохранена: {path} ({len(content)} символов)"


# ─── ПЕРЕСБОРКА АГЕНТА ───────────────────────────────────────────────────────

ticket_tools = [read_ticket_file, get_employee_info,
                search_documentation, write_resolution_file]

llm_agent = ChatOllama(model="qwen3:8b", reasoning=False).bind_tools(ticket_tools)


class TicketAgentState(TypedDict):
    messages:       Annotated[list, add_messages]
    ticket_content: str
    employee_info:  str
    rag_findings:   str
    resolution:     str
    steps_taken:    int


def agent_node(state: TicketAgentState) -> dict:
    response = llm_agent.invoke(state["messages"])
    return {"messages": [response], "steps_taken": state.get("steps_taken", 0) + 1}


def should_continue(state: TicketAgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


builder = StateGraph(TicketAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(ticket_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
ticket_agent = builder.compile()

print("✅ Агент пересобран (демо-режим, без RAG)")

# ─── СИСТЕМНЫЙ ПРОМПТ ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты специалист IT-поддержки корпоративного уровня.

ОБЯЗАТЕЛЬНЫЙ ПРОТОКОЛ ОБРАБОТКИ ТИКЕТА — 4 ШАГА:

ШАГ 1 → read_ticket_file
   Прочитай тикет: path="/workspace/ticket_TK2025001847.txt"
   Найди: номер тикета, ID сотрудника (формат EMP###), hostname сервера.

ШАГ 2 → get_employee_info
   Получи данные сотрудника по найденному ID (например EMP042).

ШАГ 3 → search_documentation
   Найди инструкцию: query="установка FreeIPA client Astra Linux 1.8"

ШАГ 4 → write_resolution_file
   Запиши резолюцию в: path="/workspace/resolution_TK2025001847.md"
   Резолюция должна содержать:
   - Данные тикета и сотрудника
   - Пошаговую инструкцию из документации
   - Статус: RESOLVED

КРИТИЧНО: вызови ВСЕ четыре инструмента последовательно.
Финальный текстовый ответ пиши ТОЛЬКО после записи файла."""

# ─── ВИЗУАЛЬНЫЙ КОНВЕЙЕР ─────────────────────────────────────────────────────

async def run_with_pipeline_visualization():

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                "Обработай тикет TK-2025-001847. "
                "Следуй протоколу строго: 4 инструмента последовательно."
            )),
        ],
        "ticket_content": "",
        "employee_info":  "",
        "rag_findings":   "",
        "resolution":     "",
        "steps_taken":    0,
    }

    TOOL_ICONS = {
        "read_ticket_file":     ("📂", "ШАГ 1: ЧИТАЕМ ТИКЕТ С ДИСКА"),
        "get_employee_info":    ("👤", "ШАГ 2: ИЩЕМ СОТРУДНИКА В РЕЕСТРЕ"),
        "search_documentation": ("🔍", "ШАГ 3: ИНСТРУКЦИЯ ИЗ БАЗЫ ЗНАНИЙ"),
        "write_resolution_file":("💾", "ШАГ 4: ЗАПИСЫВАЕМ РЕЗОЛЮЦИЮ НА ДИСК"),
    }

    print()
    print("╔" + "═"*74 + "╗")
    print("║  🚀 АГЕНТ ЗАПУЩЕН  │  LangGraph State Machine + ReAct               ║")
    print("╚" + "═"*74 + "╝")
    print()
    print("📨 ВХОДЯЩИЙ ЗАПРОС")
    print("   Тикет : TK-2025-001847")
    print("   Хост  : server-prod-app-04.corp.local")
    print("   Задача : настройка FreeIPA client для EMP042")
    print()

    thought_step = 0

    async for event in ticket_agent.astream_events(initial_state, version="v2"):
        kind      = event["event"]
        node_name = event.get("metadata", {}).get("langgraph_node", "")

        if kind == "on_chat_model_start" and node_name == "agent":
            thought_step += 1
            print(f"┌{'─'*74}")
            print(f"│  🧠 THOUGHT #{thought_step}  →  Агент анализирует State...")

        elif kind == "on_tool_start":
            tool_name  = event["name"]
            tool_input = event["data"].get("input", {})
            icon, label = TOOL_ICONS.get(tool_name, ("⚡", tool_name))
            print(f"│")
            print(f"│      ↓")
            print(f"│  {icon} ACTION  →  {label}")
            print(f"│      Tool    : {tool_name}()")
            for k, v in tool_input.items():
                v_str = (str(v)[:100] + "…") if len(str(v)) > 100 else str(v)
                print(f"│      Аргумент: {k} = {v_str}")

        elif kind == "on_tool_end":
            output    = str(event["data"].get("output", ""))
            tool_name = event["name"]
            print(f"│      ↓")
            print(f"│  ✅ OBSERVATION  →  {len(output)} символов получено")
            if tool_name == "read_ticket_file":
                print(f"│      Заголовок : {output.split(chr(10))[0]}")
            elif tool_name == "get_employee_info":
                print(f"│      Данные    : {output}")
            elif tool_name == "search_documentation":
                print(f"│      Найдено   : {output[:150].replace(chr(10), ' ')}…")
            elif tool_name == "write_resolution_file":
                print(f"│      Результат : {output}")
            print(f"│")

        elif kind == "on_chat_model_end" and node_name == "agent":
            response = event["data"].get("output")
            if response and not (hasattr(response, "tool_calls") and response.tool_calls):
                print(f"│      ↓")
                print(f"│  🤖 FINAL ANSWER  →  Агент завершил обработку тикета")
                print(f"└{'─'*74}")

    print()
    print("╔" + "═"*74 + "╗")
    print("║  ✅ КОНВЕЙЕР ЗАВЕРШЁН                                                ║")
    print("╚" + "═"*74 + "╝")

    resolution_path = "/workspace/resolution_TK2025001847.md"
    try:
        with open(resolution_path, "r", encoding="utf-8") as f:
            resolution_text = f.read()
        print(f"\n📄 ФАЙЛ НА ДИСКЕ : {resolution_path}")
        print(f"📏 РАЗМЕР        : {len(resolution_text)} символов")
        print("─" * 76)
        print(resolution_text)
        print("─" * 76)
        print("\n✅ State Machine отработал:")
        print("   START → agent → tools × 4 → agent → END")
    except FileNotFoundError:
        print(f"\n⚠️  Файл {resolution_path} не найден")
        print("    Агент не завершил цепочку — проверьте вывод выше")


# ─── ЗАПУСК ──────────────────────────────────────────────────────────────────
await run_with_pipeline_visualization()
```

---

**CrewAI: Role-based Multi-Agent Framework**

Там где LangGraph заставляет разработчика явно рисовать граф (узлы, рёбра, State, conditional edges) — CrewAI предлагает другую ментальную модель: **описываете команду специалистов с ролями, а не машину состояний**. Вы говорите *кто* делает *что*, а не *как именно* переходить между шагами.

**LangChain vs LangGraph vs CrewAI: Три слоя абстракции**

**LangChain** — это низкоуровневая библиотека для работы с LLM. Основные компоненты: LLMChain (шаблон → LLM → парсинг), Retrievers, Memory, Tools. Логика цепочки шагов — в коде разработчика: `chain1() → chain2() → if condition then chain3()`. Нет встроенной концепции "мультиагентности" — вы сами пишете loops и координацию. Удобна для: быстрого скрипта, интеграции одного LLM-сервиса, прототипирования simple retrieval chains.

**LangGraph** — явное моделирование через граф переходов состояния. Вы определяете:
- `State`: структура данных (dict с полями вроде `messages`, `documents`, `current_agent`)
- `Nodes`: функции, которые преобразуют State
- `Edges`: условные переходы между узлами (`if output['next'] == 'tool_use' → goto node_X`)

LangGraph — это микроядро для orchestration. Вся сложность маршрутизации видна в коде. Когда нужны interrupts, human approval, циклы инструментов — LangGraph даёт полный контроль. Цена: боilerplate, необходимо думать в терминах State transitions.

**CrewAI: Role-based Multi-Agent Framework**

Там где LangGraph заставляет явно рисовать граф (узлы, рёбра, State, conditional edges) — CrewAI предлагает другую ментальную модель: **описываете команду специалистов с ролями, а не машину состояний**. Вы говорите *кто* делает *что*, а не *как именно* переходить между шагами.

Три ключевых примитива:

```
Agent  — специалист с ролью, целью, backstory, tools, LLM model
Task   — конкретное задание с description, expected_output, context
         (context = результаты предыдущих Task, видимые агенту)
Crew   — команда агентов + список задач + process (sequential/hierarchical)
```

**Ключевые отличия:**

| Аспект               | LangChain                 | LangGraph              | CrewAI                               |
| -------------------- | ------------------------- | ---------------------- | ------------------------------------ |
| **Абстракция**       | Компоненты LLM-цепочек    | Граф состояний         | Role-based агенты                    |
| **Control flow**     | Вручную в коде            | Явная декларация Edges | Последовательность Tasks или Manager |
| **Memory**           | Список messages (простая) | Custom State (сложная) | Context между Tasks (встроенная)     |
| **Multi-agent**      | Нет (DIY)                 | Возможно (но сложно)   | First-class citizen                  |
| **Условная логика**  | Python if/else            | Conditional edges      | Hierarchical manager или sequence    |
| **Human interrupts** | Нет встроено              | Есть (interrupt API)   | Нет встроено                         |

**Когда CrewAI уместен:**
- Быстрый прототип multi-agent системы
- Задачи с естественной ролевой декомпозицией ("исследователь → аналитик → редактор")
- Порядок шагов очевиден, не нужна сложная условная логика
- Нужна встроенная координация между агентами

**Когда переходить на LangGraph:**
- Нужен кастомный State с несколькими полями и interdependent updates
- Сложный conditional routing (например, "если инструмент вернул ошибку → retry, иначе continue")
- Human-in-the-loop interrupts и approvals
- Детальный контроль над памятью (не просто список messages, а структурированное хранилище)
- Циклы инструментов с recovery logic

**Когда оставаться на LangChain:**
- Single-agent задачи (классический RAG, summarization)
- Быстрая интеграция готовых компонентов
- Не нужна явная оркестрация сложного flow

---

```python
# ── ЯЧЕЙКА 7 ──────────────────────────────────────────────────────────────
# ⏱ ~15 мин первый запуск (установка) | ~7 мин повторный (только демо)
# 📌 CrewAI vs LangGraph — безопасная демонстрация через изолированный venv
#
# ИДЕМПОТЕНТНОСТЬ:
#   Первый запуск  → создаёт venv, устанавливает crewai, запускает демо
#   Повторный запуск → venv уже есть, crewai уже установлен, просто запускает
#
# ПОЧЕМУ VENV А НЕ pip install В ЯЧЕЙКЕ:
#   Мы уже прошли через это. pip install crewai в главном ядре:
#   → chromadb обновляется → Rust panic в SQLite (мы видели это)
#   → protobuf понижается → ломает langfuse
#   → torchcodec появляется → ломает langchain_text_splitters
#   → pydantic обновляется → конфликты везде
#   Subprocess + venv = главное ядро Jupyter не трогается вообще.

import subprocess
import sys
import os

# ─── 0. ДИАГНОСТИКА ГЛАВНОГО ЯДРА ДО ЗАПУСКА ─────────────────────────────────

print("🔍 ДИАГНОСТИКА ГЛАВНОГО ЯДРА (до запуска CrewAI):")
print("─" * 60)

import torch, chromadb, pydantic
torch_ok  = torch.__version__.startswith('2.8')
chroma_ok = chromadb.__version__.startswith('1.1')

print(f"  {'✅' if torch_ok  else '❌'} torch:    {torch.__version__}")
print(f"  {'✅' if chroma_ok else '⚠️ '} chromadb: {chromadb.__version__}")
print(f"  ✅ pydantic: {pydantic.__version__}")

try:
    import torchcodec
    print("  ❌ torchcodec: УСТАНОВЛЕН — опасно!")
    torchcodec_ok = False
except ImportError:
    print("  ✅ torchcodec: отсутствует")
    torchcodec_ok = True

print("─" * 60)

if not torch_ok or not torchcodec_ok:
    print("⛔ Главное ядро в плохом состоянии — сначала восстановите его")
    raise SystemExit("Диагностика провалена")

print("✅ Главное ядро OK — продолжаем\n")

# ─── 1. СОЗДАЁМ VENV ЕСЛИ НЕ СУЩЕСТВУЕТ ──────────────────────────────────────

VENV_DIR    = "/workspace/venv_crewai"
VENV_PYTHON = f"{VENV_DIR}/bin/python"
VENV_PIP    = f"{VENV_DIR}/bin/pip"

if not os.path.exists(VENV_PYTHON):
    print("📦 Создаём изолированный venv для CrewAI...")
    print("   Это займёт ~30 секунд...\n")

    # --without-pip=False: убеждаемся что pip есть в новом venv
    result = subprocess.run(
        [sys.executable, "-m", "venv", VENV_DIR],
        capture_output=False,  # видим вывод напрямую
    )
    if result.returncode != 0:
        raise RuntimeError("Не удалось создать venv")

    print(f"\n✅ venv создан: {VENV_DIR}")
else:
    print(f"⚡ venv уже существует: {VENV_DIR}")

# ─── 2. УСТАНАВЛИВАЕМ CREWAI В VENV ЕСЛИ НЕ УСТАНОВЛЕН ──────────────────────
# Проверяем через import — если упадёт, значит не установлен

check = subprocess.run(
    [VENV_PYTHON, "-c", "import crewai; print(crewai.__version__)"],
    capture_output=True, text=True
)

if check.returncode != 0:
    print("\n📥 Устанавливаем CrewAI в изолированный venv...")
    print("   Это займёт ~10-15 минут (первый раз)...\n")

    # --ignore-installed cryptography: cryptography установлен debian-пакетом
    # без RECORD файла → pip падает при попытке его переустановить.
    # В чистом venv это нужно явно указать чтобы pip не трогал системный.
    install = subprocess.run(
        [VENV_PIP, "install", "crewai",
         "--ignore-installed", "cryptography"],
        capture_output=False,  # видим полный вывод — важно для диагностики
    )

    if install.returncode != 0:
        raise RuntimeError("Установка CrewAI провалилась — см. вывод выше")

    # Перепроверяем после установки
    check2 = subprocess.run(
        [VENV_PYTHON, "-c", "import crewai; print(crewai.__version__)"],
        capture_output=True, text=True
    )
    if check2.returncode != 0:
        raise RuntimeError(f"CrewAI не импортируется после установки: {check2.stderr}")

    print(f"\n✅ CrewAI {check2.stdout.strip()} установлен в venv")
else:
    print(f"⚡ CrewAI {check.stdout.strip()} уже установлен в venv")

# ─── 3. ПИШЕМ SELF-CONTAINED CREWAI СКРИПТ НА ДИСК ──────────────────────────
# Скрипт полностью самодостаточен:
#   - все инструменты определены внутри
#   - нет зависимостей от переменных главного ядра
#   - search_documentation — встроенная инструкция (демо без RAG)

SCRIPT_PATH = "/workspace/crewai_demo.py"

crewai_script = '''
import sys, time

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

print("=" * 70)
print("  CrewAI Demo — изолированный venv_crewai")
print(f"  Python: {sys.version.split()[0]}")
import crewai
print(f"  CrewAI: {crewai.__version__}")
print("=" * 70)
print()

# ── LLM ──────────────────────────────────────────────────────────────────────
# CrewAI использует LiteLLM под капотом.
# Префикс "ollama/" — обязателен, LiteLLM определяет по нему провайдера.
# Ollama запущен в том же поде RunPod на порту 11434.
llm = LLM(
    model="ollama/qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

# ── ИНСТРУМЕНТЫ — CREWAI BaseTool СТИЛЬ ──────────────────────────────────────
# В CrewAI инструменты — наследники BaseTool с методом _run().
# Это аналог @tool декоратора из LangChain, просто другой API.
# Оба подхода передают инструмент агенту через список tools=[].

class ReadTicketFileTool(BaseTool):
    name: str = "read_ticket_file"
    description: str = "Читает тикет технической поддержки из файла на диске"

    def _run(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"ОШИБКА: файл {path} не найден"


class GetEmployeeInfoTool(BaseTool):
    name: str = "get_employee_info"
    description: str = "Возвращает данные сотрудника по корпоративному ID (EMP###)"

    def _run(self, employee_id: str) -> str:
        # В production: запрос к LDAP / HR API
        registry = {
            "EMP001": "Иванов Сергей | Менеджер проектов | Отдел ИТ",
            "EMP042": "Петров Александр | Senior DevOps Engineer | Инфраструктура",
            "EMP100": "Сидоров Иван | Менеджер по развитию | Коммерческий отдел",
        }
        return registry.get(employee_id, f"Сотрудник {employee_id} не найден")


class SearchDocumentationTool(BaseTool):
    name: str = "search_documentation"
    description: str = "Ищет инструкции по Astra Linux и FreeIPA в базе знаний"

    def _run(self, query: str) -> str:
        # Демо-режим: встроенная инструкция без RAG и Qdrant.
        # В production здесь был бы вызов hybrid_retriever.invoke(query)
        return """
[Документ 1]: Установка FreeIPA Client на Astra Linux 1.8

Шаг 1. Установить клиентское ПО FreeIPA:
  sudo apt install astra-freeipa-client

Шаг 2. Запустить настройку клиента:
  sudo astra-freeipa-client -d corp.astra -n server-prod-app-04

Шаг 3. Настройка Kerberos:
  - Укажите домен CORP.ASTRA (верхний регистр обязателен!)

Шаг 4. Настройка PAM:
  - Согласитесь на автоматическое переопределение настроек pam.d

Шаг 5. Добавить пользователя в группы безопасности:
  sudo ipa group-add-member devops-team --users=petrov_a
  sudo ipa group-add-member infrastructure-admins --users=petrov_a
  sudo ipa group-add-member sudo-users --users=petrov_a

Шаг 6. Перезагрузить систему:
  sudo reboot

Шаг 7. Проверить аутентификацию:
  kinit petrov_a@CORP.ASTRA
"""


class WriteResolutionFileTool(BaseTool):
    name: str = "write_resolution_file"
    description: str = "Записывает финальную резолюцию по тикету в файл на диске"

    def _run(self, path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Резолюция сохранена: {path} ({len(content)} символов)"


tools_list = [
    ReadTicketFileTool(),
    GetEmployeeInfoTool(),
    SearchDocumentationTool(),
    WriteResolutionFileTool(),
]

# ── AGENT ─────────────────────────────────────────────────────────────────────
# Ключевое отличие от LangGraph:
#   LangGraph  → вы рисуете ГРАФ (State, узлы, рёбра, conditional_edges)
#   CrewAI     → вы описываете РОЛЬ (кто, зачем, что умеет)
# Модель сама решает порядок действий в рамках своей роли и задач.

it_specialist = Agent(
    role="Senior IT Support Specialist",
    goal=(
        "Обработать тикет строго по протоколу: "
        "прочитать → найти сотрудника → найти инструкцию → записать резолюцию."
    ),
    backstory=(
        "Опытный инженер корпоративной IT-поддержки. "
        "Специализируется на Astra Linux и FreeIPA. "
        "Никогда не пропускает шаги протокола."
    ),
    tools=tools_list,
    llm=llm,
    verbose=True,
    allow_delegation=False,
    # max_iter — ОБЯЗАТЕЛЬНО для production.
    # Без лимита застрявший агент может крутиться бесконечно
    # и накрутить тысячи LLM-вызовов за ночь ($$$).
    max_iter=8,
)

# ── TASKS ─────────────────────────────────────────────────────────────────────
# context=[предыдущая_task] — аналог add_messages в LangGraph State:
# каждая задача видит output предыдущей без повторного чтения файлов.

task_read = Task(
    description=(
        "Прочитай тикет из файла: /workspace/ticket_TK2025001847.txt\\n"
        "Извлеки: номер тикета, ID сотрудника (EMP###), hostname сервера."
    ),
    expected_output="Резюме: номер тикета, ID сотрудника, hostname, суть проблемы.",
    agent=it_specialist,
)

task_employee = Task(
    description="Получи данные сотрудника из реестра по ID из предыдущей задачи.",
    expected_output="ФИО, должность, отдел сотрудника.",
    context=[task_read],
    agent=it_specialist,
)

task_docs = Task(
    description=(
        "Найди инструкцию по установке FreeIPA client на Astra Linux 1.8."
    ),
    expected_output="Пошаговая инструкция с конкретными командами.",
    context=[task_read],
    agent=it_specialist,
)

task_resolve = Task(
    description=(
        "Составь резолюцию и запиши в файл:\\n"
        "path=/workspace/resolution_TK2025001847_crewai.md\\n"
        "Включи: шапку (номер/дата/RESOLVED), данные сотрудника, "
        "инструкцию пошагово, подпись: обработано CrewAI агентом."
    ),
    expected_output="Подтверждение что файл resolution_TK2025001847_crewai.md записан.",
    # Видит ВСЕ предыдущие задачи — полный контекст для резолюции
    context=[task_read, task_employee, task_docs],
    agent=it_specialist,
)

# ── CREW ──────────────────────────────────────────────────────────────────────
# Process.sequential = задачи строго по порядку списка tasks
# (аналог прямых рёбер A→B→C→D в LangGraph без conditional_edges)

crew = Crew(
    agents=[it_specialist],
    tasks=[task_read, task_employee, task_docs, task_resolve],
    process=Process.sequential,
    verbose=True,
)

print()
print("╔" + "═"*68 + "╗")
print("║  🚀 CrewAI ЗАПУЩЕНА │ Role-based Agent + Sequential Tasks         ║")
print("╚" + "═"*68 + "╝")
print()

start_time = time.time()

# kickoff() — синхронный запуск.
# В отличие от LangGraph где нужен await run_with_pipeline_visualization(),
# CrewAI работает синхронно — проще запустить, меньше контроля над потоком.
crew_result = crew.kickoff()
elapsed = time.time() - start_time

print()
print("╔" + "═"*68 + "╗")
print("║  ✅ CrewAI ЗАВЕРШИЛА РАБОТУ                                      ║")
print("╚" + "═"*68 + "╝")
print(f"\\n⏱  Время: {elapsed:.1f} сек")
print(f"\\n📋 ФИНАЛЬНЫЙ OUTPUT (последняя задача):")
print("─" * 70)
print(crew_result.raw)

# Читаем файл который агент записал на диск
resolution_path = "/workspace/resolution_TK2025001847_crewai.md"
try:
    with open(resolution_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"\\n📄 ФАЙЛ НА ДИСКЕ: {resolution_path}")
    print(f"📏 РАЗМЕР: {len(text)} символов")
    print("─" * 70)
    print(text)
except FileNotFoundError:
    print(f"\\n⚠️  Файл {resolution_path} не создан")
    print("    Агент не вызвал write_resolution_file — см. verbose выше")
'''

# Записываем скрипт на диск
with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
    f.write(crewai_script)

print(f"\n✅ Скрипт записан: {SCRIPT_PATH}\n")

# ─── 4. ЗАПУСКАЕМ В ИЗОЛИРОВАННОМ VENV ───────────────────────────────────────
# subprocess.Popen с stdout=PIPE + построчное чтение = real-time вывод.
# Главное ядро Jupyter физически не может быть изменено дочерним процессом —
# это фундаментальное свойство процессной изоляции в Unix.

print("╔" + "═"*74 + "╗")
print("║  ▶  Запускаем crewai_demo.py через /workspace/venv_crewai/bin/python ║")
print("║     Главное ядро Jupyter — полностью изолировано                     ║")
print("╚" + "═"*74 + "╝")
print()

process = subprocess.Popen(
    [VENV_PYTHON, SCRIPT_PATH],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # stderr → stdout чтобы видеть ошибки сразу
    text=True,
    bufsize=1,  # построчная буферизация = вывод появляется по мере генерации
)

# Читаем построчно — видим CrewAI verbose output в реальном времени
for line in process.stdout:
    print(line, end="", flush=True)

process.wait()

print()
if process.returncode == 0:
    print("✅ CrewAI subprocess завершился (код 0)")
else:
    print(f"⚠️  CrewAI subprocess завершился с кодом {process.returncode}")

# ─── 5. ДИАГНОСТИКА ГЛАВНОГО ЯДРА ПОСЛЕ ЗАПУСКА ──────────────────────────────
# Убеждаемся что subprocess ничего не сломал в главном ядре.

print()
print("🔍 ДИАГНОСТИКА ГЛАВНОГО ЯДРА (после запуска CrewAI):")
print("─" * 60)

import importlib
torch    = importlib.import_module("torch")
chromadb = importlib.import_module("chromadb")
pydantic = importlib.import_module("pydantic")

print(f"  {'✅' if torch.__version__.startswith('2.8') else '❌'} torch:    {torch.__version__}")
print(f"  {'✅' if chromadb.__version__.startswith('1.1') else '⚠️ '} chromadb: {chromadb.__version__}")
print(f"  ✅ pydantic: {pydantic.__version__}")

try:
    import torchcodec
    print("  ❌ torchcodec: ПОЯВИЛСЯ — нужно удалить!")
except ImportError:
    print("  ✅ torchcodec: отсутствует")

print("─" * 60)
print("✅ Главное ядро не пострадало — venv изоляция работает\n")

# ─── 6. СРАВНЕНИЕ ПАРАДИГМ ───────────────────────────────────────────────────

print("╔" + "═"*74 + "╗")
print("║  📊 LangGraph (Ячейка 6c) vs CrewAI (Ячейка 7)                   ║")
print("╚" + "═"*74 + "╝")
print()
print(f"  {'Критерий':<40} {'LangGraph':^14} {'CrewAI':^14}")
print("  " + "─"*70)
print(f"  {'Парадигма':<40} {'State Machine':^14} {'Role-based':^14}")
print(f"  {'Безопасность установки':<40} {'✅ Всегда':^14} {'⚠️ Только venv':^14}")
print(f"  {'Контроль потока':<40} {'Полный':^14} {'Ограничен':^14}")
print(f"  {'Скорость прототипа':<40} {'Медленнее':^14} {'Быстрее':^14}")
print(f"  {'Production-ready':<40} {'Да':^14} {'Частично':^14}")
print(f"  {'Кастомный State (TypedDict)':<40} {'Да':^14} {'Нет':^14}")
print(f"  {'Observability':<40} {'astream_events':^14} {'verbose':^14}")
print()
print("  💡 ВЫВОД:")
print("     CrewAI    → быстрый PoC, понятная ролевая модель, только в venv")
print("     LangGraph → production, полный контроль, безопасен в основном env")
print("     Путь: CrewAI PoC за день → переписать ядро на LangGraph в prod")
```


---
### notebook_06_advanced.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Sliding Window через RemoveMessage (Наглядное демо)

import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage

# =====================================================================
# 1. ЗАГЛУШКА: Объявляем структуру AgentState, чтобы избежать NameError
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =====================================================================
# 2. ВАША ФУНКЦИЯ УПРАВЛЕНИЯ ПАМЯТЬЮ
# =====================================================================
def memory_management_node(state: AgentState) -> dict:
    """
    Запускается ПЕРЕД agent_node.
    Если > MAX_MESSAGES: удаляем старые, сохраняем SystemMessage.
    """
    MAX_MESSAGES = 10
    messages = state["messages"]
    
    if len(messages) <= MAX_MESSAGES:
        return {"messages": []}  # ничего не удаляем
    
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Отсекаем старые сообщения, оставляя место для системного промпта
    msgs_to_delete = other_msgs[:-(MAX_MESSAGES - len(system_msgs))]
    
    # Генерируем специальные объекты RemoveMessage по ID сообщения
    delete_ops = [RemoveMessage(id=m.id) for m in msgs_to_delete]
    
    print(f"🧹 [MemoryNode] Превышен лимит памяти! Удалено {len(delete_ops)} старых сообщений.")
    return {"messages": delete_ops}

print("✅ Sliding Window узел создан!\n")


# =====================================================================
# 3. НАГЛЯДНАЯ ДЕМОНСТРАЦИЯ 
# =====================================================================
print("🎬 ДЕМОНСТРАЦИЯ (Имитация длинного диалога)...\n")

# Создаем фейковую историю из 14 сообщений
# Важно: каждому сообщению генерируем уникальный ID (как это делает настоящий LangChain)
mock_history = [SystemMessage(content="Ты полезный ИИ", id=str(uuid.uuid4()))]

for i in range(1, 14):
    if i % 2 != 0:
        msg = HumanMessage(content=f"Старый вопрос {i}", id=str(uuid.uuid4()))
    else:
        msg = AIMessage(content=f"Старый ответ {i}", id=str(uuid.uuid4()))
    mock_history.append(msg)

print(f"📦 Текущий размер истории: {len(mock_history)} сообщений (Лимит: 10)")
print("Начало переписки:")
for m in mock_history[:3]:
    print(f"  - [{m.__class__.__name__}] {m.content}")
print("  ...\n")

# Прогоняем фейковую историю через нашу функцию
mock_state = {"messages": mock_history}
result = memory_management_node(mock_state)

# Показываем, как LangGraph удаляет сообщения
print("\n🛠 ЧТО СГЕНЕРИРОВАЛ УЗЕЛ (Команды для LangGraph):")
for op in result["messages"]:
    # Выводим команду RemoveMessage и начало ID сообщения, которое будет удалено
    print(f"  - 🗑 RemoveMessage(id='{op.id[:8]}...')")

print("\n💡 КАК ЭТО РАБОТАЕТ:")
print("LangGraph не удаляет сообщения напрямую из массива. Вместо этого функция возвращает список объектов `RemoveMessage`.")
print("Редьюсер `add_messages` видит эти объекты, находит в памяти старые сообщения с такими же ID и стирает их.")
print("В итоге останется ровно 10 сообщений: 1 Системное (оно неприкосновенно) и 9 самых свежих из диалога!")
```

```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Суммаризация в LangGraph State (НАГЛЯДНОЕ ДЕМО)

import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages

# =====================================================================
# 1. СТРУКТУРА СОСТОЯНИЯ
# =====================================================================
# Расширяем стандартное состояние: теперь тут есть история сообщений + поле для резюме
class SummarizingAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str   # Здесь будет храниться накопленный сжатый контекст

# =====================================================================
# 2. УЗЕЛ СУММАРИЗАЦИИ (С заглушкой вместо реальной LLM)
# =====================================================================
def summarize_node(state: SummarizingAgentState) -> dict:
    """Узел суммаризации: следит за длиной диалога и сжимает старые сообщения"""
    messages = state["messages"]
    
    # 🚨 Триггер: если сообщений меньше 8, сжимать рано, выходим
    if len(messages) < 8:
        print(f"⏩ [SummarizeNode] Сообщений {len(messages)} < 8. Диалог короткий, пропускаем.")
        return {}
    
    print(f"🛑 [SummarizeNode] Тревога! Сообщений уже {len(messages)}. Запускаем архивацию...")
    
    # Отсекаем старые сообщения для сжатия, оставляем ТОЛЬКО 4 самых свежих 
    # (чтобы агент помнил непосредственный контекст текущей беседы)
    msgs_to_summarize = messages[:-4]
    
    # 🧠 ИМИТАЦИЯ LLM (в реальности тут был бы llm.invoke)
    print(f"   🧠 [LLM Mock] Читаю {len(msgs_to_summarize)} старых сообщений и пишу резюме...")
    fake_summary = (
        f"Пользователь задал вопросы с 1 по {len(msgs_to_summarize)//2}. "
        f"Ассистент успешно ответил. Главная тема: обсуждение настроек системы."
    )
    
    # 🧹 Генерируем команды на удаление старых сообщений из памяти LangGraph
    delete_ops = [RemoveMessage(id=m.id) for m in msgs_to_summarize]
    
    # 📦 Упаковываем наше резюме в новое системное сообщение
    summary_msg = SystemMessage(content=f"Резюме предыдущего диалога: {fake_summary}")
    
    print(f"   🗜️ Сжато {len(msgs_to_summarize)} сообщений -> в 1 короткое системное сообщение.")
    
    # Возвращаем операции удаления, новое сообщение-резюме и обновляем текстовое поле summary
    return {"messages": delete_ops + [summary_msg], "summary": fake_summary}


# =====================================================================
# 3. НАГЛЯДНАЯ ДЕМОНСТРАЦИЯ  
# =====================================================================
print("🎬 ДЕМОНСТРАЦИЯ (Имитация долгой беседы)...\n")
print("="*70)

# Создаем фейковую переписку на 10 сообщений
mock_history = []
for i in range(1, 6):
    mock_history.append(HumanMessage(content=f"Мой долгий и сложный вопрос {i}", id=str(uuid.uuid4())))
    mock_history.append(AIMessage(content=f"Мой развернутый ответ {i}", id=str(uuid.uuid4())))

print(f"📦 ДО СУММАРИЗАЦИИ (В памяти {len(mock_history)} сообщений):")
for m in mock_history:
    print(f"  - [{m.__class__.__name__}] {m.content}")

print("\n" + "="*70)
print("⚙️ ПРОГОНЯЕМ СОСТОЯНИЕ ЧЕРЕЗ УЗЕЛ СУММАРИЗАЦИИ...")
print("="*70)

# Вызываем наш узел
mock_state = {"messages": mock_history, "summary": ""}
result = summarize_node(mock_state)

print("\n" + "="*70)
print("🛠 ЧТО ВОЗВРАЩАЕТ УЗЕЛ (Команды обновления для графа):")
for op in result["messages"]:
    if isinstance(op, RemoveMessage):
        print(f"  🗑 УДАЛИТЬ: Сообщение с ID {op.id[:8]}...")
    elif isinstance(op, SystemMessage):
        print(f"  ➕ ДОБАВИТЬ: [{op.__class__.__name__}] {op.content}")

print("\n💡 КАК ЭТО ОБЪЯСНИТЬ СЛУШАТЕЛЯМ:")
print("1. Мы не удаляем всю историю подчистую. Мы оставляем 4 последних сообщения, чтобы агент 'не терял нить' текущего диалога.")
print("2. Все остальные (старые) сообщения LLM сжимает в пару предложений.")
print("3. В итоге вместо 10 длинных сообщений (которые тратят много токенов) в памяти графа останется 1 короткое резюме + 4 свежих сообщения!")
```

```python
# ── ЯЧЕЙКА 3 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 Fallback по счётчику ошибок (НАГЛЯДНОЕ ДЕМО)

# Импортируем всё необходимое, чтобы ячейка работала абсолютно независимо
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. СТРУКТУРА СОСТОЯНИЯ
# =====================================================================
class ResilientState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Наш главный герой здесь — счетчик ошибок
    error_count: int

# =====================================================================
# 2. ЗАГЛУШКИ (Моки для демонстрации)
# =====================================================================

# Имитируем агента, который "ломается" (например, таймаут API)
def mock_agent_node(state: ResilientState) -> dict:
    # Увеличиваем счетчик ошибок при каждом заходе в агента
    current_errors = state.get("error_count", 0) + 1
    print(f"🤖 [Агент] Попытка №{current_errors}... Пытаюсь вызвать инструмент поиска.")
    
    # Агент формирует запрос к инструменту
    fake_msg = AIMessage(
        content="", 
        tool_calls=[{"name": "broken_database", "args": {}, "id": "call_123"}]
    )
    # Возвращаем сообщение и обновленный счетчик ошибок
    return {"messages": [fake_msg], "error_count": current_errors}

# Имитируем инструмент, который всегда выдает ошибку
def mock_tool_node(state: ResilientState) -> dict:
    print("   🔧 [Инструмент] ОШИБКА 500: База данных недоступна!")
    fake_error = ToolMessage(content="Error 500: Database timeout", name="broken_database", tool_call_id="call_123")
    return {"messages": [fake_error]}

# =====================================================================
# 3. ЛОГИКА МАРШРУТИЗАЦИИ И FALLBACK
# =====================================================================

# Тот самый умный роутер, который следит за счетчиком
def resilient_router(state: ResilientState) -> str:
    # 🚨 ПРОВЕРКА ЛИМИТА: Если ошибок 3 или больше — спасаем ситуацию!
    if state.get("error_count", 0) >= 3:
        print(f"   🚦 [Роутер] ВНИМАНИЕ! Достигнут лимит ошибок ({state['error_count']}). Переключаю на Fallback!")
        return "fallback"
    
    # Если лимит не достигнут, продолжаем стандартную работу (идем в инструменты)
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# Запасной узел, который отрабатывает, когда всё сломалось
def fallback_node(state: ResilientState) -> dict:
    print("   🛡️ [Fallback] Генерирую извинение для пользователя...")
    return {
        "messages": [AIMessage(content=
            "К сожалению, система сейчас недоступна из-за технических проблем. "
            "Пожалуйста, обратитесь в IT-поддержку: ext. 1234 или it@company.local"
        )],
        # Сбрасываем счетчик, чтобы система могла нормально работать дальше
        "error_count": 0
    }

# =====================================================================
# 4. СБОРКА И ЗАПУСК ГРАФА
# =====================================================================

resilient_builder = StateGraph(ResilientState)
resilient_builder.add_node("agent", mock_agent_node)
resilient_builder.add_node("tools", mock_tool_node)
resilient_builder.add_node("fallback", fallback_node)

resilient_builder.add_edge(START, "agent")
# Маршрутизатор стоит сразу после агента
resilient_builder.add_conditional_edges(
    "agent", resilient_router,
    {"tools": "tools", "end": END, "fallback": "fallback"}
)
resilient_builder.add_edge("tools", "agent")
resilient_builder.add_edge("fallback", END)

# Компилируем нашего демо-агента
resilient_app = resilient_builder.compile()

print("✅ Отказоустойчивый граф собран. Запускаем симуляцию!\n")
print("="*70)

# Запускаем графа с тестовым сообщением от пользователя
final_state = resilient_app.invoke({
    "messages": [HumanMessage(content="Найди мне отчет за 2023 год.")],
    "error_count": 0
})

print("="*70)
print("\n🎉 ИТОГОВЫЙ ОТВЕТ ПОЛЬЗОВАТЕЛЮ:")
print(f"💬 {final_state['messages'][-1].content}")
```

---

## ЧАСТЬ 9: МОНИТОРИНГ И PRODUCTION (БЛОК 7)

**📌 Термин: TTFT** (Time to First Token) — время до первого токена. При streaming: < 0.5 сек = хороший UX.

**📌 Термин: WandB Weave** — расширение WandB для трассировки LLM. Уже используем с МК1.

**📌 Термин: Langfuse** — open-source аналог LangSmith. Self-hosted (данные не уходят). Нативная интеграция с LangChain через CallbackHandler.

**Метрики:**
```
Latency        < 8 сек (RAG запрос)
TTFT           < 0.5 сек (streaming)
Token Usage    input + output на запрос
Tool Calls     > 10 на запрос = аномалия
Error Rate     < 1% в production
Fallback Rate  < 5%
```

### notebook_07_monitoring_ui.ipynb

```python
# ── ЯЧЕЙКА 1 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 7.1 + 7.2: Трейс как дерево спанов — реальный RAG-вызов
# Показываем разницу: что видит APM (HTTP 200, 1.5s) vs LLM-трейсинг

import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# ─── GUARD: восстановление стека если ядро было перезапущено ─────────────────

import os

# Шаг 1: Embeddings
if "embeddings_bge" not in globals():
    print("🔄 Загружаем BGE-M3...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_bge = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("BGE-M3 ✅")

# Шаг 2: Qdrant
if "vectorstore_qdrant" not in globals():
    print("🔄 Подключаемся к Qdrant...")
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore

    if "qdrant_client" not in globals():
        try:
            qdrant_client = QdrantClient(path="/workspace/qdrant_data")
        except RuntimeError:
            lock = "/workspace/qdrant_data/.lock"
            if os.path.exists(lock):
                os.remove(lock)
                print(f"🔓 Удалён stale lock")
            qdrant_client = QdrantClient(path="/workspace/qdrant_data")

    count = qdrant_client.count("enterprise_docs_bge").count
    vectorstore_qdrant = QdrantVectorStore(
        client=qdrant_client,
        collection_name="enterprise_docs_bge",
        embedding=embeddings_bge,
    )
    print(f"Qdrant: {count} чанков ✅")

# Шаг 3: chunks для BM25
if "chunks" not in globals():
    print("🔄 Перечитываем документы для BM25...")
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    DOCS_PATH = "/workspace/docs"   # ← ваш путь к PDF
    if os.path.exists(DOCS_PATH):
        loader = DirectoryLoader(
            DOCS_PATH, glob="**/*.pdf",
            loader_cls=PyPDFLoader, show_progress=True
        )
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=60
        ).split_documents(loader.load())
        print(f"chunks: {len(chunks)} ✅")
    else:
        chunks = None
        print(f"⚠️  {DOCS_PATH} не найден — BM25 отключён")

# Шаг 4: hybrid_retriever
if "hybrid_retriever" not in globals():
    from langchain_community.retrievers import BM25Retriever

    vector_retriever = vectorstore_qdrant.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    if chunks:
        bm25_retriever   = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        try:
            from langchain_classic.retrievers import EnsembleRetriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            print("hybrid_retriever = EnsembleRetriever ✅")
        except ImportError:
            hybrid_retriever = vector_retriever
            print("hybrid_retriever = vector_retriever (fallback) ✅")
    else:
        hybrid_retriever = vector_retriever
        print("hybrid_retriever = vector_retriever (без BM25) ✅")

# Шаг 5: rag_prompt (нужен Ячейке 1 и 3)
if "rag_prompt" not in globals():
    from langchain_core.prompts import ChatPromptTemplate
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты ассистент технической поддержки. "
         "Отвечай строго по контексту. Контекст:\n{context}"),
        ("human", "{question}")
    ])
    print("rag_prompt ✅")

print("\n🟢 Стек восстановлен — запускаем мониторинг...")

# ─── SPAN — атомарная единица трейса ─────────────────────────────────────────
# Аналог того что @observe() создаёт автоматически в Langfuse

@dataclass
class Span:
    name:          str
    start_time:    float = field(default_factory=time.time)
    end_time:      Optional[float] = None
    input_preview: str = ""
    output_preview:str = ""
    metadata:      dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def finish(self, output: str = "", **meta):
        self.end_time = time.time()
        self.output_preview = str(output)[:120]
        self.metadata.update(meta)


# ─── LOCAL TRACER — имитация Langfuse @observe() ─────────────────────────────

class LocalTracer:
    """
    Упрощённая версия того что делает Langfuse под капотом.
    @observe() — контекстный менеджер создаёт Span, записывает input/output/latency.
    В production: replace with `from langfuse import observe` + @observe() декоратор.
    """
    def __init__(self, trace_name: str):
        self.trace_name  = trace_name
        self.trace_start = time.time()
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, input_data: str = ""):
        """Аналог @observe() — оборачивает блок кода в именованный Span."""
        s = Span(name=name, input_preview=input_data[:100])
        self.spans.append(s)
        try:
            yield s       # внутри блока: s.finish() можно вызвать явно
        finally:
            if s.end_time is None:
                s.finish()

    def print_trace(self):
        total_ms = (time.time() - self.trace_start) * 1000

        # ── ЧТО ВИДИТ ТРАДИЦИОННЫЙ APM ───────────────────────────────────
        print(f"\n{'═'*72}")
        print(f"  TRACE: {self.trace_name}  ({total_ms:.0f} ms total)")
        print(f"{'═'*72}")
        print(f"\n📊 ЧТО ВИДИТ ТРАДИЦИОННЫЙ APM (Datadog / Prometheus):")
        print(f"   POST /api/chat  →  200 OK  →  {total_ms:.0f} ms")
        print(f"   [всё. больше никакой информации]")
        print(f"\n   → Тормозит БД или LLM? Галлюцинация или неверный ретривал?")
        print(f"   → APM не знает. Это и есть проблема из раздела 7.1.")

        # ── ЧТО ВИДИТ LLM-ТРЕЙСИНГ ───────────────────────────────────────
        print(f"\n🔍 ЧТО ВИДИТ LLM-ТРЕЙСИНГ (Langfuse @observe):")
        print(f"{'─'*72}")

        retrieval_ms = 0.0
        llm_ms       = 0.0

        for i, s in enumerate(self.spans):
            connector = "└──" if i == len(self.spans) - 1 else "├──"
            bar_len   = max(1, int((s.duration_ms / total_ms) * 35))
            bar       = "█" * bar_len + "░" * (35 - bar_len)

            print(f"│ {connector} Span: {s.name:<28} {s.duration_ms:>6.0f} ms  [{bar}]")

            if s.input_preview:
                print(f"│        ↳ input : {s.input_preview}")
            if s.output_preview:
                print(f"│        ↳ output: {s.output_preview}")
            for k, v in s.metadata.items():
                print(f"│        ↳ {k:<15}: {v}")

            if "retrieval" in s.name.lower():
                retrieval_ms += s.duration_ms
            if "llm" in s.name.lower():
                llm_ms += s.duration_ms

        other_ms = total_ms - retrieval_ms - llm_ms

        print(f"{'─'*72}")
        print(f"\n⏱  BREAKDOWN — кто виноват в latency:")
        print(f"   Ретривал (Qdrant + BM25) : {retrieval_ms:>6.0f} ms  "
              f"({retrieval_ms/total_ms*100:.0f}%)")
        print(f"   LLM генерация (Qwen3)    : {llm_ms:>6.0f} ms  "
              f"({llm_ms/total_ms*100:.0f}%)")
        print(f"   Прочее (форматирование)  : {other_ms:>6.0f} ms  "
              f"({other_ms/total_ms*100:.0f}%)")
        print(f"   {'─'*40}")
        print(f"   ИТОГО                    : {total_ms:>6.0f} ms")
        print(f"\n   → Теперь видно ГДЕ тормозит. APM этого не даёт.")
        print(f"{'═'*72}\n")


# ─── ДЕЛАЕМ РЕАЛЬНЫЙ RAG-ВЫЗОВ С ТРАССИРОВКОЙ ────────────────────────────────

QUERY = "Как установить FreeIPA client на Astra Linux 1.8?"

# LLM без reasoning — нам нужна скорость, не рассуждения
llm_monitor = ChatOllama(model="qwen3:8b", reasoning=False)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты ассистент технической поддержки. "
     "Отвечай строго по контексту. Контекст:\n{context}"),
    ("human", "{question}")
])

# Запускаем трейс — каждый with-блок = один Span в Langfuse
tracer = LocalTracer("customer_support_request")

print(f"🚀 Трассируем реальный RAG-запрос: «{QUERY}»\n")

# SPAN 1: форматирование промпта
with tracer.span("prompt_formatting", input_data=QUERY) as s:
    s.finish(output="System: Ты ассистент технической поддержки...",
             prompt_version="v2")

# SPAN 2: ретривал (реальный вызов hybrid_retriever)
with tracer.span("qdrant_retrieval", input_data=QUERY) as s:
    retrieved_docs = hybrid_retriever.invoke(QUERY)
    context = "\n\n".join(d.page_content[:300] for d in retrieved_docs[:3])
    s.finish(
        output=retrieved_docs[0].page_content[:120] if retrieved_docs else "пусто",
        chunks_found=len(retrieved_docs),
        top_source=retrieved_docs[0].metadata.get("source", "—") if retrieved_docs else "—"
    )

# SPAN 3: вызов LLM (реальная генерация)
with tracer.span("llm_call", input_data=QUERY) as s:
    messages   = rag_prompt.format_messages(context=context, question=QUERY)
    response   = llm_monitor.invoke(messages)
    rag_answer = response.content  # сохраняем для Ячейки 3
    s.finish(
        output=rag_answer,
        model="qwen3:8b",
        output_tokens_approx=len(rag_answer.split())
    )

# SPAN 4: парсинг вывода
with tracer.span("output_parsing") as s:
    s.finish(output=f"Validated: {len(rag_answer)} chars, no JSON required")

# Печатаем дерево спанов
tracer.print_trace()

# Сохраняем для следующих ячеек
print(f"💾 rag_answer сохранён для Ячеек 2 и 3.")
```

```
...
🔍 ЧТО ВИДИТ LLM-ТРЕЙСИНГ (Langfuse @observe):
────────────────────────────────────────────────────────────────────────
│ ├── Span: prompt_formatting                 0 ms  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
│        ↳ input : Как установить FreeIPA client на Astra Linux 1.8?
│        ↳ output: System: Ты ассистент технической поддержки...
│        ↳ prompt_version : v2
│ ├── Span: qdrant_retrieval                636 ms  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
│        ↳ input : Как установить FreeIPA client на Astra Linux 1.8?
│        ↳ output: 9 Установить серверное программное обеспечение FreeIPA.
sudo apt install astra-freeipa-server
10 Установить клиентское п
│        ↳ chunks_found   : 5
│        ↳ top_source     : /workspace/data/docs/Astra_Linux_1.8_FreeIPA_Server_Install.txt
│ ├── Span: llm_call                      17720 ms  [█████████████████████████████████░░]
│        ↳ input : Как установить FreeIPA client на Astra Linux 1.8?
│        ↳ output: Для установки FreeIPA client на Astra Linux 1.8 выполните следующие шаги:
... 
```


```python
# ── ЯЧЕЙКА 2 ──────────────────────────────────────────────────────────────
# ⏱ ~4 мин | 📌 7.3: Prometheus-стиль метрик — Counter, Histogram, Gauge
# Запускаем N реальных запросов, собираем метрики, показываем P50/P95/P99

import time, random
from collections import defaultdict

# ─── МИНИ PROMETHEUS CLIENT — чистый Python, без новых пакетов ──────────────
# В production: pip install prometheus_client → те же классы, тот же API

class Counter:
    """Только возрастает. Сброс только при рестарте. Запросы, ошибки, токены."""
    def __init__(self, name: str, description: str = ""):
        self.name = name; self._value = 0.0
        self._labels: dict = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels):
        self._value += amount
        if labels:
            self._labels[tuple(sorted(labels.items()))] += amount

    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Распределение значений по бакетам. Из него считаются перцентили."""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self._observations: list[float] = []

    def observe(self, value: float):
        self._observations.append(value)

    def percentile(self, p: float) -> float:
        if not self._observations: return 0.0
        s = sorted(self._observations)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return s[idx]

    @property
    def mean(self) -> float:
        return sum(self._observations) / len(self._observations) \
            if self._observations else 0.0

    @property
    def count(self) -> int:
        return len(self._observations)


class Gauge:
    """Может расти и убывать. Текущее значение. VRAM, активные сессии."""
    def __init__(self, name: str, description: str = ""):
        self.name = name; self._value = 0.0

    def set(self, v: float): self._value = v
    def inc(self, a: float = 1.0): self._value += a
    def dec(self, a: float = 1.0): self._value -= a

    @property
    def value(self) -> float:
        return self._value


# ─── РЕЕСТР МЕТРИК ───────────────────────────────────────────────────────────

requests_total   = Counter("llm_requests_total",
                            "Всего запросов к LLM-пайплайну")
errors_total     = Counter("llm_errors_total",
                            "Ошибок (таймаут, OOM, parse error)")
tokens_generated = Counter("llm_tokens_generated_total",
                            "Суммарно выходных токенов")
request_latency  = Histogram("llm_request_latency_seconds",
                              "Латентность полного пайплайна RAG")
retrieval_latency= Histogram("retrieval_latency_seconds",
                              "Латентность только ретривала")
active_sessions  = Gauge("llm_active_sessions",
                          "Активных сессий прямо сейчас")

# ─── ТЕСТ-ЗАПРОСЫ ────────────────────────────────────────────────────────────

test_queries = [
    "Как установить FreeIPA client на Astra Linux 1.8?",
    "Как проверить статус службы через systemctl?",
    "Что такое Kerberos и зачем он нужен?",
    "Как установить astra-freeipa-server?",
    "Как добавить пользователя в группу FreeIPA?",
]

print("🔄 Запускаем тестовые запросы и собираем метрики...\n")
print(f"{'─'*72}")

for i, query in enumerate(test_queries, 1):
    active_sessions.inc()
    t_total_start = time.time()

    try:
        # Ретривал — замеряем отдельно
        t_ret = time.time()
        docs  = hybrid_retriever.invoke(query)
        retrieval_latency.observe(time.time() - t_ret)

        # Генерация — только если нашли документы
        ctx   = "\n".join(d.page_content[:200] for d in docs[:2])
        msgs  = rag_prompt.format_messages(context=ctx, question=query)
        resp  = llm_monitor.invoke(msgs)

        elapsed = time.time() - t_total_start
        request_latency.observe(elapsed)
        requests_total.inc()
        tokens_generated.inc(len(resp.content.split()))  # приближение

        status = "✅"
        note   = f"{len(docs)} chunks, ~{len(resp.content.split())} tokens"

    except Exception as e:
        elapsed = time.time() - t_total_start
        request_latency.observe(elapsed)
        requests_total.inc()
        errors_total.inc()
        status = "❌"
        note   = str(e)[:60]

    finally:
        active_sessions.dec()

    print(f"[{i}/{len(test_queries)}] {status}  {elapsed:.2f}s  "
          f"│  {query[:45]:<45}  │  {note}")

# ─── DASHBOARD ───────────────────────────────────────────────────────────────

print(f"\n{'═'*72}")
print(f"  📊 PROMETHEUS DASHBOARD  —  /metrics эндпоинт")
print(f"{'═'*72}")

print(f"\n{'COUNTER метрики':}")
print(f"{'─'*50}")
print(f"  llm_requests_total          {requests_total.value:>6.0f}")
print(f"  llm_errors_total            {errors_total.value:>6.0f}")
print(f"  llm_tokens_generated_total  {tokens_generated.value:>6.0f}")
err_rate = errors_total.value / requests_total.value * 100 \
           if requests_total.value else 0
print(f"  error_rate                  {err_rate:>5.1f} %")

print(f"\n{'HISTOGRAM метрики (latency)':}")
print(f"{'─'*50}")
print(f"  {'Метрика':<35} {'P50':>6} {'P95':>6} {'P99':>6} {'mean':>6}")
print(f"  {'─'*50}")

def fmt(v): return f"{v*1000:.0f}ms"

print(f"  {'llm_request_latency (полный)':<35} "
      f"{fmt(request_latency.percentile(50)):>6} "
      f"{fmt(request_latency.percentile(95)):>6} "
      f"{fmt(request_latency.percentile(99)):>6} "
      f"{fmt(request_latency.mean):>6}")

print(f"  {'retrieval_latency (только БД)':<35} "
      f"{fmt(retrieval_latency.percentile(50)):>6} "
      f"{fmt(retrieval_latency.percentile(95)):>6} "
      f"{fmt(retrieval_latency.percentile(99)):>6} "
      f"{fmt(retrieval_latency.mean):>6}")

print(f"\n{'GAUGE метрики':}")
print(f"{'─'*50}")
print(f"  active_sessions             {active_sessions.value:>6.0f}")

# ── Демонстрация почему P99 важнее среднего ──────────────────────────────────
print(f"\n{'═'*72}")
print(f"  💡 ПОЧЕМУ P99 ВАЖНЕЕ СРЕДНЕГО:")
print(f"{'─'*72}")
mean_ms = request_latency.mean * 1000
p99_ms  = request_latency.percentile(99) * 1000
print(f"  Среднее latency : {mean_ms:.0f} ms  → «всё хорошо»")
print(f"  P99 latency     : {p99_ms:.0f} ms  → каждый 100-й запрос ждёт {p99_ms:.0f} мс")
print(f"")
print(f"  При 1 000 RPS = {10:.0f} плохих запросов в секунду")
print(f"  SLA-соглашения всегда определяются через P95 / P99, не через mean.")
print(f"{'═'*72}")
```

```
...
════════════════════════════════════════════════════════════════════════
  📊 PROMETHEUS DASHBOARD  —  /metrics эндпоинт
════════════════════════════════════════════════════════════════════════

COUNTER метрики
──────────────────────────────────────────────────
  llm_requests_total               5
  llm_errors_total                 0
  llm_tokens_generated_total     201
  error_rate                    0.0 %

HISTOGRAM метрики (latency)
──────────────────────────────────────────────────
  Метрика                                P50    P95    P99   mean
  ──────────────────────────────────────────────────
  llm_request_latency (полный)        3089ms 9040ms 9040ms 4492ms
  retrieval_latency (только БД)        482ms  613ms  613ms  507ms
...
```

```python
# ── ЯЧЕЙКА 3 ──────────────────────────────────────────────────────────────
# ⏱ ~3 мин | 📌 7.4 + 7.5: LLM-as-a-Judge + финальный production-дашборд
# Qwen3 оценивает качество собственного ответа из Ячейки 1
# Демонстрируем: Faithfulness, Answer Relevance, Tool Accuracy

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# ─── СХЕМА ОЦЕНКИ ────────────────────────────────────────────────────────────

class RAGQualityScore(BaseModel):
    faithfulness: float = Field(
        ge=0.0, le=1.0,
        description="Доля утверждений в ответе, подтверждённых контекстом"
    )
    answer_relevance: float = Field(
        ge=0.0, le=1.0,
        description="Насколько ответ релевантен вопросу"
    )
    reasoning: str = Field(
        description="Краткое обоснование оценок в 2-3 предложениях"
    )

# LLM-судья с reasoning=True — судья должен думать тщательно
# reasoning=True изолирует <think>-блок от JSON-парсинга
judge_llm    = ChatOllama(model="qwen3:8b", reasoning=True)
judge_chain  = judge_llm.with_structured_output(RAGQualityScore)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты строгий аудитор качества RAG-систем. "
     "Оцени качество ответа по двум метрикам:\n\n"
     "1. FAITHFULNESS (0.0–1.0): доля утверждений в ОТВЕТЕ, "
     "подтверждённых КОНТЕКСТОМ. "
     "Утверждение без опоры на контекст = галлюцинация → снижает score.\n\n"
     "2. ANSWER_RELEVANCE (0.0–1.0): насколько ответ отвечает на ВОПРОС. "
     "Ответ не по теме → низкий score.\n\n"
     "Будь строгим. Верни JSON с числами 0.0–1.0."),
    ("human",
     "ВОПРОС:\n{question}\n\n"
     "КОНТЕКСТ (retrieved чанки):\n{context}\n\n"
     "ОТВЕТ LLM:\n{answer}\n\n"
     "Оцени Faithfulness и Answer Relevance.")
])

# ─── ЗАПУСКАЕМ ОЦЕНКУ ────────────────────────────────────────────────────────

print("⚖️  LLM-as-a-Judge: Qwen3 оценивает ответ из Ячейки 1...\n")

# rag_answer, QUERY, context, retrieved_docs — из Ячейки 1
try:
    score: RAGQualityScore = judge_chain.invoke(
        judge_prompt.format_messages(
            question=QUERY,
            context=context[:2000],   # обрезаем чтобы не перегружать контекст
            answer=rag_answer
        )
    )
    judge_ok = True
except Exception as e:
    print(f"⚠️ Судья вернул ошибку: {e}")
    # Fallback — нейтральные значения
    score = RAGQualityScore(
        faithfulness=0.5,
        answer_relevance=0.5,
        reasoning="Оценка недоступна (ошибка парсинга)"
    )
    judge_ok = False

# ─── ФИНАЛЬНЫЙ PRODUCTION-ДАШБОРД ────────────────────────────────────────────

def score_bar(v: float, width: int = 20) -> str:
    filled = int(v * width)
    color  = "🟢" if v >= 0.8 else ("🟡" if v >= 0.6 else "🔴")
    return f"{color} [{'█'*filled}{'░'*(width-filled)}] {v:.2f}"

def score_verdict(v: float) -> str:
    if v >= 0.8: return "✅ Хорошо"
    if v >= 0.6: return "⚠️  Приемлемо"
    return "❌ Требует доработки"

print(f"\n{'╔'+'═'*70+'╗'}")
print(f"║{'  🎯 PRODUCTION MONITORING DASHBOARD':^70}║")
print(f"{'╚'+'═'*70+'╝'}")

# ── Блок 1: Trace (из Ячейки 1) ──────────────────────────────────────────────
total_ms = sum(s.duration_ms for s in tracer.spans)
ret_ms   = next((s.duration_ms for s in tracer.spans
                 if "retrieval" in s.name), 0)
llm_ms   = next((s.duration_ms for s in tracer.spans
                 if "llm" in s.name), 0)

print(f"\n📍 TRACE  —  {tracer.trace_name}")
print(f"{'─'*72}")
print(f"  Запрос              : {QUERY[:60]}")
print(f"  Total latency       : {total_ms:.0f} ms")
print(f"  ├── Ретривал        : {ret_ms:.0f} ms  ({ret_ms/total_ms*100:.0f}%)")
print(f"  └── LLM генерация   : {llm_ms:.0f} ms  ({llm_ms/total_ms*100:.0f}%)")
print(f"  Чанков найдено      : {len(retrieved_docs)}")
print(f"  Длина ответа        : {len(rag_answer)} символов")

# ── Блок 2: Prometheus Metrics (из Ячейки 2) ─────────────────────────────────
print(f"\n📊 METRICS  —  /metrics (Prometheus)")
print(f"{'─'*72}")
print(f"  requests_total           : {requests_total.value:.0f}")
print(f"  errors_total             : {errors_total.value:.0f}  "
      f"(error rate: {errors_total.value/requests_total.value*100:.1f}%)")
print(f"  tokens_generated_total   : {tokens_generated.value:.0f}")
print(f"  latency P50 / P95 / P99  : "
      f"{request_latency.percentile(50)*1000:.0f} ms / "
      f"{request_latency.percentile(95)*1000:.0f} ms / "
      f"{request_latency.percentile(99)*1000:.0f} ms")

# ── Блок 3: LLM-as-a-Judge Scores (из текущей ячейки) ────────────────────────
print(f"\n⚖️  LLM QUALITY SCORES  —  Qwen3 as Judge")
print(f"{'─'*72}")
print(f"  Faithfulness      : {score_bar(score.faithfulness)}  "
      f"{score_verdict(score.faithfulness)}")
print(f"  Answer Relevance  : {score_bar(score.answer_relevance)}  "
      f"{score_verdict(score.answer_relevance)}")
print(f"\n  Обоснование судьи : {score.reasoning}")

# ── Блок 4: Диагностика — что делать если метрики плохие ─────────────────────
print(f"\n🔧 ДИАГНОСТИКА (автоматические рекомендации)")
print(f"{'─'*72}")

if score.faithfulness < 0.7:
    print(f"  ❌ Faithfulness низкий → галлюцинации в генераторе")
    print(f"     Действие: ужесточить системный промпт, включить reasoning=True")

if score.answer_relevance < 0.7:
    print(f"  ❌ Answer Relevance низкий → ответ не по теме")
    print(f"     Действие: проверить ретривал, улучшить чанкинг")

if ret_ms > 500:
    print(f"  ⚠️  Ретривал > 500ms → узкое место в поиске")
    print(f"     Действие: HNSW-индекс в Qdrant, кэш частых запросов")

if request_latency.percentile(99) > 5.0:
    print(f"  ⚠️  P99 > 5s → длинный хвост latency")
    print(f"     Действие: streaming ответов, async обработка")

if score.faithfulness >= 0.8 and score.answer_relevance >= 0.8:
    print(f"  ✅ Все метрики в норме — система готова к production")

# ── Блок 5: Инфраструктура ───────────────────────────────────────────────────
print(f"\n🏗️  PRODUCTION STACK  —  текущая конфигурация")
print(f"{'─'*72}")
print(f"  Inference    : Ollama (dev) → vLLM + T-lite-dpo (prod, МК1)")
print(f"  Vector DB    : Qdrant local mode → Qdrant server (prod)")
print(f"  Tracing      : LocalTracer → Langfuse self-hosted (FSL лицензия)")
print(f"  Metrics      : LocalMetrics → Prometheus + Grafana (prod)")
print(f"  OTel Export  : OTLP → Langfuse / Jaeger / Datadog (без смены кода)")

print(f"\n{'╔'+'═'*70+'╗'}")
print(f"║{'  ✅ МАСТЕР-КЛАСС ЗАВЕРШЁН':^70}║")
print(f"║{'  МК1 (Fine-tuning) → МК2 (LangChain Enterprise) → МК3 (LLMOps)':^70}║")
print(f"{'╚'+'═'*70+'╝'}")
```

```
...
📊 METRICS  —  /metrics (Prometheus)
────────────────────────────────────────────────────────────────────────
  requests_total           : 5
  errors_total             : 0  (error rate: 0.0%)
  tokens_generated_total   : 201
  latency P50 / P95 / P99  : 3089 ms / 9040 ms / 9040 ms

⚖️  LLM QUALITY SCORES  —  Qwen3 as Judge
────────────────────────────────────────────────────────────────────────
  Faithfulness      : 🟢 [████████████████████] 1.00  ✅ Хорошо
  Answer Relevance  : 🟢 [████████████████████] 1.00  ✅ Хорошо
...
```

---

**Финальная архитектура (то что мы построили):**

```
Пользователь
     ↓
Gradio UI (порт 7860)
     ↓
LangGraph Agent
  ├── Qwen3:8b Ollama (порт 11434) ← разработка
  │   или
  ├── T-lite-dpo vLLM (порт 8001) ← production (из МК1)
  │
  ├── RAG Tool
  │    ├── Qdrant local mode (/workspace/qdrant_data)
  │    ├── BGE-M3 embeddings (CPU)
  │    └── BM25 + Hybrid (EnsembleRetriever)
  │
  ├── Knowledge Graph (NetworkX)
  │    └── LLMGraphTransformer
  │
  └── Custom Tools (get_employee_info, calculate_cost...)
           ↓
     WandB Weave (трассировка)
     или Langfuse (self-hosted)
```

---
