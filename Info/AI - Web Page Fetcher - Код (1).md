
## Проверка через curl из терминала:

```bash
curl -s -X POST 'https://llm.api.cloud.yandex.net/v1/chat/completions' \
  -H "Authorization: Api-Key <ВАШ_YANDEX_API_КЛЮЧ>" \
  -H "x-folder-id: <ВАШ_YANDEX_FOLDER_ID>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt://b1g8l25elp2gclf7dlai/yandexgpt/latest",
    "messages": [
      {"role": "user", "content": "Кто ты и что умеешь?"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }' | jq .


curl -s -X POST 'https://llm.api.cloud.yandex.net/v1/chat/completions' \
  -H "Authorization: Api-Key <НОВЫЙ_КЛЮЧ>" \
  -H "x-folder-id: b1g8l25elp2gclf7dlai" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt://b1g8l25elp2gclf7dlai/yandexgpt/latest",
    "messages": [
      {"role": "user", "content": "Кто ты и что умеешь?"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }' | jq .

```

---
## Структура проекта

```
/home/$USER/projects/module-01-scrapping/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                  ← из .env.example, не в git
├── .env.example
├── .gitignore
├── notebooks/            ← volume → /workspace/notebooks
│   └── notebook_01_scraping.ipynb
└── data/
    └── raw/              ← HTML + TXT результаты (volume → /workspace/data)
```

```bash
cat << 'EOF' > create_structure.sh
#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/$USER/projects/module-01-scrapping"

# Каталоги
mkdir -pv "$PROJECT/notebooks"
mkdir -pv "$PROJECT/data/raw"

# Файлы
touch "$PROJECT/Dockerfile"
touch "$PROJECT/docker-compose.yml"
touch "$PROJECT/requirements.txt"
touch "$PROJECT/.env.example"
cp -n "$PROJECT/.env.example" "$PROJECT/.env"
touch "$PROJECT/.gitignore"
touch "$PROJECT/notebooks/notebook_01_scraping.ipynb"

echo "Структура создана:"
find "$PROJECT" | sort

cd "$PROJECT"
EOF

bash create_structure.sh


```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

LABEL project="module-01-scrapping"

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Копируем requirements отдельным слоем — кэш при ребилде
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаём директории (volume перекроет, но mkdir нужен для первого старта)
RUN mkdir -p /workspace/notebooks /workspace/data/raw

# JupyterLab конфиг: без токена, без браузера, слушаем всех
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'"            >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888"               >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False"      >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''"               >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''"            >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'"       >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.root_dir = '/workspace'"  >> /root/.jupyter/jupyter_lab_config.py

EXPOSE 8888

# --allow-root нужен т.к. на Kali работаем под root
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
```

---

## docker-compose.yml

```yaml
cat << 'EOF' > docker-compose.yml
services:
  jupyter:
    build: .
    container_name: module01_jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/workspace/notebooks
      - ./data:/workspace/data
      # .env монтируем явно — load_dotenv("/workspace/.env") найдёт его гарантированно
      - ./.env:/workspace/.env:ro
    env_file:
      - .env
    extra_hosts:
      # FIX 1.1: на Linux host.docker.internal не работает без этой строки
      # host-gateway = IP хостовой машины изнутри контейнера
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
EOF
```

---

```bash
# для обычной железной Debian 12 

sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker

docker --version
docker compose version
```

---

## requirements.txt

```
cat << 'EOF' > requirements.txt
jupyterlab==4.2.5
httpx==0.27.2
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
python-dotenv==1.0.1
langchain==0.3.7
langchain-openai==0.2.9
ipywidgets==8.1.5
EOF
```

---

## .env

```bash
cat << 'EOF' > .env
# Выбор провайдера: yandex | deepseek | qwen3
PROVIDER=yandex

# YandexGPT
YANDEX_API_KEY=AQVNzg...
YANDEX_FOLDER_ID=b1g8l25elp2gclf7dlai

# DeepSeek (если PROVIDER=deepseek)
DEEPSEEK_API_KEY=

# Qwen3 через Ollama на хосте (если PROVIDER=qwen3) — менять не нужно
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
EOF
```

---

## .gitignore

```
cat << 'EOF' > .gitignore
.env
data/
__pycache__/
*.pyc
.DS_Store
*.ipynb_checkpoints
EOF
```

---


```bash
# Собрать образ и запустить контейнер
docker compose up --build -d

# Смотреть логи (убедиться что Jupyter стартовал)
docker compose logs -f
```

Jupyter готов когда в логах увидим:

```
http://0.0.0.0:8888/lab
```

Тогда открываем браузер: http://localhost:8888


**Если порт 8888 занят:**
```bash
lsof -i :8888    # смотрим кто занял
```

**Остановить:**
```bash
docker compose down
```

**Пересобрать после изменений в Dockerfile/requirements:**
```bash
docker compose down && docker compose up --build -d
```


---

## ЯЧЕЙКА 0 — Проверка API

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

## ЯЧЕЙКА 0 — Тесты

```python
# =============================================================================
# ЯЧЕЙКА 0.1: Функциональные тесты для фабрики get_llm
# Запускаются локально внутри ноутбука. Используется встроенный unittest.
# Изоляция (Mock) переменных окружения гарантирует отсутствие сайд-эффектов.
# =============================================================================
import os
import unittest
from unittest.mock import patch
from langchain_openai import ChatOpenAI

# Примечание: предполагается, что Ячейка 1 уже выполнена,
# и функция get_llm доступна в глобальной области видимости.


class TestGetLLM(unittest.TestCase):
    
    # --- YandexGPT ---
    
    def test_yandex_valid_env_returns_configured_instance(self):
        # Arrange: Подменяем os.environ. clear=True удаляет все реальные ключи для чистоты теста.
        env_mock = {
            "PROVIDER": "yandex",
            "YANDEX_API_KEY": "test_yandex_key",
            "YANDEX_FOLDER_ID": "test_folder_123"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertEqual(llm.model_name, "gpt://test_folder_123/yandexgpt/latest")
            self.assertEqual(llm.temperature, 0.7)

    def test_yandex_missing_api_key_raises_value_error(self):
        # Arrange: Имитируем отсутствие YANDEX_API_KEY
        env_mock = {
            "PROVIDER": "yandex",
            "YANDEX_FOLDER_ID": "test_folder_123"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act & Assert
            with self.assertRaisesRegex(ValueError, "Не заданы YANDEX_API_KEY или YANDEX_FOLDER_ID"):
                get_llm()

    def test_yandex_missing_folder_id_raises_value_error(self):
        # Arrange: Имитируем отсутствие YANDEX_FOLDER_ID
        env_mock = {
            "PROVIDER": "yandex",
            "YANDEX_API_KEY": "test_yandex_key"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act & Assert
            with self.assertRaisesRegex(ValueError, "Не заданы YANDEX_API_KEY или YANDEX_FOLDER_ID"):
                get_llm()
                
    def test_yandex_sets_required_custom_headers(self):
        # Arrange: Яндексу нужен кастомный заголовок x-folder-id
        env_mock = {
            "PROVIDER": "yandex",
            "YANDEX_API_KEY": "test_key",
            "YANDEX_FOLDER_ID": "target_folder_999"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertTrue(hasattr(llm, "default_headers"))
            self.assertEqual(llm.default_headers.get("x-folder-id"), "target_folder_999")

    # --- DeepSeek ---
            
    def test_deepseek_valid_env_returns_configured_instance(self):
        # Arrange
        env_mock = {
            "PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test_deepseek_key"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertEqual(llm.model_name, "deepseek-chat")
            self.assertEqual(llm.temperature, 0.7)

    def test_deepseek_missing_api_key_raises_value_error(self):
        # Arrange
        env_mock = {"PROVIDER": "deepseek"}
        with patch.dict(os.environ, env_mock, clear=True):
            # Act & Assert
            with self.assertRaisesRegex(ValueError, "Не задан DEEPSEEK_API_KEY"):
                get_llm()

    # --- Qwen3 (Ollama) ---

    def test_qwen3_valid_env_returns_configured_instance(self):
        # Arrange: Qwen3 не требует внешних ключей, мокаем только PROVIDER
        env_mock = {"PROVIDER": "qwen3"}
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertEqual(llm.model_name, "qwen3:8b")
            # Проверяем, что используется локальный docker base_url
            actual_base_url = getattr(llm, "openai_api_base", "")
            self.assertEqual(actual_base_url, "http://host.docker.internal:11434/v1")

    # --- Общее поведение функции ---

    def test_default_provider_returns_yandex_instance(self):
        # Arrange: Если PROVIDER не задан вообще, фоллбэк на Yandex
        env_mock = {
            "YANDEX_API_KEY": "test_yandex_key",
            "YANDEX_FOLDER_ID": "test_folder_123"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertIn("yandexgpt", llm.model_name)

    def test_provider_case_insensitive_returns_configured_instance(self):
        # Arrange: Тестируем смешанный регистр провайдера (например, DeepSeek)
        env_mock = {
            "PROVIDER": "DeepSeek",
            "DEEPSEEK_API_KEY": "test_deepseek_key"
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Act
            llm = get_llm()
            
            # Assert
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertEqual(llm.model_name, "deepseek-chat")

    def test_unknown_provider_raises_value_error(self):
        # Arrange: Задан неизвестный провайдер
        env_mock = {"PROVIDER": "aws_bedrock"}
        with patch.dict(os.environ, env_mock, clear=True):
            # Act & Assert
            with self.assertRaisesRegex(ValueError, "Неизвестный PROVIDER='aws_bedrock'"):
                get_llm()


# Запуск тестов внутри Jupyter-ячейки
if __name__ == '__main__':
    print("⏳ Запуск тестов для функции get_llm()...\n")
    # exit=False обязателен, чтобы ядро Jupyter не падало после прогона
    result = unittest.main(argv=[''], exit=False)
    
    if result.result.wasSuccessful():
        print("\n✅ Все 10 тестов успешно пройдены! Функция get_llm работает корректно.")
    else:
        print("\n❌ Есть упавшие тесты. Проверьте логи выше.")
```

---

## ЯЧЕЙКА 1 — URL → Сырой HTML

```python
# =============================================================================
# ЯЧЕЙКА 1: Ввод URL → HTTP fetch → сохранение raw .html в volume
# =============================================================================
import requests
import time
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display

RAW_HTML_DIR = Path("/workspace/data/raw")
RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 10 МБ

# FIX 3.4: кортеж (connect_timeout, read_timeout)
# 5 сек на TCP+TLS handshake, 20 сек на чтение данных
REQUEST_TIMEOUT = (5.0, 20.0)


def validate_url(url: str) -> str:
    """
    Проверяет схему и наличие хоста.
    Raises ValueError для невалидного URL.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL не может быть пустым")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Недопустимая схема: '{parsed.scheme}'. Ожидается http или https")
    if not parsed.netloc:
        raise ValueError("URL не содержит хоста")

    return url


def fetch_html(url: str) -> tuple[str, dict]:
    """
    Загружает HTML страницы по URL через stream=True.
    Проверяет размер по заголовку Content-Length И при чтении чанков.

    Returns:
        (html: str, meta: dict)
    Raises:
        ValueError             — невалидный URL, не HTML, превышен размер
        requests.HTTPError     — HTTP 4xx / 5xx
        requests.Timeout       — превышен таймаут
        requests.RequestException — прочие сетевые ошибки
    """
    url = validate_url(url)
    logger.info(f"GET {url}")

    start = time.monotonic()

    # FIX 1.2: stream=True — сервер не сдампит гигабайт в RAM до проверки
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,  # FIX 3.4: (connect, read)
        stream=True,
    )
    response.raise_for_status()

    # Проверка content-type до скачивания тела
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Ожидается text/html, получен: '{content_type}'")

    # Быстрая проверка по заголовку — если сервер его отдал
    raw_cl = response.headers.get("Content-Length")
    if raw_cl is not None:
        content_length = int(raw_cl)
        if content_length > MAX_CONTENT_BYTES:
            raise ValueError(
                f"Файл слишком большой по Content-Length: {content_length:,} байт "
                f"(лимит {MAX_CONTENT_BYTES // 1024 // 1024} МБ)"
            )

    # Читаем чанками — останавливаемся при превышении лимита
    # Защита от сервера без Content-Length или с ложным заголовком
    chunks: list[bytes] = []
    total_bytes = 0

    for chunk in response.iter_content(chunk_size=8192):
        total_bytes += len(chunk)
        if total_bytes > MAX_CONTENT_BYTES:
            raise ValueError(
                f"Превышен лимит {MAX_CONTENT_BYTES // 1024 // 1024} МБ при скачивании "
                f"(прочитано {total_bytes:,} байт)"
            )
        chunks.append(chunk)

    content = b"".join(chunks)
    elapsed = time.monotonic() - start

    # Декодируем: берём кодировку из заголовка, fallback utf-8
    encoding = response.encoding or "utf-8"
    html = content.decode(encoding, errors="replace")

    meta = {
        "url":          url,
        "status":       response.status_code,
        "size_bytes":   total_bytes,
        "elapsed_sec":  round(elapsed, 2),
        "content_type": content_type,
        "scraped_at":   datetime.now().isoformat(),
    }

    logger.info(f"Получено {total_bytes:,} байт за {elapsed:.2f}с")
    return html, meta


def save_raw_html(html: str, url: str, output_dir: Path) -> Path:
    """Сохраняет сырой HTML. Имя файла: hostname_YYYYMMDD_HHMMSS.html"""
    hostname  = urlparse(url).netloc.replace(".", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"{hostname}_{timestamp}.html"
    filepath.write_text(html, encoding="utf-8")
    logger.info(f"Сохранён: {filepath}")
    return filepath


# ─── UI ──────────────────────────────────────────────────────────────────────

url_input = widgets.Text(
    value="",
    placeholder="https://example.com",
    description="URL:",
    layout=widgets.Layout(width="600px"),
)
run_button = widgets.Button(
    description="▶ Скрапить",
    button_style="primary",
    layout=widgets.Layout(width="140px"),
)
output_area = widgets.Output()

# Передаём результат в ячейку 3 через модульный уровень
saved_html_path:    Path | None = None
saved_html_content: str  | None = None


def on_scrape_click(_button) -> None:
    """Validate → fetch → save → отчёт пользователю."""
    global saved_html_path, saved_html_content

    output_area.clear_output()
    with output_area:
        try:
            html, meta = fetch_html(url_input.value)
            filepath   = save_raw_html(html, meta["url"], RAW_HTML_DIR)

            saved_html_path    = filepath
            saved_html_content = html

            print(f"✅ HTML сохранён")
            print(f"   URL    : {meta['url']}")
            print(f"   Статус : {meta['status']}")
            print(f"   Размер : {meta['size_bytes']:,} байт")
            print(f"   Время  : {meta['elapsed_sec']} сек")
            print(f"   Файл   : {filepath}")

        except ValueError                as e: print(f"❌ Ошибка: {e}")
        except requests.HTTPError        as e: print(f"❌ HTTP {e.response.status_code}: {url_input.value}")
        except requests.Timeout:               print(f"❌ Timeout: {url_input.value}")
        except requests.RequestException as e: print(f"❌ Сеть: {e}")


run_button.on_click(on_scrape_click)
display(widgets.VBox([url_input, run_button, output_area]))
```

---

```python
# =============================================================================
# ЯЧЕЙКА 1.1: Функциональные тесты — с подробным выводом по каждому тесту
# =============================================================================
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import requests

# ─── Описания тестов (ключ = имя метода) ─────────────────────────────────────

TEST_DESCRIPTIONS = {
    "test_validate_url_valid_http_and_https_returns_url":
        "validate_url: корректные http:// и https:// URL принимаются",
    "test_validate_url_empty_string_raises_value_error":
        "validate_url: пустая строка → ValueError",
    "test_validate_url_invalid_scheme_raises_value_error":
        "validate_url: схема ftp:// → ValueError",
    "test_validate_url_missing_host_raises_value_error":
        "validate_url: URL без хоста → ValueError",
    "test_fetch_html_valid_response_returns_html_and_meta":
        "fetch_html: 200 OK → возвращает html + meta (url, status, size_bytes, scraped_at)",
    "test_fetch_html_invalid_content_type_raises_value_error":
        "fetch_html: Content-Type: application/json → ValueError",
    "test_fetch_html_huge_content_length_header_raises_value_error":
        "fetch_html: Content-Length > 10 МБ в заголовке → ValueError",
    "test_fetch_html_huge_stream_raises_value_error":
        "fetch_html: поток > 10 МБ при чтении чанков → ValueError",
    "test_fetch_html_http_error_raises_exception":
        "fetch_html: HTTP 403 → requests.HTTPError пробрасывается",
    "test_save_raw_html_creates_correct_file_and_writes_data":
        "save_raw_html: файл создаётся с правильным именем и содержимым",
}

# ─── Кастомный Result — перехватывает каждый тест ────────────────────────────

class VerboseResult(unittest.TestResult):
    """
    Печатает ✅/❌ + описание после каждого теста.
    Накапливает итоговую статистику.
    """

    def __init__(self):
        super().__init__()
        self._results: list[tuple[str, bool, str]] = []  # (method, ok, detail)

    def _label(self, test: unittest.TestCase) -> str:
        return TEST_DESCRIPTIONS.get(test._testMethodName, test._testMethodName)

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._results.append((self._label(test), True, ""))

    def addFailure(self, test, err) -> None:
        super().addFailure(test, err)
        # Берём только последнюю строку трейсбека — кратко
        detail = str(err[1]).splitlines()[-1] if err[1] else ""
        self._results.append((self._label(test), False, detail))

    def addError(self, test, err) -> None:
        super().addError(test, err)
        detail = str(err[1]).splitlines()[-1] if err[1] else ""
        self._results.append((self._label(test), False, f"ERROR: {detail}"))

    def print_report(self) -> None:
        """Финальный отчёт: каждый тест на отдельной строке."""
        passed = sum(1 for _, ok, _ in self._results if ok)
        total  = len(self._results)

        print(f"\n{'─' * 62}")
        print(f"  Результаты тестов ({passed}/{total})")
        print(f"{'─' * 62}")

        for label, ok, detail in self._results:
            icon = "✅" if ok else "❌"
            print(f"  {icon}  {label}")
            if detail:
                # Показываем суть ошибки с отступом
                print(f"       └─ {detail}")

        print(f"{'─' * 62}")
        if passed == total:
            print(f"  ✅ Все {total} тестов прошли успешно")
        else:
            print(f"  ❌ Провалено: {total - passed} из {total}")
        print(f"{'─' * 62}\n")


# ─── Тесты ───────────────────────────────────────────────────────────────────

class TestScrapingCell(unittest.TestCase):

    # ── validate_url ─────────────────────────────────────────────────────────

    def test_validate_url_valid_http_and_https_returns_url(self):
        self.assertEqual(validate_url("http://example.com"), "http://example.com")
        self.assertEqual(
            validate_url("https://example.com/path?args=1"),
            "https://example.com/path?args=1",
        )

    def test_validate_url_empty_string_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "URL не может быть пустым"):
            validate_url("   \n ")

    def test_validate_url_invalid_scheme_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Недопустимая схема: 'ftp'"):
            validate_url("ftp://files.example.com")

    def test_validate_url_missing_host_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "URL не содержит хоста"):
            validate_url("https:///path/only")

    # ── fetch_html ───────────────────────────────────────────────────────────

    @patch('requests.get')
    def test_fetch_html_valid_response_returns_html_and_meta(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers     = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.encoding    = "utf-8"
        mock_response.iter_content.return_value = [b"<html>", b"<body>Hi</body></html>"]
        mock_get.return_value = mock_response

        html, meta = fetch_html("https://example.com")

        self.assertEqual(html, "<html><body>Hi</body></html>")
        self.assertEqual(meta["url"],        "https://example.com")
        self.assertEqual(meta["status"],     200)
        self.assertEqual(meta["size_bytes"], 28)
        self.assertIn("text/html",   meta["content_type"])
        self.assertIn("scraped_at",  meta)
        self.assertTrue(mock_get.call_args.kwargs.get("stream"))
        self.assertIn("timeout", mock_get.call_args.kwargs)

    @patch('requests.get')
    def test_fetch_html_invalid_content_type_raises_value_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_get.return_value = mock_response
        with self.assertRaisesRegex(ValueError, "Ожидается text/html"):
            fetch_html("https://api.example.com/data")

    @patch('requests.get')
    def test_fetch_html_huge_content_length_header_raises_value_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type":   "text/html",
            "Content-Length": str(11 * 1024 * 1024),
        }
        mock_get.return_value = mock_response
        with self.assertRaisesRegex(ValueError, "Файл слишком большой по Content-Length"):
            fetch_html("https://example.com/huge")

    @patch('requests.get')
    def test_fetch_html_huge_stream_raises_value_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.iter_content.return_value = [b"A" * (10 * 1024 * 1024 + 100)]
        mock_get.return_value = mock_response
        with self.assertRaisesRegex(ValueError, "Превышен лимит 10 МБ при скачивании"):
            fetch_html("https://example.com/infinite-stream")

    @patch('requests.get')
    def test_fetch_html_http_error_raises_exception(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response
        with self.assertRaises(requests.HTTPError):
            fetch_html("https://example.com/protected")

    # ── save_raw_html ────────────────────────────────────────────────────────

    @patch('__main__.datetime')
    def test_save_raw_html_creates_correct_file_and_writes_data(self, mock_dt):
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20260911_120000"
        mock_dt.now.return_value = mock_now

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path    = Path(tmp)
            result_path = save_raw_html("<h1>Test</h1>", "https://sub.domain.com:8080/page", tmp_path)
            expected    = tmp_path / "sub_domain_com_8080_20260911_120000.html"

            self.assertEqual(result_path, expected)
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.read_text(encoding="utf-8"), "<h1>Test</h1>")


# ─── Запуск ──────────────────────────────────────────────────────────────────

print("⏳ Запуск тестов...\n")

suite  = unittest.TestLoader().loadTestsFromTestCase(TestScrapingCell)
result = VerboseResult()
suite.run(result)
result.print_report()
```

---

## ЯЧЕЙКА 2 — BeautifulSoup → Чистый текст

```python
# =============================================================================
# ЯЧЕЙКА 2: Сырой HTML → структурированный текст → .txt
# =============================================================================
import re
from bs4 import BeautifulSoup, Tag
from pathlib import Path

# Теги удаляем полностью вместе с содержимым
REMOVE_TAGS = frozenset({
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "noscript",
    "svg", "img", "picture", "video", "audio",
})

# Куки-баннеры и попапы по class/id
_NOISE_PATTERN = re.compile(
    r"cookie|gdpr|consent|banner|popup|overlay|modal|advert",
    re.IGNORECASE,
)

# Заголовки → уровень вложенности
HEADING_PREFIX = {
    "h1": "## ",
    "h2": "### ",
    "h3": "#### ",
    "h4": "#### ",
    "h5": "##### ",
    "h6": "##### ",
}

# FIX 1.4: расширенный список тегов с текстовым контентом
# find_all с явным списком возвращает каждый элемент ровно один раз
# без проблемы "parent div включает текст всех детей"
TEXT_TAGS = frozenset({"p", "li", "blockquote", "td", "figcaption"})

# Минимальная длина — фильтруем мусорные обрывки
MIN_PARAGRAPH_LEN = 25


def remove_noise(soup: BeautifulSoup) -> None:
    """
    Удаляет шумные теги in-place:
    — по имени (REMOVE_TAGS)
    — по class/id (куки, попапы, баннеры)
    """
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", []))
        tag_id  = tag.get("id", "")
        if _NOISE_PATTERN.search(classes) or _NOISE_PATTERN.search(tag_id):
            tag.decompose()


def extract_structured_text(soup: BeautifulSoup) -> list[str]:
    """
    Извлекает текст в порядке появления в документе.

    Заголовки → с маркерами ## / ### / ####
    Текст     → из p, li, blockquote, td, figcaption

    FIX 1.3: last_line — дедупликация только ПОДРЯД идущих одинаковых строк.
             «### Выводы» в конце каждой главы — все сохраняются.

    FIX 1.4: find_all([теги]) вместо итерации по descendants.
             Каждый элемент возвращается ровно один раз → нет дублей.
    """
    body  = soup.body or soup
    lines: list[str] = []

    # last_line — только для подряд идущих дублей (пустые строки между секциями)
    last_line: str | None = None

    # Собираем все нужные теги одним проходом по документу
    all_target_tags = list(HEADING_PREFIX.keys()) + list(TEXT_TAGS)

    for element in body.find_all(all_target_tags):
        tag_name = element.name.lower()

        if tag_name in HEADING_PREFIX:
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue
            line = f"\n{HEADING_PREFIX[tag_name]}{text}\n"

        elif tag_name in TEXT_TAGS:
            text = element.get_text(separator=" ", strip=True)
            if len(text) < MIN_PARAGRAPH_LEN:
                continue
            line = text

        else:
            continue

        # Дедуплицируем только ПОДРЯД идущие одинаковые строки
        if line != last_line:
            lines.append(line)
            last_line = line

    return lines


def normalize_whitespace(text: str) -> str:
    """Схлопывает 3+ пустых строки в 2. Убирает trailing пробелы."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def clean_html_to_text(html: str) -> str:
    """Pipeline: HTML → BS4 → remove_noise → extract → join → normalize."""
    soup = BeautifulSoup(html, "lxml")
    remove_noise(soup)
    lines = extract_structured_text(soup)
    return normalize_whitespace("\n".join(lines))


def save_clean_text(text: str, html_path: Path) -> Path:
    """Сохраняет .txt с тем же именем что и .html."""
    txt_path = html_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    logger.info(f"Сохранён: {txt_path}")
    return txt_path


# ─── Запуск ──────────────────────────────────────────────────────────────────

if saved_html_content is None or saved_html_path is None:
    print("❌ Сначала выполни Ячейку 2 — сырой HTML не загружен")
else:
    print(f"📄 Обрабатываем: {saved_html_path.name}")

    clean_text = clean_html_to_text(saved_html_content)
    txt_path   = save_clean_text(clean_text, saved_html_path)

    # Считаем заголовки по всем уровням
    heading_count = sum(
        1 for line in clean_text.splitlines()
        if line.startswith(("## ", "### ", "#### ", "##### "))
    )

    print(f"\n✅ Текст очищен и сохранён")
    print(f"   Файл       : {txt_path}")
    print(f"   Символов   : {len(clean_text):,}")
    print(f"   Строк      : {clean_text.count(chr(10)):,}")
    print(f"   Заголовков : {heading_count}")
    print(f"\n{'─' * 50}")
    print(f"Превью (первые 600 символов):\n")
    print(clean_text[:600])
    print("...")
```


---

```python
# =============================================================================
# ЯЧЕЙКА 3 (БЛОК 3): clean_text → ToC (LLM вызов N1) + section .txt файлы
# Стратегия: LLM видит ТОЛЬКО список заголовков (экономия токенов)
#            Python делит полный текст по границам → сохраняет каждый раздел
# =============================================================================
import json
import os
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ─── LLM с увеличенным лимитом токенов для JSON-ответа ───────────────────────
# llm из Ячейки 1 имеет max_tokens=200 — мало для JSON оглавления
# Создаём отдельный instance с нужными параметрами

def _make_llm(max_tokens: int = 2000, temperature: float = 0.3) -> ChatOpenAI:
    """
    Создаёт LLM instance читая провайдера из .env.
    Дублирует логику get_llm() но с настраиваемыми max_tokens/temperature.
    """
    provider  = os.getenv("PROVIDER", "yandex").lower()
    folder_id = os.getenv("YANDEX_FOLDER_ID", "")

    # Конфиги провайдеров — только специфичные параметры
    provider_configs = {
        "yandex": dict(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=os.getenv("YANDEX_API_KEY", ""),
            model=f"gpt://{folder_id}/yandexgpt/latest",
            default_headers={"x-folder-id": folder_id},
        ),
        "deepseek": dict(
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model="deepseek-chat",
        ),
        "qwen3": dict(
            base_url="http://host.docker.internal:11434/v1",
            api_key="EMPTY",
            model="qwen3:8b",
        ),
    }

    if provider not in provider_configs:
        raise ValueError(f"Неизвестный PROVIDER='{provider}'")

    return ChatOpenAI(
        **provider_configs[provider],
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Создаём один раз — доступен во всей ячейке
llm_toc = _make_llm(max_tokens=2000, temperature=0.3)


# ─── Pydantic схема для .with_structured_output() ────────────────────────────

class TocSection(BaseModel):
    """Один логический раздел страницы."""
    number:        int  = Field(description="Порядковый номер раздела, начиная с 1")
    title:         str  = Field(description="Заголовок раздела на языке страницы")
    slug:          str  = Field(description="Якорная ссылка: только [a-z0-9-], например 'about-company'")
    first_heading: str  = Field(
        description=(
            "ТОЧНЫЙ текст первого заголовка этого раздела из нумерованного списка выше. "
            "Копируй дословно. Пустая строка только для Summary."
        )
    )


class TableOfContents(BaseModel):
    """Оглавление всей страницы."""
    page_title: str            = Field(description="Общий заголовок страницы из контекста заголовков")
    sections:   List[TocSection] = Field(description="5-10 логических разделов, последний — Summary")


# ─── Утилиты: анализ текста ──────────────────────────────────────────────────

def extract_headings_list(text: str) -> list[str]:
    """
    Извлекает текст всех заголовков из clean_text.
    Заголовки помечены маркерами ## / ### / #### из ячейки BeautifulSoup.
    Возвращает только текст без маркеров — для передачи в LLM.
    """
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        # Ищем строки начинающиеся с 2-5 символов #
        if re.match(r'^#{2,5}\s+', stripped):
            heading_text = re.sub(r'^#+\s+', '', stripped)
            headings.append(heading_text)
    return headings


def _normalize(s: str) -> str:
    """Нормализация для нечёткого сравнения: lowercase + схлопываем пробелы."""
    return re.sub(r'\s+', ' ', s.strip().lower())


def find_section_starts(text: str, sections: List[TocSection]) -> dict[int, int]:
    """
    Находит номер строки (line index) начала каждого раздела
    по полю first_heading из LLM-ответа.
    Нечёткое совпадение — регистронезависимо, нормализованные пробелы.

    Returns:
        {section_number: line_index}
    """
    lines   = text.splitlines()
    starts: dict[int, int] = {}

    for section in sections:
        # Summary не имеет first_heading — пропускаем (сохраняется отдельно)
        if not section.first_heading or section.title.lower() == "summary":
            continue

        target = _normalize(section.first_heading)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^#{2,5}\s+', stripped):
                heading_text = _normalize(re.sub(r'^#+\s+', '', stripped))
                if heading_text == target:
                    starts[section.number] = i
                    break

        if section.number not in starts:
            logger.warning(
                f"Раздел {section.number} '{section.title}': "
                f"заголовок '{section.first_heading}' не найден в тексте"
            )

    return starts


def split_text_into_sections(text: str, sections: List[TocSection]) -> dict[int, str]:
    """
    Делит clean_text на куски по границам разделов.
    Оригинальный текст НЕ изменяется — только нарезается.

    Returns:
        {section_number: section_text}
    """
    lines     = text.splitlines(keepends=True)
    starts    = find_section_starts(text, sections)

    # Сортируем разделы по позиции в тексте
    sorted_bounds = sorted(starts.items(), key=lambda x: x[1])

    result: dict[int, str] = {}

    for idx, (sec_num, start_line) in enumerate(sorted_bounds):
        # Конец раздела = начало следующего (или EOF)
        end_line = (
            sorted_bounds[idx + 1][1]
            if idx + 1 < len(sorted_bounds)
            else len(lines)
        )
        result[sec_num] = "".join(lines[start_line:end_line]).strip()

    return result


# ─── LLM вызов N1 ────────────────────────────────────────────────────────────

def generate_toc(headings: list[str], llm: ChatOpenAI) -> TableOfContents:
    """
    LLM вызов N1: список заголовков → структурированное оглавление.

    Стратегия двух попыток:
    1. with_structured_output(TableOfContents) — нативный JSON через tool calling
    2. Fallback: обычный invoke + ручной парсинг JSON из текста ответа
    """
    # Нумерованный список заголовков — минимальный контекст для LLM
    headings_text = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headings))

    prompt = f"""Вот заголовки страницы в порядке появления:
{headings_text}

Сгруппируй их в 5-10 логических разделов.
Верни ТОЛЬКО валидный JSON без markdown-блоков и пояснений:
{{
  "page_title": "...",
  "sections": [
    {{
      "number": 1,
      "title": "...",
      "slug": "slug-только-латиница-дефисы",
      "first_heading": "ТОЧНЫЙ текст первого заголовка раздела из списка выше"
    }}
  ]
}}

Правила:
- number   : порядковый номер начиная с 1
- title    : заголовок раздела на языке страницы
- slug     : только [a-z0-9-], например "technical-requirements"
- first_heading : копируй ДОСЛОВНО из списка выше
- Последний раздел ВСЕГДА добавляй отдельно:
  {{"number": N, "title": "Summary", "slug": "summary", "first_heading": ""}}
- page_title : общий заголовок из контекста всех заголовков"""

    # Попытка 1: with_structured_output (нативный JSON через tool calling)
    try:
        llm_structured = llm.with_structured_output(TableOfContents)
        result         = llm_structured.invoke(prompt)
        logger.info("ToC получен через with_structured_output")
        return result

    except Exception as e:
        logger.warning(f"with_structured_output упал ({type(e).__name__}). Fallback → ручной парсинг JSON")

    # Попытка 2: обычный invoke + парсинг JSON из текстового ответа
    response  = llm.invoke([HumanMessage(content=prompt)])
    raw       = response.content

    # Убираем возможные ```json ... ``` блоки
    raw_clean = re.sub(r'```(?:json)?\s*', '', raw).strip()

    # Ищем JSON объект
    json_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
    if not json_match:
        raise ValueError(
            f"LLM не вернул распознаваемый JSON.\n"
            f"Ответ (первые 400 символов):\n{raw[:400]}"
        )

    data = json.loads(json_match.group())
    logger.info("ToC получен через ручной JSON парсинг")
    return TableOfContents(**data)


# ─── Сохранение ──────────────────────────────────────────────────────────────

def save_toc_files(toc: TableOfContents, output_dir: Path) -> tuple[Path, Path]:
    """
    Сохраняет оглавление в двух форматах:
    - 00_toc.txt  : читаемый список с якорными ссылками
    - 00_toc.json : сырой JSON для следующих ячеек
    """
    # Читаемый формат
    toc_lines = [
        f"# {toc.page_title}",
        "",
        "## Оглавление",
        "",
    ]
    for s in toc.sections:
        toc_lines.append(f"{s.number}. [{s.title}](#{s.slug})")

    toc_txt  = output_dir / "00_toc.txt"
    toc_json = output_dir / "00_toc.json"

    toc_txt.write_text("\n".join(toc_lines), encoding="utf-8")
    # ensure_ascii=False → кириллица читается в файле без экранирования
    toc_json.write_text(toc.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    return toc_txt, toc_json


def save_section_files(
    clean_text: str,
    toc: TableOfContents,
    output_dir: Path,
) -> dict[int, Path]:
    """
    Сохраняет каждый раздел в отдельный файл NN_slug.txt.
    Оригинальный текст НЕ изменяется — только нарезается на части.
    Summary сохраняется как заглушка — заполнится LLM вызовом №2.
    """
    section_chunks = split_text_into_sections(clean_text, toc.sections)
    saved: dict[int, Path] = {}

    for section in toc.sections:
        filename = f"{section.number:02d}_{section.slug}.txt"
        filepath = output_dir / filename

        if section.title.lower() == "summary":
            # Заглушка — будет заполнена в следующей ячейке
            content = (
                f"# Summary\n\n"
                f"[PLACEHOLDER — будет заполнено LLM вызовом №2 в следующей ячейке]\n"
            )
        else:
            content = section_chunks.get(section.number)
            if content is None:
                logger.warning(f"Текст раздела {section.number} '{section.title}' не найден")
                content = f"# {section.title}\n\n[Текст раздела не найден]\n"

        filepath.write_text(content, encoding="utf-8")
        saved[section.number] = filepath
        logger.info(f"Раздел {section.number} сохранён: {filepath.name}")

    return saved


# ─── Основной запуск ─────────────────────────────────────────────────────────

# globals() надёжнее dir() в Jupyter — проверяет реальное наличие переменной
if 'clean_text' not in globals() or clean_text is None:
    print("❌ clean_text не найден — выполни ячейку BeautifulSoup (БЛОК 2) сначала")

elif 'saved_html_path' not in globals() or saved_html_path is None:
    print("❌ saved_html_path не найден — выполни Ячейку 2 (URL → HTML) сначала")

else:
    # Выходная папка: /workspace/data/raw/hostname_timestamp/
    output_dir = saved_html_path.parent / saved_html_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Статистика входных данных
    headings = extract_headings_list(clean_text)

    print(f"📖 Входные данные:")
    print(f"   Символов в тексте  : {len(clean_text):,}")
    print(f"   Найдено заголовков : {len(headings)}")

    if not headings:
        print("\n⚠️  Заголовки не найдены.")
        print("   Возможные причины:")
        print("   1. Ячейка BeautifulSoup не выполнялась в этой сессии")
        print("   2. Страница не содержит тегов <h1>-<h6>")
        print("   3. Все заголовки были в удалённых тегах (nav/header/footer)")

    else:
        print(f"\n🤖 LLM вызов №1: генерируем оглавление...")
        print(f"   Провайдер : {os.getenv('PROVIDER', 'yandex').upper()}")

        toc = generate_toc(headings, llm_toc)

        # Вывод результата
        print(f"\n📋 Страница : {toc.page_title}")
        print(f"   Разделов : {len(toc.sections)}\n")
        for s in toc.sections:
            marker = "📌" if s.title.lower() == "summary" else "  "
            print(f"   {marker} {s.number:2d}. {s.title:<35} #{s.slug}")

        # Сохраняем ToC
        toc_txt, toc_json = save_toc_files(toc, output_dir)
        print(f"\n💾 Оглавление сохранено:")
        print(f"   {toc_txt}")
        print(f"   {toc_json}")

        # Сохраняем разделы
        section_paths = save_section_files(clean_text, toc, output_dir)
        print(f"\n📁 Разделы ({len(section_paths)} файлов) → {output_dir}")
        for num, path in sorted(section_paths.items()):
            size = path.stat().st_size
            print(f"   {path.name:<45} {size:>7,} байт")

        # Передаём переменные в следующие ячейки
        generated_toc       = toc          # TableOfContents object
        sections_output_dir = output_dir   # Path к папке с разделами

        print(f"\n✅ БЛОК 3 завершён")
        print(f"   Доступны : generated_toc, sections_output_dir")

```

---

```python
# =============================================================================
# ЯЧЕЙКА 4.5 (БЛОК 4): Section .txt файлы → Summary (LLM вызов №2)
# Стратегия: читаем каждый раздел последовательно → LLM даёт выжимку раздела
#            → аккумулируем в один Summary файл → перезаписываем placeholder
# Целевой объём Summary: ~10% от объёма полного текста
# =============================================================================
import os
import time
from pathlib import Path


# ─── LLM для Summary ─────────────────────────────────────────────────────────
# Отдельный instance с temperature=0.5 — баланс точности и читаемости

def _make_summary_llm() -> "ChatOpenAI":
    """Создаёт LLM instance для генерации Summary разделов."""
    provider  = os.getenv("PROVIDER", "yandex").lower()
    folder_id = os.getenv("YANDEX_FOLDER_ID", "")

    configs = {
        "yandex": dict(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=os.getenv("YANDEX_API_KEY", ""),
            model=f"gpt://{folder_id}/yandexgpt/latest",
            default_headers={"x-folder-id": folder_id},
        ),
        "deepseek": dict(
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model="deepseek-chat",
        ),
        "qwen3": dict(
            base_url="http://host.docker.internal:11434/v1",
            api_key="EMPTY",
            model="qwen3:8b",
        ),
    }

    if provider not in configs:
        raise ValueError(f"Неизвестный PROVIDER='{provider}'")

    return ChatOpenAI(
        **configs[provider],
        temperature=0.5,
        max_tokens=1000,  # на один раздел хватит, Summary лаконичный
    )


# ─── Промпт генерации выжимки одного раздела ─────────────────────────────────

def _build_section_prompt(section_title: str, section_text: str, target_chars: int) -> str:
    """
    Строит промпт для выжимки одного раздела.
    target_chars — целевая длина выжимки в символах (~10% от раздела).
    """
    return f"""Ты — редактор. Тебе дан текст раздела «{section_title}».

Напиши краткую ёмкую выжимку этого раздела длиной около {target_chars} символов.
Требования:
- Только самое важное — ключевые факты, выводы, данные
- Без воды, без вводных фраз вроде «В этом разделе рассматривается...»
- На том же языке что и исходный текст
- Сплошной текст — без маркеров списка и заголовков
- Не добавляй ничего от себя — только то что есть в тексте

ТЕКСТ РАЗДЕЛА:
{section_text[:4000]}"""  # обрезаем до 4000 символов — защита от огромных разделов


# ─── Обработка одного раздела ────────────────────────────────────────────────

def summarize_section(
    section_title: str,
    section_text:  str,
    target_chars:  int,
    llm:           "ChatOpenAI",
    retries:       int = 2,
) -> str:
    """
    LLM вызов для одного раздела. Возвращает текст выжимки.
    При ошибке — повторяет retries раз с паузой, затем возвращает заглушку.
    """
    prompt = _build_section_prompt(section_title, section_text, target_chars)

    for attempt in range(1, retries + 2):  # +2: первая попытка + retries
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result   = response.content.strip()

            if result:
                return result

            logger.warning(f"  Раздел '{section_title}': LLM вернул пустой ответ (попытка {attempt})")

        except Exception as e:
            logger.warning(f"  Раздел '{section_title}': ошибка LLM попытка {attempt}: {e}")
            if attempt <= retries:
                time.sleep(2 * attempt)  # экспоненциальная пауза: 2с, 4с

    # Все попытки исчерпаны — возвращаем заглушку чтобы не остановить весь pipeline
    return f"[Выжимка недоступна — ошибка LLM при обработке раздела «{section_title}»]"


# ─── Основной pipeline ───────────────────────────────────────────────────────

def generate_summary(
    toc:        "TableOfContents",
    output_dir: Path,
    llm:        "ChatOpenAI",
    full_text:  str,
) -> Path:
    """
    Последовательно обрабатывает каждый раздел через LLM.
    Аккумулирует выжимки в Summary файл.

    Целевой объём Summary: ~10% от длины полного текста.
    Квота на раздел: 10% / N_sections — пропорционально.

    Returns:
        Path к записанному summary .txt файлу
    """
    # Находим Summary секцию из ToC
    summary_section = next(
        (s for s in toc.sections if s.slug == "summary"),
        None
    )
    if summary_section is None:
        raise ValueError("В оглавлении нет раздела Summary. Проверь Ячейку 3 (БЛОК 3).")

    summary_filepath = output_dir / f"{summary_section.number:02d}_{summary_section.slug}.txt"

    # Разделы для обработки — все кроме Summary
    content_sections = [s for s in toc.sections if s.slug != "summary"]

    # Целевые символы: 10% от полного текста / на каждый раздел поровну
    total_chars   = len(full_text)
    target_total  = max(500, total_chars // 10)  # минимум 500 символов
    target_per_sec = max(100, target_total // len(content_sections)) if content_sections else 200

    print(f"   Полный текст    : {total_chars:,} символов")
    print(f"   Цель Summary    : ~{target_total:,} символов (~10%)")
    print(f"   Квота на раздел : ~{target_per_sec:,} символов")
    print(f"   Разделов        : {len(content_sections)}\n")

    # Аккумулятор Summary
    summary_parts: list[str] = [
        f"# Summary: {toc.page_title}",
        f"",
        f"*Автоматически сгенерировано. Источник: {toc.page_title}*",
        f"",
        f"---",
        f"",
    ]

    total_generated = 0

    for idx, section in enumerate(content_sections, start=1):
        filename  = f"{section.number:02d}_{section.slug}.txt"
        filepath  = output_dir / filename

        print(f"  [{idx}/{len(content_sections)}] 📄 {filename}")

        # Читаем текст раздела
        if not filepath.exists():
            logger.warning(f"  Файл не найден: {filepath}")
            section_text = ""
        else:
            section_text = filepath.read_text(encoding="utf-8").strip()

        # Пропускаем пустые разделы
        if not section_text or len(section_text) < 50:
            print(f"          ⚠️  Раздел пустой или слишком короткий — пропускаем")
            continue

        print(f"          Символов в разделе : {len(section_text):,}")
        print(f"          Целевая выжимка    : ~{target_per_sec:,} символов")

        # LLM вызов
        start       = time.monotonic()
        section_sum = summarize_section(section.title, section_text, target_per_sec, llm)
        elapsed     = time.monotonic() - start

        print(f"          Получено           : {len(section_sum):,} символов за {elapsed:.1f}с")

        # Добавляем в аккумулятор
        summary_parts.append(f"## {section.title}")
        summary_parts.append(f"")
        summary_parts.append(section_sum)
        summary_parts.append(f"")
        summary_parts.append(f"---")
        summary_parts.append(f"")

        total_generated += len(section_sum)

    # Финальная статистика в конце файла
    summary_parts.append(
        f"*Итого: {total_generated:,} символов | "
        f"{(total_generated / total_chars * 100):.1f}% от исходного текста*"
    )

    # Записываем — перезаписываем placeholder из Ячейки 3
    full_summary = "\n".join(summary_parts)
    summary_filepath.write_text(full_summary, encoding="utf-8")
    logger.info(f"Summary сохранён: {summary_filepath}")

    return summary_filepath


# ─── Основной запуск ─────────────────────────────────────────────────────────

_required = {
    "generated_toc":       "Ячейка 4 (БЛОК 3 — ToC)",
    "sections_output_dir": "Ячейка 4 (БЛОК 3 — ToC)",
    "clean_text":          "Ячейка 3 (БЛОК 2 — BeautifulSoup)",
}

_missing = [
    f"'{var}' — выполни {cell}"
    for var, cell in _required.items()
    if var not in globals() or globals()[var] is None
]

if _missing:
    for msg in _missing:
        print(f"❌ {msg}")

else:
    print(f"🤖 БЛОК 4: Генерация Summary через LLM")
    print(f"   Провайдер : {os.getenv('PROVIDER', 'yandex').upper()}")
    print(f"   Папка     : {sections_output_dir}\n")

    llm_summary = _make_summary_llm()

    summary_path = generate_summary(
        toc        = generated_toc,
        output_dir = sections_output_dir,
        llm        = llm_summary,
        full_text  = clean_text,
    )

    # Читаем готовый файл для превью
    summary_content = summary_path.read_text(encoding="utf-8")
    size_kb         = summary_path.stat().st_size / 1024

    print(f"\n✅ БЛОК 4 завершён")
    print(f"   Файл        : {summary_path}")
    print(f"   Размер      : {size_kb:.1f} КБ")
    print(f"\n{'─' * 50}")
    print(f"Превью Summary (первые 800 символов):\n")
    print(summary_content[:800])
    print("...\n")
    print(f"💡 Следующий шаг — БЛОК 5 (Ячейка 5): сборка финального HTML")
```

---

```python
# =============================================================================
# ЯЧЕЙКА 5 (БЛОК 5): Сборка финального HTML
# Входные данные: generated_toc, sections_output_dir, clean_text, saved_html_path
# Выход: sections_output_dir/final_page.html
# Jinja2 уже есть в контейнере (зависимость JupyterLab) — pip не нужен
# FIX: toc_sections (все разделы включая Summary) отделён от sections (только тело)
# =============================================================================
from jinja2 import Template
from pathlib import Path
from datetime import datetime
import re


# ─── Конвертер: clean_text формат → HTML фрагмент ────────────────────────────

_HEADING_MAP = {
    "##":    "h2",
    "###":   "h3",
    "####":  "h4",
    "#####": "h5",
}

_SUMMARY_PLACEHOLDER = "[PLACEHOLDER"


def _escape_html(text: str) -> str:
    """Минимальное экранирование — только критичные символы."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_to_html_fragment(text: str) -> str:
    """
    Конвертирует clean_text (с маркерами ## / ### / ####) в HTML фрагмент.
    ## Заголовок  → <h2 id="slug">
    обычная строка → накапливаем в абзац
    пустая строка  → закрываем абзац
    """
    lines:      list[str] = text.splitlines()
    html_parts: list[str] = []
    para_buf:   list[str] = []

    def flush_para() -> None:
        if para_buf:
            content = " ".join(para_buf).strip()
            if content:
                html_parts.append(f"<p>{_escape_html(content)}</p>")
            para_buf.clear()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_para()
            continue

        heading_tag  = None
        heading_text = ""
        for marker, tag in _HEADING_MAP.items():
            if stripped.startswith(marker + " "):
                heading_tag  = tag
                heading_text = stripped[len(marker):].strip()
                break

        if heading_tag:
            flush_para()
            # slug: lowercase, пробелы → дефисы
            slug = re.sub(r'[^\w\s-]', '', heading_text.lower())
            slug = re.sub(r'\s+', '-', slug).strip('-')
            html_parts.append(
                f'<{heading_tag} id="{_escape_html(slug)}">'
                f'{_escape_html(heading_text)}'
                f'</{heading_tag}>'
            )
        else:
            para_buf.append(stripped)

    flush_para()
    return "\n".join(html_parts)


# ─── Чтение section файлов ───────────────────────────────────────────────────

def read_section_files(
    toc:        "TableOfContents",
    output_dir: Path,
) -> dict[int, str]:
    """
    Читает NN_slug.txt файлы с диска.
    Returns: {section_number: raw_text}
    """
    contents: dict[int, str] = {}

    for section in toc.sections:
        filename = f"{section.number:02d}_{section.slug}.txt"
        filepath = output_dir / filename

        if not filepath.exists():
            logger.warning(f"Файл раздела не найден: {filepath}")
            contents[section.number] = f"[Файл {filename} не найден]"
        else:
            contents[section.number] = filepath.read_text(encoding="utf-8")

    return contents


# ─── Jinja2 шаблон ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ page_title }}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; font-size: 16px; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      color: #1a1a1a;
      background: #f7f7f7;
      line-height: 1.7;
    }

    .container { max-width: 860px; margin: 0 auto; padding: 2rem 1.5rem; }

    .page-title {
      font-size: 2rem; font-weight: 700; color: #111;
      margin-bottom: 2rem; padding-bottom: 0.75rem;
      border-bottom: 3px solid #2563eb;
    }

    .toc {
      background: #fff; border: 1px solid #e5e7eb;
      border-left: 4px solid #2563eb; border-radius: 6px;
      padding: 1.5rem 2rem; margin-bottom: 3rem;
    }
    .toc h2 {
      font-size: 1rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; color: #6b7280; margin-bottom: 1rem;
      border: none; padding: 0;
    }
    .toc ol  { padding-left: 1.25rem; }
    .toc li  { margin: 0.35rem 0; }
    .toc a   { color: #2563eb; text-decoration: none; font-size: 0.95rem; }
    .toc a:hover { text-decoration: underline; }

    /* Summary ссылка в ToC — выделяем визуально */
    .toc .toc-summary a {
      color: #1d4ed8;
      font-weight: 600;
    }

    .section {
      background: #fff; border-radius: 6px;
      padding: 2rem 2.5rem; margin-bottom: 2rem;
      border: 1px solid #e5e7eb;
    }

    h2 { font-size: 1.5rem; font-weight: 700; color: #111; margin: 1.5rem 0 0.75rem; }
    h3 { font-size: 1.2rem; font-weight: 600; color: #374151; margin: 1.25rem 0 0.6rem; }
    h4 { font-size: 1.05rem; font-weight: 600; color: #4b5563; margin: 1rem 0 0.5rem; }
    h5 { font-size: 0.95rem; font-weight: 600; color: #6b7280; margin: 0.75rem 0 0.4rem; }
    p  { margin-bottom: 0.85rem; color: #374151; }
    li { margin: 0.3rem 0; color: #374151; }

    .summary-block {
      background: #eff6ff; border: 1px solid #bfdbfe;
      border-left: 4px solid #2563eb; border-radius: 6px;
      padding: 2rem 2.5rem; margin-bottom: 2rem;
    }
    .summary-block .summary-label {
      font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.1em; color: #2563eb; margin-bottom: 1rem;
    }
    .summary-block p { color: #1e3a5f; }
    .summary-placeholder { color: #9ca3af; font-style: italic; font-size: 0.9rem; }

    .footer {
      margin-top: 3rem; padding-top: 1.5rem;
      border-top: 1px solid #e5e7eb; font-size: 0.8rem; color: #9ca3af;
      display: flex; flex-direction: column; gap: 0.3rem;
    }
    .footer a { color: #6b7280; word-break: break-all; }
    .footer a:hover { color: #374151; }

    .back-to-top {
      display: inline-block; margin-top: 0.75rem;
      font-size: 0.8rem; color: #9ca3af; text-decoration: none;
    }
    .back-to-top:hover { color: #2563eb; }
  </style>
</head>
<body>
  <div class="container" id="top">

    <h1 class="page-title">{{ page_title }}</h1>

    <!-- ══ Оглавление ══ -->
    <!-- FIX: итерируем toc_sections — ВСЕ разделы включая Summary -->
    <nav class="toc" id="toc" aria-label="Оглавление">
      <h2>Содержание</h2>
      <ol>
        {% for section in toc_sections %}
          {% if section.slug == "summary" %}
            <li class="toc-summary"><a href="#{{ section.slug }}">{{ section.title }}</a></li>
          {% else %}
            <li><a href="#{{ section.slug }}">{{ section.title }}</a></li>
          {% endif %}
        {% endfor %}
      </ol>
    </nav>

    <!-- ══ Разделы (только контент, без Summary) ══ -->
    {% for section in sections %}
      <div class="section" id="{{ section.slug }}">
        {{ section.html_content }}
        <a class="back-to-top" href="#top">↑ наверх</a>
      </div>
    {% endfor %}

    <!-- ══ Summary ══ -->
    <div class="summary-block" id="summary">
      <div class="summary-label">📋 Summary</div>
      {% if summary_is_placeholder %}
        <p class="summary-placeholder">{{ summary_text }}</p>
      {% else %}
        {{ summary_html }}
      {% endif %}
    </div>

    <!-- ══ Footer ══ -->
    <footer class="footer">
      <div>🔗 Источник: <a href="{{ source_url }}" target="_blank" rel="noopener">{{ source_url }}</a></div>
      <div>🕐 Дата скрапинга: {{ scraped_at }}</div>
      <div>🛠️ Сгенерировано: LangChain + YandexGPT | BeautifulSoup</div>
    </footer>

  </div>
</body>
</html>"""


# ─── Сборка и сохранение ─────────────────────────────────────────────────────

def build_html_page(
    toc:              "TableOfContents",
    section_contents: dict[int, str],
    source_url:       str,
    scraped_at:       str,
) -> str:
    """
    Рендерит Jinja2 шаблон.
    FIX: toc_sections — ВСЕ разделы для ToC (включая Summary)
         sections     — только контентные разделы для тела страницы
    """
    summary_section = next(
        (s for s in toc.sections if s.slug == "summary"), None,
    )
    summary_raw    = section_contents.get(summary_section.number, "") if summary_section else ""
    is_placeholder = _SUMMARY_PLACEHOLDER in summary_raw

    # FIX: ToC включает ВСЕ разделы — Summary появляется последним пунктом
    toc_sections = [
        {"slug": s.slug, "title": s.title}
        for s in toc.sections
    ]

    # Тело страницы — только контентные разделы (Summary рендерится отдельным блоком)
    sections_ctx = []
    for s in toc.sections:
        if s.slug == "summary":
            continue
        raw_text = section_contents.get(s.number, "")
        sections_ctx.append({
            "slug":         s.slug,
            "title":        s.title,
            "html_content": text_to_html_fragment(raw_text),
        })

    template = Template(_HTML_TEMPLATE)
    return template.render(
        page_title             = toc.page_title,
        toc_sections           = toc_sections,    # ← все разделы для ToC
        sections               = sections_ctx,     # ← только контент для тела
        summary_text           = summary_raw if is_placeholder else "",
        summary_html           = text_to_html_fragment(summary_raw) if not is_placeholder else "",
        summary_is_placeholder = is_placeholder,
        source_url             = source_url,
        scraped_at             = scraped_at,
    )


def save_final_html(html: str, output_dir: Path) -> Path:
    """Сохраняет финальный HTML в output_dir/final_page.html."""
    filepath = output_dir / "final_page.html"
    filepath.write_text(html, encoding="utf-8")
    logger.info(f"Финальный HTML сохранён: {filepath}")
    return filepath


# ─── Основной запуск ─────────────────────────────────────────────────────────

_required = {
    "generated_toc":       "Ячейка 4 (БЛОК 3 — ToC)",
    "sections_output_dir": "Ячейка 4 (БЛОК 3 — ToC)",
    "saved_html_path":     "Ячейка 2 (URL → HTML)",
}

_missing = [
    f"'{var}' — выполни {cell}"
    for var, cell in _required.items()
    if var not in globals() or globals()[var] is None
]

if _missing:
    for msg in _missing:
        print(f"❌ {msg}")

else:
    # URL: берём из виджета если он живой, иначе из имени файла
    try:
        _source_url = url_input.value.strip() or f"(из файла: {saved_html_path.name})"
    except NameError:
        _source_url = f"(из файла: {saved_html_path.name})"

    # Дата скрапинга: из имени файла hostname_YYYYMMDD_HHMMSS.html
    try:
        _ts_raw  = saved_html_path.stem.split("_")[-2:]
        _scraped = f"{_ts_raw[0][:4]}-{_ts_raw[0][4:6]}-{_ts_raw[0][6:]} {_ts_raw[1][:2]}:{_ts_raw[1][2:4]}"
    except Exception:
        _scraped = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"📖 Входные данные:")
    print(f"   Разделов  : {len(generated_toc.sections)}")
    print(f"   Папка     : {sections_output_dir}")
    print(f"   URL       : {_source_url}")

    # Читаем файлы разделов с диска
    section_contents = read_section_files(generated_toc, sections_output_dir)

    # Проверяем Summary
    _summary_sec = next((s for s in generated_toc.sections if s.slug == "summary"), None)
    _summary_raw = section_contents.get(_summary_sec.number, "") if _summary_sec else ""
    if _SUMMARY_PLACEHOLDER in _summary_raw:
        print(f"\n⚠️  Summary — placeholder. Запусти БЛОК 4 перед сборкой.")
        print(f"   HTML собирается с заглушкой — пересобери после генерации Summary.\n")

    print(f"\n🔨 Собираем HTML...")
    html = build_html_page(
        toc              = generated_toc,
        section_contents = section_contents,
        source_url       = _source_url,
        scraped_at       = _scraped,
    )

    final_path = save_final_html(html, sections_output_dir)

    # Симлинк в /workspace/data/ для удобного доступа
    link_path = Path("/workspace/data/latest_page.html")
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(final_path)

    size_kb = final_path.stat().st_size / 1024
    print(f"\n✅ БЛОК 5 завершён")
    print(f"   Файл    : {final_path}")
    print(f"   Размер  : {size_kb:.1f} КБ")
    print(f"   Симлинк : {link_path}")
    print(f"\n💡 Следующий шаг — БЛОК 6: скопировать в /var/www/html/ → Apache → браузер")

    # Передаём в следующую ячейку
    final_html_path = final_path
```

---
## Запуск

```bash
cd /home/j.smith/projects/module-01-scrapping

# 1. Создать .env из шаблона и вписать ключи
cp -v .env.example .env
nano .env

# 2. Собрать и запустить
docker compose up --build -d

# 3. Открыть в браузере
# http://localhost:8888

# 4. Логи если что-то не так
docker compose logs -f
```

---

**Критерии готовности этих трёх ячеек:**
- `✅ API доступен` в ячейке 1
- В `data/raw/` появился файл `*.html` после ячейки 2
- В `data/raw/` появился файл `*.txt` с маркерами `##` после ячейки 3
