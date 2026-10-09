## LangChain Enterprise — от локальной модели до агентных систем


**LangChain — это не язык программирования.** Это мощный open-source **фреймворк** (набор библиотек) для Python и JavaScript/TypeScript.

**Мы используем LangChain как оркестратор**. Если Kubernetes управляет контейнерами, а Terraform — инфраструктурой, то LangChain управляет логикой работы AI-приложений. Он предоставляет нам унифицированные абстракции (интерфейсы), чтобы мы не писали "костыли" для интеграции с разными провайдерами и инструментами.

Сама по себе "голая" LLM (GPT-4, Claude, Llama) — это просто текстовый генератор, "мозг в банке" без доступа к внешнему миру. С помощью LangChain мы превращаем этот мозг в полноценную систему.

**Что конкретно мы делаем с его помощью:**
1. **Строим пайплайны (Chains):** связываем промпты, модели и парсеры ответов в единый конвейер. Именно это мы реализуем через протокол `Runnable` .
2. **Даем модели "руки" (Agents / Tools):** пишем функции, которые позволяют LLM самостоятельно ходить в интернет, выполнять SQL-запросы к нашим базам, дергать корпоративные API или запускать Python-код.
3. **Подключаем корпоративные знания (RAG):** интегрируем векторные базы данных (VectorStores), чтобы модель читала нашу приватную документацию и отвечала по ней, а не галлюцинировала.
4. **Добавляем память (Memory):** сохраняем историю диалога, чтобы система помнила контекст прошлых обращений.

### Схема архитектуры

```text
	 [ Входящий запрос ]
	         │
	         ▼
 ┌─────────────────────────┐
 │       LangChain         │ ◄ Корпоративный "клей" / Оркестратор
 │  (Цепочки, Роутеры)     │
 └──────┬────────┬───────┬─┘
        │        │       │
        ▼        ▼       ▼
     ┌─────┐  ┌─────┐ ┌───────┐
     │ LLM │  │ RAG │ │ Tools │
     └─────┘  └─────┘ └───────┘
    (Мозги)    (Базы)  (API/Утилиты)

```

Это стандарт индустрии для создания LLM-приложений. Мы используем этот фреймворк, чтобы быстро, надежно и единообразно собирать разрозненные компоненты (модели, базы данных, утилиты) в отказоустойчивый production-ready продукт.


## БЛОК 2: ОСНОВЫ LANGCHAIN И ЛОКАЛЬНЫЕ МОДЕЛИ

### 2.1 Архитектурная эволюция LangChain: от цепочек к Runnable

LangChain прошёл три отчётливых архитектурных поколения. Понимать их важно: большинство примеров в интернете написаны под первое или второе поколение, и запустить их без переработки не получится.

**Поколение 1 (2022–2023): класс-наследники Chain.** Всё строилось вокруг объектов типа `LLMChain`, `SequentialChain`, `RouterChain`. Конфигурация передавалась через конструктор, выход одной цепочки вручную подавался на вход другой. Стриминг требовал отдельной реализации.

**Поколение 2 (2024): LangChain Expression Language (LCEL).** (/ˌel siː aɪ ˈel/) Команда переосмыслила архитектуру: ввела единый интерфейс `Runnable`, который реализуют абсолютно все компоненты — промпты, модели, парсеры, ретриверы, лямбды. Поверх него — оператор `|`, делающий композицию читаемой.

**Поколение 3 (октябрь 2025): LangChain 1.0.** Официальный stable-релиз закрепил LCEL как единственно рекомендуемый способ построения всего. Пакет `langchain-community` разделён на поставщик-специфичные пакеты (`langchain-openai`, `langchain-ollama`, `langchain-qdrant` (`/'kwɒdrənt/`) ). Мета-пакет `langchain` тянет `langchain-core`, но не тянет интеграции — их нужно ставить явно.

По данным публичного GitHub репозитория (июль 2026): 235 миллионов ежемесячных загрузок из PyPI, 35% компаний из Fortune 500 используют в production. 84.7% сообщества — Python.

Официальная рекомендация команды LangChain в 2026 году:
```
LangChain   → строительные блоки (промпты, парсеры, ретриверы)
LangGraph   → оркестрация агентов и multi-step workflows
LangSmith   → observability и eval (или Langfuse для self-hosted)
```


### 2.2 Протокол Runnable: единый интерфейс

`Runnable` — это абстрактный базовый класс из `langchain-core`. Любой объект, реализующий этот протокол, получает следующий набор методов:

```
invoke(input)  → синхронный вызов, один результат
- используем этот метод например в изолированных скриптах автоматизации и для разовой классификации инцидентов, где скорость отклика не критична.

ainvoke(input)  → async-версия invoke
- архитектурный стандарт для высоконагруженных FastAPI-микросервисов, позволяющий запрашивать LLM при обработке вебхуков от CI/CD, не блокируя event loop сервера.

stream(input)  → генератор, токены по мере генерации
- применяем его в утилитах, чтобы пользователи видели процесс написания длинных, например чтобы DevOps-инженеры видели процесс написания Terraform-манифестов прямо в терминале без минутного ожидания.

astream(input)  → async-версия stream
- ядро корпоративного AI-хелпдеска, которое транслирует текст через WebSockets в браузер пользователя, обеспечивая метрику TTFT (Time To First Token) менее 500 миллисекунд.

batch(inputs)  → пакетный вызов, список результатов
- запускаем этот метод в тяжелых ночных ETL-пайплайнах, например чтобы за один сетевой проход разметить тегами критичности сотни тысяч накопившихся за день логов безопасности.

abatch(inputs)  → async-версия batch
- используем асинхронное пакетирование для параллельной быстрой обработки десятков тысяч событий например из MTS WS, выявляя аномалии ИБ с максимальной пропускной способностью сети

astream_events(input) → async-генератор событий всего графа
- интегрируем его для глубокого SRE-мониторинга сложных AI-агентов, чтобы в реальном времени логировать каждый шаг системы: старт поиска в базе, вызов внешнего API, промежуточные ошибки графа и т.д.
```

Символ `|` — это встроенный оператор Python (изначально — побитовое «ИЛИ»). Но в данном случае мы имеем дело с **перегрузкой операторов (Operator Overloading)**. Разработчики LangChain переопределили под капотом магический метод `__or__` для класса `Runnable`. Мы используем этот синтаксис как прямую отсылку к **UNIX-пайпам (pipes)**. Точно так же, как в bash мы передаём поток данных от одной утилиты к другой (`cat logs.txt | grep error | wc -l`), в LangChain мы пробрасываем выход (output) одного компонента на вход (input) следующего: `prompt | llm | parser`.

**Пример:**
```python
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def write_to_disk(chunk: str) -> str:
    """
    Перехватывает каждый токен и дописывает его в файл (режим 'a' - append).
    Именно так работают базовые File Tools у AI-агентов.
    """
    with open("/workspace/data/agent_report.md", "a", encoding="utf-8") as f:
        f.write(chunk)
    return chunk
    
prompt = ChatPromptTemplate.from_template("""
Ты — корпоративный IT-ассистент. Отвечай строго по контексту.
Если ответа нет — скажи "Информация отсутствует в базе знаний".

Контекст:
{context}

Вопрос: {question}
""")

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", reasoning=False)

# Подготовка: создаем директорию и очищаем файл перед запуском
os.makedirs("/workspace/data", exist_ok=True)
with open("/workspace/data/agent_report.md", "w", encoding="utf-8") as f:
    f.write("# Автоматический RAG-отчет\n\n")

# ── 6 компонентов, каждый делает одно дело ──────────────

# 1. Retriever    — ищет релевантные чанки в векторной БД
# 2. format_docs  — список Document → одна строка текста
# 3. Prompt       — вставляет контекст и вопрос в шаблон
# 4. LLM          — генерирует ответ на основе контекста
# 5. Parser       — срезает служебные теги, возвращает str
# 6. FS Operator  — пишет стрим в файловую систему (Agent Tool)

# context: Берет вопрос пользователя, находит релевантные куски текста через гибридный поиск (hybrid_retriever) и сразу передает их в функцию format_docs, чтобы склеить в единый читаемый текст
# question: RunnablePassthrough() — просто берет вопрос пользователя и прокидывает его дальше по конвейеру "как есть".

rag_chain = (
    {"context": hybrid_retriever | RunnableLambda(format_docs),  # 1 + 2
     "question": RunnablePassthrough()}                          # вопрос без изменений
    | prompt             # 3
    | llm                # 4
    | StrOutputParser()  # 5
    | RunnableLambda(write_to_disk)  # 6 файловый оператор
)

# Стриминг — токены появляются по мере генерации
print("ОТВЕТ С RAG (параллельно пишется в /workspace/data/agent_report.md):\n")
for chunk in rag_chain.stream("Как установить freeipa-server на Astra Linux 1.8?"):
    print(chunk, end="", flush=True)
```

Ключевое следствие: поскольку все компоненты реализуют один интерфейс, оператор `|` работает между любыми из них. `ChatPromptTemplate` — это `Runnable`. `ChatOllama` — это `Runnable`. `StrOutputParser` — это `Runnable`. Поэтому выражение `prompt | llm | parser` технически означает создание объекта `RunnableSequence`, который сам тоже является `Runnable` и может быть вложен в другую цепочку.

---

#### `astream` vs `astream_events` — критическое различие для агентов

Обычный `astream` возвращает только токены **финального ответа** модели. Для простых цепочек этого достаточно. Но в агентных системах с Tool Calling между первым запросом и финальным ответом происходит множество промежуточных шагов — вызовы инструментов, RAG-ретривал, внутренние рассуждения. Пользователь видит лишь молчание.

`astream_events(version="v2")` решает эту проблему: он транслирует **каждое событие** внутри графа в реальном времени. Каждое событие — это словарь с полями `event`, `name`, `data`:

```python
async for event in chain.astream_events({"question": query}, version="v2"):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        # Токены LLM по мере генерации
        print(event["data"]["chunk"].content, end="", flush=True)

    elif kind == "on_tool_start":
        # Инструмент начал выполняться — можно показать спиннер в UI
        print(f"\n[▶ Вызов инструмента: {event['name']}]")

    elif kind == "on_tool_end":
        # Инструмент завершился — можно скрыть спиннер
        print(f"[✓ {event['name']} завершён]\n")

    elif kind == "on_retriever_end":
        # RAG-ретривал завершён — можно показать источники
        docs = event["data"]["output"]
```

Типичные события и их значение:

```
on_chat_model_start   → LLM начал генерацию
on_chat_model_stream  → новый токен от LLM
on_chat_model_end     → LLM завершил генерацию
on_tool_start         → агент вызвал инструмент
on_tool_end           → инструмент вернул результат
on_retriever_start    → начался RAG-ретривал
on_retriever_end      → ретривал завершён, документы получены
on_chain_start/end    → старт и конец любого Runnable-узла
```

Версии протокола: `v2` — текущий стандарт, поддерживает кастомные события; `v1` — устаревший, будет удалён в `0.4.0`. `astream_log` (старый метод) официально депрекирован — не использовать в новом коде.

**Практическое правило:** в простых цепочках (`prompt | llm | parser`) — достаточно `astream`. Как только появляются агенты, инструменты или LangGraph-граф — переходить на `astream_events(version="v2")`.

**Пример:**
```python
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def write_to_disk(chunk: str) -> str:
    """
    Перехватывает каждый токен и дописывает его в файл (режим 'a' - append).
    Именно так работают базовые File Tools у AI-агентов.
    """
    with open("/workspace/data/agent_report.md", "a", encoding="utf-8") as f:
        f.write(chunk)
    return chunk
    
prompt = ChatPromptTemplate.from_template("""
Ты — корпоративный IT-ассистент. Отвечай строго по контексту.
Если ответа нет — скажи "Информация отсутствует в базе знаний".

Контекст:
{context}

Вопрос: {question}
""")

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")

# Подготовка: создаем директорию и очищаем файл перед запуском
os.makedirs("/workspace/data", exist_ok=True)
with open("/workspace/data/agent_report.md", "w", encoding="utf-8") as f:
    f.write("# Автоматический RAG-отчет\n\n")

# ── 6 компонентов, каждый делает одно дело ──────────────
rag_chain = (
    {"context": hybrid_retriever | RunnableLambda(format_docs),  # 1 + 2
     "question": RunnablePassthrough()}                          # вопрос без изменений
    | prompt             # 3
    | llm                # 4
    | StrOutputParser()  # 5
    | RunnableLambda(write_to_disk)  # 6 файловый оператор
)

print("ОТВЕТ С RAG (Глубокая телеметрия через astream_events):\n")

# ДЕМОНСТРАЦИЯ: Асинхронный перехват событий графа
# В Jupyter Notebook top-level await работает по умолчанию
async for event in rag_chain.astream_events(
    "Как установить astra-freeipa-server на Astra Linux 1.8?", 
    version="v2"
):
    kind = event["event"]

    # Событие 1: Ретривер нашел документы (Идеально для дебага контекста)
    if kind == "on_retriever_end":
        docs = event["data"]["output"]
        print(f"[🔍 Векторный поиск завершен: найдено {len(docs)} чанков]")
        print("─" * 60)

    # Событие 2: Нейросеть начала работу (Идеально для показа UI-спиннера)
    elif kind == "on_chat_model_start":
        print("[▶ LLM начала генерацию...]\n")

    # Событие 3: Прилетел новый токен (Идеально для стриминга текста пользователю)
    elif kind == "on_chat_model_stream":
        # .content достает строку из AIMessageChunk
        print(event["data"]["chunk"].content, end="", flush=True)

print("\n\n[✓ Генерация завершена. Файл /workspace/data/agent_report.md сохранен]")
```

---

#### Примитивы LCEL, выходящие за пределы простой последовательности

`RunnablePassthrough` просто пробрасывает входные данные без изменений — стандартная идиома для передачи исходного вопроса сквозь параллельную ветку.

**`RunnableParallel` и неявное приведение типов**

`RunnableParallel` выполняет несколько ветвей одновременно и возвращает словарь. Стандартная RAG-идиома выглядит так:

```python
{"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

Здесь важно понимать механику: обычный Python `dict` (часть выражения, заключённая в фигурные скобки `{}`) **не является** `RunnableParallel` сам по себе. Когда оператор `|` встречает словарь, внутри метода вызывается вспомогательная функция `coerce_to_runnable()`, которая явно конвертирует его:

```python
# Из исходника langchain-core/runnables/base.py
def coerce_to_runnable(thing):
    if isinstance(thing, Runnable):
        return thing
    elif callable(thing):
        return RunnableLambda(thing)
    elif isinstance(thing, dict):
        return RunnableParallel(thing)  # ← вот здесь происходит преобразование
```

Почему это важно знать? При возникновении ошибки в цепочке стектрейс будет ссылаться на `RunnableParallel`, хотя в коде написан словарь. Без понимания этой конвертации дебаг превращается в загадку.

Стектрейсы при дебаге:
Когда у инженера упадет ячейка (например, ретривер вернет None или затаймаутит сеть), в логе Python traceback будет написано не KeyError in dict, а ошибка внутри RunnableParallel.invoke() или RunnableParallel.transform().

Без этого объяснения разработчик потратит полчаса на поиск слова RunnableParallel у себя в файле и не найдет его, так как в коде фигурируют только фигурные скобки {}.

---

**`RunnableBranch`** — условная маршрутизация. Принимает список пар `(условие, runnable)` и обязательный fallback. Первый runnable, для которого условие вернуло `True`, выполняется. Аналог `if/elif/else`, но компонуемый:

```python
branch = RunnableBranch(
    # IF: если текст содержит "urgent" -> запрос уходит в urgent_chain
    (lambda x: "urgent" in x["text"], urgent_chain),
    # ELIF: если текст содержит "billing" -> запрос уходит в billing_chain
    (lambda x: "billing" in x["text"], billing_chain),
    
    # ELSE: если ни одно условие не подошло -> отрабатывает дефолтный general_chain
    general_chain  # fallback
)
```

---

**`RunnableLambda`** — оборачивает обычную Python-функцию в `Runnable`. Позволяет вставлять произвольную логику (форматирование, фильтрацию, трансформации) в середину цепочки без нарушения интерфейса:

```python
def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)

chain = retriever | RunnableLambda(format_docs) | prompt | llm
```

---

**`RunnableWithFallbacks` и метод `.with_fallbacks()`**

Оборачивает `Runnable` и задаёт список резервных вариантов. При исключении (таймаут, API error, OOM) пробует следующий по порядку. Production-необходимость при работе с несколькими провайдерами.

Технически класс `RunnableWithFallbacks` можно инстанцировать напрямую, но **идиоматичный (естественный) и рекомендуемый способ** — метод `.with_fallbacks()`, который доступен любому `Runnable`-объекту:

```python
# Идиоматично — метод .with_fallbacks()
llm = ChatOllama(model="qwen3:8b").with_fallbacks(
    # TRY/CATCH под капотом: если Qwen3 (основная модель) упал, недоступен или выдал таймаут -> 
    # запрос бесшовно и автоматически перенаправляется на резервный vLLM-сервер
    [ChatOpenAI(base_url="http://backup-vllm/v1", model="tlite-dpo")]
)

# Работает, но не идиоматично — прямая инстанция
from langchain_core.runnables import RunnableWithFallbacks
llm = RunnableWithFallbacks(
    # Логика абсолютно та же, но синтаксис громоздкий. 
    # В production-коде LangChain так писать не принято (нарушает читаемость пайплайнов)
    runnable=ChatOllama(model="qwen3:8b"),
    fallbacks=[ChatOpenAI(base_url="http://backup-vllm/v1", model="tlite-dpo")]
)

# Fallback можно повесить и на всю цепочку целиком
chain_with_fallback = (
    # Пытаемся выполнить основную ветку
    prompt | primary_llm | StrOutputParser()
).with_fallbacks([
    # Если на ЛЮБОМ этапе выше произошла ошибка (например, OutputParser не смог распарсить кривой JSON от primary_llm) -> 
    # фреймворк откатится назад и запустит эту запасную ветку целиком с самого начала
    prompt | backup_llm | StrOutputParser()
])
```

---

### Пример 1: Неявный `RunnableParallel` и `RunnablePassthrough` (Классический RAG)

Здесь мы демонстрируем, как разделить один входящий запрос на два параллельных потока: первый поток пойдет в базу данных за контекстом, а второй поток пробросит сам вопрос без изменений, чтобы затем мы объединили их в промпте.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Создаем функцию-заглушку. В реальном проекте здесь мы обращаемся 
# к векторной БД (VectorStoreRetriever), чтобы найти нужные документы.
retriever = lambda query: f"Извлеченный контекст для: {query}"

# 2. Формируем шаблон промпта. Ожидаем две переменные: {question} и {context}.
prompt = ChatPromptTemplate.from_template(
    "Ответь на вопрос: {question}\nИспользуй контекст: {context}"
)

# 3. Собираем саму цепочку (Pipeline).
rag_chain = (
    {
        # ДЕМОНСТРАЦИЯ RUNNABLE PARALLEL:
        # Мы передаем обычный словарь {...}, но LangChain "под капотом" 
        # автоматически превращает его в RunnableParallel. 
        # Это значит, что обе строчки ниже выполнятся ОДНОВРЕМЕННО:
        
        # Ветка А: Отправляем наш начальный запрос в функцию поиска контекста
        "context": retriever, 
        
        # Ветка Б: ДЕМОНСТРАЦИЯ RUNNABLE PASSTHROUGH
        # Мы используем RunnablePassthrough(), чтобы просто "пропустить" 
        # изначальный текст запроса без изменений и положить его в ключ "question"
        "question": RunnablePassthrough()
    } 
    # 3. Результат предыдущего шага (словарь с ключами context и question) 
    # мы через оператор | (pipe) передаем на вход нашему шаблону промпта.
    | prompt             # 3
    | llm                # 4
    | StrOutputParser()  # 5
)

# 6. Вызываем цепочку. Мы передаем строку "Что такое eBPF?".
# Она параллельно пойдет в ветку А (искать контекст) и ветку Б (просто пробросится).
print("=== RAG Chain ===")
print(rag_chain.invoke("Что такое eBPF?"))

```

### Пример 2: Условная маршрутизация `RunnableBranch` + `RunnableLambda`

Здесь мы демонстрируем паттерн "Маршрутизатор" (Router). Мы анализируем входящий запрос пользователя и "на лету" решаем, в какой специализированный пайплайн его направить — к безопасникам, к девопсам или в общую модель.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 1. Подготавливаем специализированные узлы-обработчики.
# В реальном production здесь мы бы вызывали разные промпты или разных AI-агентов.
# Мы оборачиваем обычные лямбда-функции в RunnableLambda, чтобы 
# интегрировать их в общую экосистему LCEL.
sec_ops_chain = RunnableLambda(lambda x: f"[SecOps Alert] Запущен пайплайн анализа: {x['query']}")
dev_ops_chain = RunnableLambda(lambda x: f"[DevOps] Проверяем статус в CI/CD: {x['query']}")

# 2. Создаем ветку по умолчанию (fallback/general).
# Мы будем направлять сюда абсолютно все запросы, которые не требуют узкоспециализированной логики.
fallback_chain = RunnableLambda(lambda x: f"[General] Стандартный запрос в базовую LLM: {x['query']}")

# 3. ДЕМОНСТРАЦИЯ RUNNABLE BRANCH:
# Мы создаем декларативный аналог конструкции if-elif-else, 
# который управляет потоком выполнения (Control Flow) в LangChain.
router = RunnableBranch(
    
    # Условие 1 (IF): Мы пишем предикат (функцию, возвращающую True/False).
    # Если мы находим подстроку "уязвимость" в запросе, 
    # мы мгновенно передаем управление в sec_ops_chain.
    (lambda x: "уязвимость" in x["query"].lower(), sec_ops_chain),
    
    # Условие 2 (ELIF): Если первое условие не сработало, проверяем дальше.
    # Если мы видим слово "деплой", 
    # мы перенаправляем запрос в dev_ops_chain.
    (lambda x: "деплой" in x["query"].lower(), dev_ops_chain),
    
    # Действие по умолчанию (ELSE): Это обязательный последний аргумент без условия.
    # Мы отправляем запрос сюда, если ни один из предикатов выше не вернул True.
    fallback_chain 
)

print("\n=== Условная маршрутизация (Router) ===")

# 4. Тестируем наш маршрутизатор профильным запросом.
# Мы передаем словарь. Роутер проверяет первую ветку, находит совпадение ("уязвимость") 
# и логика уходит в SecOps-обработчик. Остальные проверки игнорируются.
print("Тест 1 (SecOps):")
print(router.invoke({"query": "Срочно проверь манифесты на уязвимость CVE-2024"}))

print("-" * 40)

# 5. Тестируем непрофильным запросом.
# Мы отправляем бытовой вопрос. Ни "уязвимость", ни "деплой" не найдены.
# Мы автоматически проваливаемся в ветку по умолчанию (fallback_chain).
print("Тест 2 (General):")
print(router.invoke({"query": "Как приготовить пиццу в домашних условиях?"}))

```

### Пример 3: Отказоустойчивость с `.with_fallbacks()` 🛡️

Здесь мы демонстрируем стандартный паттерн SRE (Graceful Degradation). Мы настраиваем систему так, чтобы при падении внешнего API (например, OpenAI ответил таймаутом), наш пайплайн не "упал", а бесшовно переключился на запасную локальную модель.

```python
from langchain_core.runnables import RunnableLambda
import time

# 1. Эмулируем нестабильный внешний API. 
# Мы специально пишем функцию, которая всегда выбрасывает критическую ошибку соединения.
def unstable_llm(prompt):
    raise ConnectionError("Timeout: Primary LLM API is down!")

# 2. Эмулируем локальную резервную модель (например, развернутую нами в кластере Ollama).
# Эта функция работает надежно и всегда отдает результат.
def local_backup_llm(prompt):
    return f"[Ollama Local Backup] Успешный ответ на: {prompt}"

# 3. Оборачиваем наши обычные Python-функции в RunnableLambda.
# Мы делаем это, чтобы функции получили интерфейс Runnable (invoke, batch и т.д.) 
# и могли связываться в цепочки LCEL.
primary_runnable = RunnableLambda(unstable_llm)
backup_runnable = RunnableLambda(local_backup_llm)

# 4. ДЕМОНСТРАЦИЯ WITH_FALLBACKS:
# Мы берем основной (падающий) узел и привязываем к нему список запасных вариантов.
resilient_chain = primary_runnable.with_fallbacks(
    # Передаем список запасных узлов (в нашем случае - один).
    [backup_runnable], 
    
    # Строго указываем, какие именно ошибки мы перехватываем. 
    # Мы не перехватываем синтаксические ошибки кода, а ловим только проблемы с сетью.
    exceptions_to_handle=(ConnectionError,)
)

# 5. Запускаем систему. 
# Мы видим, что скрипт не падает с красным трейсбеком ошибки. 
# Под капотом primary_runnable выбрасывает ConnectionError, LangChain это замечает 
# и мы мгновенно получаем ответ от backup_runnable.
print("\n=== Fallback Mechanism ===")
print(resilient_chain.invoke("Объясни Kubernetes RBAC"))

```

---

### 2.3 Инверсия зависимостей от провайдера

Центральная ценность LangChain для on-premise сценариев — единый интерфейс поверх любого провайдера. Замена `ChatOpenAI` на `ChatOllama` требует изменения ровно одного объекта при сохранении всей остальной логики:

```python
# Облако (разработка)
llm = ChatOpenAI(model="gpt-4o", api_key="...")

# On-premise через Ollama
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")

# On-premise через vLLM (OpenAI-совместимый API)
llm = ChatOpenAI(base_url="http://localhost:8001/v1", api_key="not-needed", model="tlite-dpo")
```

Цепочка `prompt | llm | parser` не меняется ни в одном случае. Это принцип инверсии зависимостей, применённый к LLM-провайдерам.

Важное практическое разграничение между двумя актуальными пакетами для Ollama:

```
langchain-community:  from langchain_community.chat_models import ChatOllama
                      (устаревший путь, deprecated)

langchain-ollama:     from langchain_ollama import ChatOllama
                      (актуальный, отдельный пакет с 2025)
```

---

### 2.4 Thinking mode в Qwen-3 и когда его отключать

Qwen-3 — гибридная модель с нативным dual-mode: переключение между thinking и обычным режимом заложено непосредственно в post-training самой модели. При включённом thinking mode модель генерирует блок `<think>...</think>` перед ответом, работая по принципу Chain-of-Thought внутри самой генерации. Если thinking успешно отключён — модель физически не генерирует рассуждения, ответ быстрее и короче.

---

**Управление через параметр `reasoning` в `ChatOllama`:**

```python
from langchain_ollama import ChatOllama

# reasoning=True — включает режим рассуждений явно.
# Содержимое <think>...</think> убирается из основного ответа
# и возвращается отдельно:
# response.additional_kwargs["reasoning_content"]
llm = ChatOllama(model="qwen3:8b", reasoning=True)

# reasoning=False — должен отключать рассуждения.
# Известный баг: для Qwen3 через Ollama этот флаг
# не всегда подавляет <think>-теги — см. ниже.
# Для DeepSeek-R1 работает корректно.
llm = ChatOllama(model="qwen3:8b", reasoning=False)

# reasoning=None (по умолчанию) — поведение модели по умолчанию.
# Для Qwen3: <think>-теги присутствуют прямо в тексте ответа.
llm = ChatOllama(model="qwen3:8b")  # reasoning=None
```

---

**Почему `reasoning=False` ведёт себя по-разному для Qwen3 и DeepSeek-R1:**

Qwen-3 имеет обученный dual-mode: переключатель thinking/non-thinking заложен в веса модели в процессе RLHF. Когда Ollama передаёт команду на отключение — модель должна сама принять решение не думать. Из-за несовершенства этого взаимодействия в ряде версий `langchain-ollama` команда до модели не доходит корректно — баг.

DeepSeek-R1 такого переключателя в весах **не имеет**: reasoning запечён в модель через RL-обучение как неотъемлемая часть генерации. При `reasoning=False` Ollama применяет prompt-trick — вставляет в шаблон пустой `<think></think>` блок до начала генерации, и модель продолжает уже без рассуждений. Это работает и даёт реальный прирост скорости, но механизм другой: не нативный режим модели, а инжекция на уровне сервера. Именно поэтому `reasoning=False` через `langchain-ollama` для R1 надёжен — Ollama полностью управляет этим на своей стороне.

---

**Надёжный способ отключить thinking для Qwen3:**

Директива `/no_think` — это инструкция **уровня самой модели Qwen3**, заложенная в её post-training. Она работает в любом serving-фреймворке: Ollama, vLLM, SGLang, llama.cpp, HuggingFace Transformers — и не зависит от корректности реализации `reasoning=False` в конкретной версии клиентской библиотеки:

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3:8b")  # reasoning=None

# /no_think — в системном промпте, работает глобально на весь диалог
prompt = ChatPromptTemplate.from_messages([
    ("system", "/no_think"),
    ("human", "{question}")
])

chain = prompt | llm
```

Примечание по написанию: официальный README Qwen3 использует `/nothink`, авторы модели в обсуждениях подтверждают `/no_think` (с подчёркиванием) как каноническую форму. Обе работают; `/no_think` — предпочтительный вариант.

---

**Когда включать (`reasoning=True`):**

- Многошаговые рассуждения: математика, алгоритмы, логические задачи
- Планировочные запросы («составь архитектуру системы», «сравни подходы»)
- Задачи с неопределённостью, где нужно взвесить несколько вариантов

**Когда отключать (директива `/no_think` в промпте):**

- **RAG-системы** — контекст из базы знаний уже даёт модели «материал»; рассуждения поверх него потребляют токены без прироста качества
- **Агентные системы с Tool Calling** — рассуждения занимают часть контекстного окна, отведённого под историю диалога и описания инструментов; каждая итерация агента дороже
- **Structured output** — не используйте `reasoning=None` совместно с `.with_structured_output()`: `<think>`-теги попадут в сырой текст и сломают JSON-парсер; используйте `reasoning=True`, чтобы теги были изолированы в `additional_kwargs["reasoning_content"]` до парсинга
- **Latency-чувствительные endpoints** — thinking mode добавляет 200–1000 токенов на запрос; при потоковом интерфейсе пользователь видит долгую паузу до первого токена ответа
---

### 2.5 Structured Output: механика `.with_structured_output()`

До появления `.with_structured_output()` разработчики использовали `JsonOutputParser` с ручными инструкциями в промпте: «отвечай только в формате JSON». Это нестабильно — модель может добавить пояснения, обернуть JSON в markdown-блок, пропустить поле.

`.with_structured_output(PydanticModel)` работает иначе в зависимости от поддержки провайдером:

**При поддержке tool_choice (OpenAI API-совместимые):** LangChain конвертирует Pydantic-схему в JSON Schema, передаёт её как параметр `tools` с `tool_choice: "required"`. Это ограничение на уровне протокола: модель обязана вернуть вызов функции с валидным JSON — текст в основном ответе вернуть невозможно.

**При отсутствии поддержки tool_choice:** LangChain добавляет JSON-схему в системный промпт и использует `JsonOutputParser`. Это мягкая гарантия — зависит от послушности модели.

**Как это работает вместе с thinking mode у Qwen-3:**

Когда `reasoning=True` задан одновременно с `.with_structured_output()`, Ollama обрабатывает запрос в два этапа:
1. Модель свободно генерирует `<think>...</think>` блок — `reasoning=True` изолирует его в `additional_kwargs["reasoning_content"]`, не допуская в основной контент
2. После закрывающего `</think>` модель генерирует вызов инструмента строго по схеме, удовлетворяя `tool_choice: "required"`

Таким образом, `<think>`-теги не попадают в JSON-полезную нагрузку — не потому что посимвольный constrained decoding блокирует эти символы, а потому что `reasoning=True` убирает thinking-блок до того, как tool_choice начинает формировать структурированный ответ.

```python
from pydantic import BaseModel
from langchain_ollama import ChatOllama

class SupportTicket(BaseModel):
    category: str
    priority: str
    summary: str

# reasoning=True: теги <think> уходят в additional_kwargs,
# не в JSON-полезную нагрузку
llm = ChatOllama(model="qwen3:8b", reasoning=True)

# Базовое использование — только Pydantic-объект
structured_llm = llm.with_structured_output(SupportTicket)
ticket = structured_llm.invoke("Упал прод, не работает авторизация с 14:00")
print(ticket.category)   # "infrastructure"
print(ticket.priority)   # "critical"
```

Если нужен доступ и к результату, и к самому рассуждению модели (для observability, логирования, аудита), используйте `include_raw=True`:

```python
# include_raw=True: возвращает словарь с parsed-объектом и raw AIMessage
structured_llm = llm.with_structured_output(SupportTicket, include_raw=True)
result = structured_llm.invoke("Упал прод, не работает авторизация с 14:00")

# Pydantic-объект
ticket = result["parsed"]
print(ticket.category)   # "infrastructure"

# Само рассуждение модели из raw AIMessage
reasoning = result["raw"].additional_kwargs.get("reasoning_content", "")
print(f"Ход рассуждений: {reasoning[:200]}...")

# Если парсинг упал — ошибка здесь, не исключением
if result["parsing_error"]:
    print(f"Ошибка парсинга: {result['parsing_error']}")
```

Qwen-3 поддерживает tool_choice через Ollama OpenAI-совместимый эндпоинт, поэтому `.with_structured_output()` работает в режиме протокольной гарантии.

---

#### Пример 3: Жесткая гарантия `Structured Output` (Native vs Emulated)

Демонстрируем, как мы заставляем модель отвечать строго в формате JSON (Pydantic), проверяя разницу между нативным `tool_choice` и эмуляцией.

```python
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# 1. Описываем жесткую схему данных, которую мы ждем от модели.
# Это контракт API между LLM и нашим бэкендом.
class VulnerabilityReport(BaseModel):
    cve_id: str = Field(description="Идентификатор уязвимости (например, CVE-2024-XXXX)")
    severity: str = Field(description="Критичность: low, medium, high, critical")
    affected_components: List[str] = Field(description="Список затронутых библиотек")

# 2. Инициализируем On-Premise модель через Ollama Native Integration
llm_ollama = ChatOllama(model="qwen2.5:8b")

# ДЕМОНСТРАЦИЯ WITH_STRUCTURED_OUTPUT:
# Мы применяем метод к модели и передаем Pydantic класс.
# LangChain автоматически сериализует класс в JSON Schema 
# и настраивает промпт/инструменты под капотом.
structured_llm = llm_ollama.with_structured_output(VulnerabilityReport)

# 3. Вызываем. Модель не может ответить текстом "Привет, вот ваш отчет...",
# она обязана вернуть валидный JSON, который LangChain сразу распарсит в объект Python.
print("=== Structured Output ===")
result = structured_llm.invoke(
    "Скрипт показал наличие CVE-2023-4863 в пакете libwebp. Уровень угрозы - высокий. Смежные пакеты: chromium."
)

# Мы получаем строгий типизированный объект (VulnerabilityReport).
print(f"CVE: {result.cve_id}")
print(f"Критичность: {result.severity.upper()}")
print(f"Затронуто: {result.affected_components}")

```

---

### 2.6 Мультимодальность: LLaVA устарел, llama3.2-vision и qwen2.5-vl — настоящее

Семейство LLaVA появилось в 2023 году и было первым популярным open-source мультимодальным решением. К 2025 году оно уступило по всем бенчмаркам следующему поколению:

```
LLaVA 1.6 (2024) → llama3.2-vision (Meta, сентябрь 2024)
                   qwen2.5-vl (Alibaba, февраль 2025)

Сравнение на задаче OCR из скриншотов:
LLaVA 1.6 13B:       ~72% точность
llama3.2-vision 11B:  ~87% точность
qwen2.5-vl 7B:        ~91% точность
```

Для корпоративного сценария (анализ скриншотов ошибок, чтение PDF-таблиц, документы с печатями) разница критична. Архитектурно два современных решения устроены принципиально по-разному.

**Llama 3.2 Vision** — cross-attention архитектура. Отдельно обученный визуальный адаптер передаёт представления ViT-энкодера в языковую часть через серию cross-attention слоёв. Языковая модель (Llama 3.1) при этом остаётся замороженной, что позволяет дообучать только адаптер.

**Qwen2.5-VL** — MLP merger архитектура с радикально переработанным ViT. Здесь нет cross-attention между визуальной и языковой частью: вместо этого визуальные токены через двухслойный MLP приводятся к размерности LLM и конкатенируются с текстовыми токенами как единая последовательность. Качество OCR 91%+ достигается за счёт двух архитектурных инноваций:

- **Динамическое разрешение**: ViT обучен с нуля без фиксированного входного размера — изображение делится на сетку патчей переменного размера. Динамическое разрешение - ViT в Qwen обучен принимать изображения в их оригинальном соотношении сторон без искажений и кропов. Модель динамически вычисляет количество визуальных токенов в зависимости от сложности и размера картинки, а не загоняет её в жесткие рамки. LLaVA 1.5 сжимал всё в фиксированный квадрат, теряя детали.
- **M-RoPE (Multimodal Rotary Positional Embedding)**: расширение стандартного RoPE на 2D пространственные координаты и временную ось для видео. Модель «знает» не просто порядок токенов, но их реальное положение на изображении.

LLaVA для сравнения: CLIP-энкодер → простой MLP-проектор → конкатенация с текстом. Ни динамического разрешения, ни пространственного позиционирования.

---

## БЛОК 3: ПОИСКОВЫЕ СИСТЕМЫ И RAG

### 3.1 Фундаментальное ограничение LLM и почему RAG — ответ

Языковая модель — это функция `f(токены_промпта) → токены_ответа`. Всё что она знает закодировано в весах, зафиксированных на момент обучения. Дата знаний T-lite-it-1.0 — приблизительно весна/лето 2024 года. Любой факт позже этой даты (новый пакет, изменившийся API, актуальный регламент) для модели не существует.

Fine-tuning изменяет веса модели и меняет её поведение, но не делает её достоверно осведомлённой о конкретных корпоративных фактах. Попытка «залить знания» через fine-tuning даёт модель, которая может казаться уверенной, но галлюцинирует в деталях с высокой вероятностью — известный феномен, задокументированный в нескольких академических работах 2024–2025.

Правильный способ думать о разделении задач:

```
Fine-tuning  → меняет КАК модель отвечает
               (формат, стиль, язык, поведение)
               
RAG          → даёт ЧТО модель знает прямо сейчас
               (факты, документы, актуальные данные)
```

RAG (Retrieval-Augmented Generation) решает проблему иначе: знания хранятся внешне, извлекаются при каждом запросе, подаются в контекст. Модель отвечает не «из головы», а опираясь на предоставленный документ. Задача LLM в RAG-системе трансформируется: не «вспомни факт», а «синтезируй ответ из прочитанных абзацев» — задача, с которой трансформеры справляются принципиально лучше.

RAG — enterprise use case №1 для LLM, обогнавший fine-tuning и prompt engineering как самостоятельную технику.

---

### 3.2 Конвейер RAG: шесть компонентов и два production-слоя

Полный production RAG-конвейер состоит из шести обязательных компонентов и двух дополнительных production-слоёв:

```
ИНДЕКСАЦИЯ (offline, однократно или по расписанию):
  Документы → [Loader] → [Splitter/Chunker] → [Embedder] → [Vector Store]

ЗАПРОС (online, каждый вызов):
  Запрос → [Embedder] → [Retriever] → (опционально: [Reranker]) → [Generator/LLM]

PRODUCTION СЛОИ (параллельно):
  → [Evaluator] : faithfulness, relevance, recall
  → [Tracer]    : Langfuse / WandB Weave
```

Каждый компонент влияет на итоговое качество независимо. Распространённая ошибка: менять только генератор (заменить модель на более сильную), игнорируя качество ретривала. Если система возвращает нерелевантные чанки — даже самая мощная модель не ответит правильно.

---

Пример, как мы собираем **Production RAG** в реальном энтерпрайзе.

Вся архитектура строится на on-premise стеке (локальные эмбеддинги, локальный Cross-Encoder, векторная БД).

```python
import os

# 1. Базовые компоненты LCEL (LangChain Expression Language)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 2. Интеграции для векторного поиска (Offline и Online слои)
# Мы используем нативные пакеты интеграций
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 3. Инструменты для Reranker (Слой повышения точности)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 4. Observability (Слой мониторинга и телеметрии)
from langfuse.callback import CallbackHandler

# 5. Провайдер LLM (Локальный инференс)
from langchain_ollama import ChatOllama

# =====================================================================
# ШАГ 1: ИНИЦИАЛИЗАЦИЯ ИНФРАСТРУКТУРЫ
# =====================================================================

# Мы инициализируем локальную модель для генерации ответов. 
# Устанавливаем temperature=0, потому что в RAG-системах нам нужны 
# сухие факты из документации, а не креативные фантазии нейросети.
llm = ChatOllama(model="qwen2.5:8b", temperature=0)

# Мы выбираем энтерпрайз-модель эмбеддингов BAAI/bge-m3. 
# Почему именно её? Она мультиязычная, отлично понимает русский текст, 
# английский технический жаргон и куски программного кода. 
# Эмбеддер превратит наш текст в плотный числовой массив (вектор).
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Мы поднимаем in-memory инстанс векторной базы Qdrant для тестов.
# В реальном production здесь будет URL до нашего Kubernetes-кластера 
# с Qdrant (например, url="http://qdrant.database.svc.cluster.local:6333").
client = QdrantClient(":memory:")

# Мы явно задаем размерность вектора (для bge-m3 это 1024) 
# и метрику вычисления дистанции (Cosine Similarity). 
# Это нужно, чтобы БД понимала, как именно сравнивать векторы между собой.
client.create_collection(
    collection_name="enterprise_docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# =====================================================================
# ШАГ 2: ИНДЕКСАЦИЯ БАЗЫ ЗНАНИЙ (Offline слой)
# =====================================================================

# Мы связываем векторную базу данных с нашей моделью эмбеддингов.
vector_store = QdrantVectorStore(
    client=client,
    collection_name="enterprise_docs",
    embedding=embeddings,
)

# Мы загружаем сырые данные в базу. 
# Под капотом LangChain пропустит каждый текст через эмбеддер (bge-m3), 
# получит векторы и сложит их в Qdrant вместе с исходным текстом.
# В реальном конвейере эти данные предварительно нарезаются (Chunking).
vector_store.add_texts([
    "Политика ИБ: пароли должны быть не менее 12 символов.",
    "Инфраструктура: Kubernetes кластер деплоится через Terraform.",
    "Секреты: Пароли от production БД хранятся в HashiCorp Vault.",
    "Офис: Кофемашина на третьем этаже работает до 20:00."
])

# =====================================================================
# ШАГ 3: ПОДГОТОВКА СЛОЯ ЗАПРОСА С RERANKER (Online слой)
# =====================================================================

# 3.1 Базовый Retriever (Грубый и быстрый поиск)
# Мы настраиваем базу возвращать 10 документов, которые БЛИЖЕ ВСЕГО по вектору.
# Проблема векторного поиска: он ищет смысловую похожесть, но не всегда точный ответ.
# Если мы запросим "пароли", он может принести и "политику ИБ", и "секреты БД".
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 3.2 Модель Reranker (Умный фильтр / Cross-Encoder)
# Мы инициализируем Cross-Encoder. В отличие от быстрых эмбеддингов, 
# эта модель работает медленно, но очень умно. Она берет вопрос пользователя,
# берет найденный документ, и оценивает их связку от 0.0 до 1.0.
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# Мы приказываем реранкеру: "Прочитай все 10 документов от векторной БД 
# и пропусти дальше только 3 самых релевантных документа".
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# 3.3 Сборка Компрессионного Ретривера
# Мы объединяем грубый поиск (base_retriever) и умный фильтр (reranker) в единый объект.
# Теперь для LangChain это выглядит как один обычный ретривер, 
# но внутри него работает сложный двухэтапный конвейер.
smart_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=base_retriever
)

# =====================================================================
# ШАГ 4: СБОРКА RAG ПАЙПЛАЙНА (LCEL) И ТЕЛЕМЕТРИЯ
# =====================================================================

# Мы настраиваем Langfuse Tracer. В энтерпрайзе мы ОБЯЗАНЫ логировать каждый шаг:
# сколько времени занял векторный поиск, что отрезал реранкер, какой промпт ушел в LLM.
# Без этого отладка галлюцинаций в production превращается в гадание на кофейной гуще.
langfuse_handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000" # Адрес нашего on-premise сервера телеметрии
)

# Мы задаем строгий системный промпт. Главное правило RAG: 
# приказать модели отвечать ТОЛЬКО на основе контекста.
prompt = ChatPromptTemplate.from_template(
    "Ты DevSecOps ассистент. Ответь на вопрос опираясь строго на предоставленный контекст.\n\n"
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}"
)

# LangChain-ретриверы возвращают список объектов (List[Document]).
# LLM не умеет читать объекты, ей нужен обычный текст. 
# Мы пишем небольшую утилиту, которая склеивает тексты найденных документов через двойной перенос строки.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Мы собираем финальный пайплайн конвейера. Читаем сверху вниз:
rag_pipeline = (
    {
        # Ветка 1: Берем вопрос пользователя -> Ищем через smart_retriever -> 
        # Превращаем результат из List[Document] в строку через RunnableLambda(format_docs)
        "context": smart_retriever | RunnableLambda(format_docs), 
        
        # Ветка 2: Просто пробрасываем вопрос пользователя дальше без изменений
        "question": RunnablePassthrough()
    }
    # Полученный словарь с ключами {context, question} пробрасываем в шаблон промпта
    | prompt
    # Готовый текст промпта отправляем в локальную нейросеть
    | llm
    # Извлекаем сырой текст из ответа (убираем метаданные токенов)
    | StrOutputParser()
)

# =====================================================================
# ШАГ 5: ВЫПОЛНЕНИЕ ЗАПРОСА
# =====================================================================

print("=== Запуск Production RAG Pipeline ===")
query_text = "Где физически хранятся пароли от production баз данных?"

# Мы запускаем конвейер через метод invoke.
# Обратите внимание: мы передаем конфигурацию (config) с обработчиками событий (callbacks).
# Именно в этот момент Langfuse начинает записывать весь граф вызовов.
result = rag_pipeline.invoke(
    query_text,
    config={
        "callbacks": [langfuse_handler], 
        "run_name": "Vault_Infrastructure_Query" # Даем имя запуску для удобного поиска в логах
    }
)

print(f"\nОтвет системы:\n{result}")

```

---

### 3.3 Чанкинг: разбивка документов на фрагменты

Чанк — минимальная единица хранения в векторной базе. Каждый чанк получает свой вектор и независимо участвует в поиске.

**Почему нельзя индексировать целый документ:**
- Вектор целого документа усредняет все его темы → плохая точность поиска
- Контекстное окно LLM ограничено → нельзя передать 100-страничный документ целиком
- Разные части документа отвечают на разные вопросы

Передача 100-страничного документа целиком в каждый промпт — это огромная переплата за токены, кратный рост latency и риск эффекта «lost in the middle» (когда модель игнорирует факты из середины текста).

**Параметры чанкинга и их влияние:**

`chunk_size` — целевое количество символов (или токенов) в чанке. Стандарт в 2026: 400-800 токенов. Слишком маленький → потеря контекста (предложение без абзаца неоднозначно). Слишком большой → один чанк охватывает несколько тем → плохая точность эмбеддинга.

`chunk_overlap` — количество символов, копируемых из конца предыдущего чанка в начало следующего. Цель: предотвратить ситуацию, когда ключевая мысль оказывается "разрезана" на границе. Стандарт: 10-15% от chunk_size.

`separators` — иерархия разделителей для `RecursiveCharacterTextSplitter`. Сначала пробует разбить по `\n\n` (абзацы), если чанк всё равно длинный — по `\n`, затем по пробелу, в крайнем случае по символу. Это семантически грамотнее, чем просто резать по длине.

**Parent-Child чанкинг** — техника (2025): в векторной базе хранятся маленькие "дочерние" чанки (~100 токенов) для высокой точности поиска, но при попадании извлекается "родительский" чанк (вся страница). Модель получает богатый контекст при высокой точности поиска.

**Семантический чанкинг** — исследование Databricks (2025) показало улучшение recall на 9% по сравнению с fixed-size подходом. Суть: граница чанка определяется не длиной, а семантическим разрывом (измеряемым через эмбеддинги соседних предложений).

---

### 3.4 Embedding-модели: вектор как геометрическое представление смысла

Embedding-модель — это нейронная сеть, преобразующая произвольный текст в вектор фиксированной размерности. Фундаментальное свойство: семантически близкие тексты → геометрически близкие векторы в пространстве.

Это свойство не тривиально: оно означает, что вектор фразы "как установить FreeIPA" близок к вектору фрагмента документации "процедура развёртывания службы каталогов", хотя они не имеют общих слов.

**BGE-M3** (BAAI General Embedding M3) — это флагманская текстовая эмбеддинг-модель нового поколения от Пекинского института искусственного интеллекта (BAAI), разработанная специально для построения передовых RAG-систем (Retrieval-Augmented Generation) и поисковых движков.
Буква M3 в названии означает три ее главные особенности: Multi-linguality (мультиязычность), Multi-granularity (многогранность длин текстов) и Multi-functionality (универсальность методов поиска).

**Ключевые характеристики embedding-моделей:**

*Размерность вектора* — число компонентов выходного вектора. Больше не значит лучше: BGE-M3 с 1024 размерностями превосходит некоторые модели с 3072. Влияет на размер индекса и скорость поиска.

*Максимальная длина контекста* — сколько токенов входного текста модель обрабатывает. У большинства моделей — 512-8192 токенов. Чанки длиннее этого порога молча обрезаются.

*Multilingual-поддержка* — критично для русскоязычных корпусов. nomic-embed-text оптимизирован преимущественно для английского. BGE-M3 обучен на 100+ языках и демонстрирует сопоставимое качество на русском и английском без потерь.

**BGE-M3 (BAAI/bge-m3, Apache 2.0)** — уникальная модель, производящая три типа векторов из одного прогона:

*Dense vector* — обычный эмбеддинг для семантического поиска по косинусному сходству. Покрывает парафразы и концептуальные совпадения.

*Sparse vector* — разреженный вектор, аналогичный TF-IDF, но лексически обученный. Покрывает точные совпадения слов лучше чем dense.

*Multi-vector (ColBERT)* — каждый токен получает отдельный вектор, сходство вычисляется через MaxSim по всем парам токенов. Точнее всего для сложных многоаспектных запросов, но дороже в хранении и вычислении.

Возможность комбинировать все три типа в одном запросе через Qdrant делает BGE-M3 стандартом для production on-premise систем в 2026.

---

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

---

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

### 3.8 Оценка RAG-системы: декомпозиция метрик и LLM-as-a-judge

Фундаментальная ошибка начинающих инженеров — оценивать RAG-систему как "чёрный ящик", ориентируясь только на качество итогового ответа пользователя. Итоговый ответ — это продукт слияния двух независимых подсистем: *Retrieval* (поиск контекста) и *Generation* (синтез ответа).

Если оценивать пайплайн монолитно, невозможно локализовать причину сбоя:

* **Слабый ретривал + Сильная LLM** → Модель красиво и грамматически безупречно синтезирует ответ, но использует нерелевантный контекст (или начинает фантазировать, пытаясь логически закрыть "дыры" в предоставленных данных).
* **Сильный ретривал + Слабая LLM** → Векторная база отдала идеальные документы, но модель не смогла их правильно прочитать, проигнорировала ключевые факты или нарушила требуемый формат JSON.

Для точной диагностики индустрия (через фреймворки вроде RAGAS и DeepEval) стандартизировала три независимые метрики.

**1. Faithfulness (Верность источнику / Отсутствие галлюцинаций)**
Оценивает исключительно работу LLM-генератора. Метрика отвечает на вопрос: *Основан ли итоговый ответ строго на фактах из предоставленного контекста?*

* **Механика:** LLM-судья разбивает сгенерированный ответ на атомарные утверждения (claims). Затем она проверяет, имеет ли каждое утверждение прямое логическое подтверждение в `retrieved`-чанках.
* **Формула:** `(Количество подтвержденных утверждений) / (Общее количество утверждений в ответе)`.
* **Нюанс:** Ответ может быть абсолютно неверным по меркам реального мира, но получить Faithfulness 1.0 (100%), если он честно пересказал устаревший документ, который ему подсунул ретривер. Галлюцинация в контексте RAG — это строго добавление фактов, которых не было во входящих чанках.

**2. Context Recall (Полнота поиска)**
Оценивает работу ретривера. Метрика отвечает на вопрос: *Смог ли поиск найти всю информацию, необходимую для идеального ответа?*

* **Механика:** Требует наличия `Ground Truth` (эталонного ответа, заранее написанного человеком). Эталонный ответ разбивается на утверждения. Судья проверяет, можно ли найти подтверждение каждому эталонному факту в найденных базой чанках.
* **Формула:** `(Утверждения из эталона, найденные в чанках) / (Все утверждения в эталоне)`.
* **Смысл:** Низкий Recall означает, что векторный поиск "промахнулся" и не достал критически важные куски документации. Это сигнал к тюнингу чанкинга, смене embedding-модели или внедрению гибридного поиска (BM25).

**3. Context Precision (Точность ранжирования)**
Также оценивает ретривер, но фокусируется на порядке сортировки результатов. Метрика отвечает на вопрос: *Находятся ли самые полезные чанки на самых верхних позициях выдачи?*

* **Механика:** Использует математику `Precision@k`. Если релевантный чанк оказался на 1-м или 2-м месте, штрафа нет. Если он оказался на 10-м месте — метрика экспоненциально падает.
* **Смысл:** LLM сильно подвержены проблеме "Lost in the middle" (потеря внимания к середине длинного промпта). Если ретривер достал правильный документ, но засунул его в конец списка из 15 чанков, языковая модель с высокой вероятностью его проигнорирует. Нам важно не просто найти документ, но и поставить его первым.
* *(Примечание: Метрика, оценивающая долю полезных чанков к бесполезному "мусору" в контексте, выделяется отдельно и называется **Context Utilization**).*

**Автоматизация (LLM-as-a-judge)**
Считать эти метрики вручную глазами асессоров на сотнях тестов — долго и дорого. Фреймворки RAGAS и DeepEval автоматизируют процесс через парадигму `LLM-as-a-judge`. Они вызывают сильную "модель-судью" (например, GPT-4o), которая по строгим внутренним промптам разбивает тексты на утверждения, сверяет их и возвращает детерминированные числовые скоры для CI/CD пайплайна.

---

## БЛОК 4: ГРАФЫ ЗНАНИЙ

### 4.1 Структурное ограничение RAG

RAG отлично работает для вопросов типа "что сказано о X в документации". Но существует класс вопросов, которые требуют не извлечения фрагмента, а обхода связей:

- "Через каких поставщиков наши три крупнейших клиента закупают лицензионное ПО?"
- "Кто из сотрудников имеет доступ ко всем системам, к которым имеет доступ Иванов?"
- "Какие риски объединяют проекты в портфеле?"

Эти вопросы требуют агрегации информации, распределённой по сотням документов, и её объединения через сущностные связи. Векторный поиск возвращает семантически похожие фрагменты — но не способен пройти по цепочке отношений.

---

### 4.2 Граф знаний: формальное определение

Граф знаний — ориентированный граф `G = (V, E)`, где:
- `V` — множество узлов-сущностей (организации, люди, продукты, концепции)
- `E` — множество рёбер-отношений, каждое ребро `e = (subject, predicate, object)`

Эта тройка называется RDF-триплет (Resource Description Framework) — стандарт W3C. Пример: `(Ромашка, закупила, Microsoft Office)`.

Ключевое свойство: граф формализует неявные связи в тексте. Документ о закупке содержит эту тройку в виде предложения на естественном языке. LLM извлекает её и делает явной — теперь можно задать структурный запрос: "найди все организации, связанные с Microsoft Office отношением 'закупила'".

---

### 4.3 Онтология и схема vs. безсхемное извлечение

**Онтология** (в контексте баз знаний) — это строгий архитектурный контракт графа: формальное, машиночитаемое описание типов сущностей (узлов), допустимых отношений (рёбер) и их атрибутов.

Пример контракта:
- `Organization` → может иметь отношение `является_партнёром` → только с `Organization`
- `Organization` → может иметь отношение `производит` / `потребляет` → только с `Product`
- `Product` → не может напрямую ссылаться на другой `Product` через `является_партнёром`

В production-системах онтология выполняет ту же защитную роль, что строгая DDL-схема (Data Definition Language) в реляционных SQL-базах: гарантирует предсказуемую форму данных, по которой можно строить детерминированные аналитические запросы.

#### Режим 1: Безсхемное извлечение (Schemaless)

По умолчанию `LLMGraphTransformer` из `langchain_experimental` работает в режиме **Schemaless**: языковая модель получает полную свободу и «на лету» придумывает названия для объектов и связей.

**Плюсы:**
- Быстрое прототипирование
- Разведочный анализ (Exploratory Data Analysis) незнакомого корпуса без предварительного проектирования схемы

**Минусы в production:**
- **Schema Drift** — схема размывается со временем, каждый запуск даёт разные результаты
- **Entity/Relation Resolution** — синонимы для одного факта множатся бесконтрольно:

```
Первый запуск:   (Ромашка) --[купила]--> (Microsoft Office)
Второй запуск:   (Ромашка) --[приобрела]--> (Microsoft Office)
Третий запуск:   (Ромашка) --[совершила_покупку]--> (Microsoft Office)
```

Аналитики вынуждены писать монструозные запросы, пытаясь предугадать все «фантазии» LLM.

#### Режим 2: Извлечение по схеме (Schema-guided) — рекомендуется для production

`LLMGraphTransformer` принимает два ограничивающих параметра:

```python
LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Organization", "Person", "Software"],
    allowed_relationships=["USES", "DEVELOPS", "WORKS_AT"]
)
```

Это фундаментально меняет механику работы модели: задача из **открытой генерации** превращается в **строгую классификацию**. LLM не придумывает сущности и связи — она проецирует факты из текста в заданные вами рамки.

**Что это даёт:**
- Предсказуемые и воспроизводимые результаты при каждой индексации
- Возможность написать Cypher-запросы (декларативный язык описания шаблонов) заранее, до появления данных:

```cypher
MATCH (p:Person)-[:WORKS_AT]->(o:Organization)
RETURN p.name, o.name
```

- Уверенность, что граф не «замусорится» неожиданными типами рёбер при ежедневной индексации новых документов

---

### 4.4 NetworkX как инструмент демонстрации и его производственные ограничения

NetworkX — Python-библиотека для работы с графами. Хранит граф в памяти как словари смежности. Поддерживает: `DiGraph` (ориентированный), `MultiDiGraph` (несколько рёбер между одной парой узлов), все классические алгоритмы (BFS, DFS, shortest paths, centrality, community detection).

Производственные ограничения:
- Всё хранится в RAM — при ~1M рёбер начинаются проблемы с памятью
- Нет персистентности — при рестарте граф нужно перестраивать
- Нет ACID-транзакций (Atomicity, Consistency, Isolation, Durability) и изоляции
- Нет встроенного API для запросов на специализированном языке (Cypher и т.д.)

Удобен для базовых простых проектов: быстро запускается, не требует инфраструктуры. Но не масштабируется: нет персистентности, нет транзакций, нет языка запросов, ~1M рёбер — уже проблема с памятью.

**Для production-масштабов:**
- **Neo4j** — самая зрелая граф-БД, язык запросов Cypher, ACID, полноценная интеграция с LangChain через `Neo4jVector` и `Neo4jGraph`
- **Memgraph** — совместим с Cypher, in-memory с persistence, значительно быстрее Neo4j для аналитических запросов
- **GraphRAG** (Microsoft Research, 2024) — полный пайплайн: LLM извлекает граф из корпуса, применяет Leiden Community Detection для обнаружения тематических кластеров, строит двухрежимный поиск (Local: точные факты о сущностях; Global: аналитические вопросы по всему корпусу). Open-source.

---

### 4.5 Гибридный подход: RAG + Граф знаний

Оба инструмента дополняют, а не заменяют друг друга:

```
Тип вопроса              Инструмент
────────────────────────────────────────────────────────────
"Что говорит документ    RAG
о процедуре X?"

"Кто такой Иванов?"      RAG (персональное дело, биография)

"С кем связан Иванов     Граф знаний (обход рёбер)
 по всем проектам?"

"Какие риски у           RAG + Граф: RAG находит описания
 проектов Иванова?"       рисков, Граф связывает их с Ивановым

"Глобальный тренд        GraphRAG Global Search
 по всему корпусу?"       (community-based summary)
```

Практический совет: начинать с RAG, добавлять граф при появлении аналитических запросов с обходом связей.

---

## БЛОК 5: АГЕНТНЫЕ СИСТЕМЫ И LANGGRAPH

### 5.1 Ключевое различие: цепочка vs. агент

Цепочка (chain) — статический граф выполнения. Маршрут вычисляется при разработке. Цепочка `документ → суммаризация → классификация` всегда выполняется именно в этом порядке, независимо от содержимого документа.

Агент — динамическая система, где маршрут выполнения определяется во время работы на основе наблюдений. После каждого шага агент оценивает текущее состояние и решает: продолжить, вызвать инструмент, запросить пользователя, завершить.

Это различие фундаментальное: цепочка — программа с фиксированным потоком управления; агент — программа с циклом обратной связи.

Согласно "State of Agent Engineering Survey" от LangChain (2025, 1340 респондентов): 57% AI-инженеров уже имеют агентов в production. Однако для остальных главным барьером (32% опрошенных) стала не нехватка возможностей LLM, а проблема качества и предсказуемости — сложность обеспечения надёжной работы агента без сбоев на масштабе.

---

### 5.2 Паттерн ReAct: теоретическая основа агентов

ReAct (Reasoning + Acting) — LLM чередует рассуждение (Thought) и действие (Action) в едином потоке токенов.

```
Мысль:   Пользователь спросил о статусе сервера prod-1.
         Мне нужно проверить текущий статус через инструмент.
         
Действие: get_server_status(server_id="prod-1")

Наблюдение: {"status": "degraded", "cpu": 94, "disk": 98}

Мысль:   Сервер в деградированном состоянии. CPU 94%, диск 98%.
         Нужно предупредить пользователя и предложить действия.
         
Ответ:   Сервер prod-1 в деградированном состоянии...
```

В реализации через Tool Calling (функциональный вызов) шаг "Действие" материализуется как структурированный JSON-вызов, а не свободный текст. LLM возвращает `{"tool": "get_server_status", "args": {"server_id": "prod-1"}}` — фреймворк исполняет реальную Python-функцию и возвращает результат обратно в контекст.

**Разрыв между "думает" и "делает"**: именно здесь безопасность. LLM не может напрямую выполнить код или изменить данные — только сгенерировать запрос на выполнение. Фреймворк решает: допускать ли этот вызов (валидация аргументов, права, sandbox).

---

### 5.3 LangGraph: State Machine для агентов

LangGraph переосмысляет агента как конечный автомат (State Machine). Это принципиально меняет модель разработки: вместо чёрного ящика, где фреймворк сам решает что и когда делать, разработчик явно описывает состояние системы, узлы обработки и правила переходов между ними. Поведение агента становится полностью прозрачным и воспроизводимым.

**State** — центральный объект типа `TypedDict`, путешествующий через граф. Каждый узел получает State на вход и возвращает словарь с обновлениями. LangGraph применяет эти обновления к State через reducer-функции.

Важно понимать: узел не перезаписывает State целиком — он возвращает только те поля, которые изменились. LangGraph сам мержит изменения в текущий State. Это означает, что разные узлы могут отвечать за разные поля State независимо друг от друга.

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer задаётся аннотацией
    user_id: str                             # обычное поле — перезапись
    error_count: int                         # счётчик — тоже перезапись
```

**`add_messages`** — специальный reducer для поля `messages`. Вместо перезаписи новые сообщения добавляются к существующим (append). Это фундаментально для агентов: история диалога накапливается, а не стирается на каждом шаге.

Без `add_messages`:
```python
State.messages = [new_message]  # только одно сообщение — вся история уничтожена
```

С `add_messages`:
```python
State.messages = State.messages + [new_message]  # история накапливается
```

Зачем это нужно: LLM принимает решения на основе полной истории диалога. Если на каждом шаге агент видит только одно последнее сообщение, он не знает контекста предыдущих вызовов инструментов и результатов наблюдений. `add_messages` гарантирует, что модель всегда работает с полным контекстом — вплоть до явной очистки через `RemoveMessage`.

**Node** — обычная Python-функция с единственным контрактом: принимает `State`, возвращает `dict` с обновлениями тех полей, которые изменились. Узел не знает о своём положении в графе и не управляет переходами — он просто выполняет свою логику и возвращает результат.

```python
def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages добавит к истории

def tool_node(state: AgentState) -> dict:
    # выполняем инструмент, возвращаем только изменённые поля
    result = execute_tool(state["messages"][-1])
    return {"messages": [result], "error_count": 0}
```

Такое разделение ответственности позволяет тестировать каждый узел изолированно — просто передавая ему нужный State и проверяя возвращаемый словарь.

**Edge** — соединяет узлы. Два типа:

- **Обычное ребро**: детерминированный переход, всегда из A в B. Используется там, где следующий шаг не зависит от результата текущего.

- **Conditional edge**: принимает функцию-роутер, которая смотрит в State и возвращает строку с именем следующего узла. Именно здесь живёт логика принятия решений агента:

```python
from langgraph.graph import END  # Не забудьте указать импорт

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:      # модель хочет вызвать инструмент
        return "tools"
    return END                       # ответ готов, выходим из графа

builder.add_conditional_edges("agent", should_continue)
```

Conditional edge позволяет реализовать цикл `think → act → observe → think` без какого-либо специального синтаксиса — просто через условный возврат в узел `agent`.

**Compilation**: `builder.compile()` преобразует описание графа в исполняемый объект с методами `invoke`, `stream`, `batch` и `astream_events`. На этапе компиляции LangGraph статически проверяет корректность всей топологии:

- достижимость каждого объявленного узла из стартового
- отсутствие висячих рёбер (ребро ведёт в несуществующий узел)
- корректность имён в `add_conditional_edges`
- наличие хотя бы одного пути до `END`

Если что-то не так — ошибка возникает при компиляции, а не во время выполнения на продакшне. Это делает LangGraph значительно надёжнее динамических агентных фреймворков, где топологические ошибки обнаруживаются только под нагрузкой.

---

### 5.4 ToolNode: стандартизированное выполнение инструментов

`ToolNode` из `langgraph.prebuilt` — готовый узел, который берёт на себя весь цикл выполнения инструментов. Когда LLM решает вызвать инструмент, она не вызывает его напрямую — она возвращает структурированный JSON с именем функции и аргументами. Кто-то должен этот JSON прочитать, найти реальную Python-функцию, вызвать её и упаковать результат обратно в формат, который LLM поймёт. Именно этим занимается `ToolNode`.

**Что происходит внутри при каждом вызове:**

1. Извлекает список `tool_calls` из последнего `AIMessage` в `state["messages"]`. Один вызов LLM может породить несколько `tool_calls` одновременно — например, когда модель решает параллельно запросить данные из нескольких источников.

2. Находит соответствующую Python-функцию среди зарегистрированных инструментов по полю `name` в каждом `tool_call`. Поиск идёт по словарю `{tool.name: tool}`, который `ToolNode` строит при инициализации.

3. Вызывает функцию с аргументами из поля `args`. Аргументы приходят как словарь — `ToolNode` распаковывает их через `**kwargs`.

4. Упаковывает результат в `ToolMessage` с обязательным полем `tool_call_id`, которое связывает ответ инструмента с конкретным запросом LLM. Без правильного `tool_call_id` модель не поймёт, какой результат к какому вызову относится — это требование протокола tool calling.

5. Возвращает `{"messages": [tool_message]}` — `add_messages` добавит ответ инструмента к истории, и на следующей итерации LLM увидит полный контекст: свой запрос и результат его выполнения.

**Что пришлось бы писать вручную без `ToolNode`:**

```python
def manual_tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        # найти функцию по имени
        tool_fn = tools_by_name.get(tool_call["name"])
        if tool_fn is None:
            result = f"Error: unknown tool {tool_call['name']}"
        else:
            try:
                result = tool_fn.invoke(tool_call["args"])
            except Exception as e:
                result = f"Error: {str(e)}"

        # упаковать с правильным tool_call_id
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_messages}
```

`ToolNode` инкапсулирует именно этот boilerplate, добавляя сверху две вещи, которые сложно реализовать правильно вручную.

**Параллельное выполнение через `asyncio.gather`:** если LLM вернула несколько `tool_calls`, `ToolNode` в async-режиме запускает их одновременно. Это критично для производительности: агент, запрашивающий данные из трёх источников, получит результаты за время самого медленного вызова, а не за сумму всех трёх.

**Обработка ошибок на уровне отдельного вызова:** если один инструмент упал с исключением, `ToolNode` упакует ошибку в `ToolMessage` и продолжит выполнение остальных. LLM получит сообщение об ошибке как обычный ответ инструмента и сможет скорректировать своё поведение — вместо того чтобы весь агент упал с необработанным исключением.

```python
# Инициализация — один раз при сборке графа
tool_node = ToolNode(tools=[search_docs, get_employee, calculate_cost])

builder.add_node("tools", tool_node)
builder.add_edge("tools", "agent")  # после выполнения — всегда обратно к LLM
```

---

### 5.5 Tool Calling: механика на уровне протокола

Tool Calling в OpenAI API-совместимых эндпоинтах (включая Ollama с Qwen-3) работает так:

**Запрос:**
```json
{
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_employee_info",
        "description": "Возвращает информацию о сотруднике по ID",
        "parameters": {
          "type": "object",
          "properties": {"employee_id": {"type": "string"}},
          "required": ["employee_id"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Ответ при намерении вызвать инструмент:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_employee_info",
          "arguments": "{\"employee_id\": \"EMP001\"}"
        }
      }]
    }
  }]
}
```

LangChain парсит этот ответ в `AIMessage` с заполненным полем `tool_calls`. `ToolNode` выполняет функцию и возвращает `ToolMessage` с `tool_call_id="call_abc123"` — протокол требует привязки ответа к вызову.

`@tool` декоратор делает три вещи: генерирует JSON Schema из сигнатуры функции и её аннотаций, использует `__doc__` как поле `description` (именно от качества docstring зависит, насколько точно агент выбирает инструмент), регистрирует функцию для последующего выполнения.

---

### 5.6 `draw_mermaid()` и визуализация архитектуры

`app.get_graph().draw_mermaid()` генерирует Mermaid-синтаксис — DSL для описания диаграмм, который рендерится в браузере через mermaid.live или встраивается в Jupyter через IPython.display.

Это не просто визуальная фича: визуализация — инструмент верификации. Граф агента с 5+ узлами и условными рёбрами сложно держать в голове. `draw_mermaid()` даёт:
- Проверку что логика "agent → tools → agent" замкнута
- Обнаружение узлов без входящих рёбер (недостижимые)
- Обнаружение узлов без исходящих рёбер (потенциальные тупики)
- Документацию для code review

Реальная production-практика: коммитить PNG-снимок графа вместе с кодом, обновлять при изменении топологии.

---

### 5.7 Масштабирование: проблема 20+ инструментов

При большом числе инструментов точность Tool Calling деградирует. Причина: описания инструментов занимают значительную часть контекстного окна, а LLM сложнее выбирать из длинного списка.

Эмпирические данные (LangChain Engineering Blog, 2025): при 5 инструментах — точность выбора ~95%, при 20 инструментах — ~78%, при 50 инструментах — ~61%.

**Двухуровневая маршрутизация** решает проблему:

```
Уровень 1: Router Agent
  Минимальный контекст, только категории инструментов
  "Это запрос про HR, финансы, или IT?"
  Выбирает из 3-5 категорий → высокая точность

Уровень 2: Specialist Agent per category
  Знает только инструменты своей категории (5-7 шт.)
  Полный контекст категории
  → высокая точность выбора конкретного инструмента
```

Это мульти-агентная архитектура с разделением ответственности. LangGraph поддерживает её через `Command` — механизм передачи управления между подграфами с сохранением checkpointing.

---

## БЛОК 6: ПРОДВИНУТЫЕ КЕЙСЫ И ОГРАНИЧЕНИЯ

### 6.1 Контекстное окно как фундаментальное ограничение

Контекстное окно (context window) — максимальное количество токенов, которое трансформер обрабатывает за один forward pass. Это не ограничение памяти в обычном смысле — это следствие архитектуры attention mechanism.

В механизме Self-Attention каждый токен взаимодействует с каждым другим токеном: вычислительная сложность O(n²) по количеству токенов. При n=32 000 (Qwen-3 8B): 32 000² = 1 024 000 000 операций на один forward pass. При n=128 000 (Llama-3.1): 16 миллиардов операций. GPU-память квадратично растёт с контекстом.

**KV-кеш и его роль в деградации качества.** При авторегрессивной генерации (токен за токеном) вычисленные Key и Value матрицы для предыдущих токенов кешируются. При полном контексте KV-кеш занимает всю доступную VRAM сверх весов модели. Это физическая причина OOM в агентных сценариях с длинной историей.

**"Lost in the middle" эффект** (Liu et al., 2023, Стэнфорд): при длинных контекстах (>10 000 токенов) LLM значительно хуже использует информацию из середины контекста по сравнению с началом и концом. Эксперименты показали: при задаче нахождения релевантного факта из 20 документов точность падает с ~80% (при факте в начале/конце) до ~30% (при факте в середине). Это не баг конкретной модели — это системное свойство attention-архитектуры.

**Практическое следствие для агентов:** длинная история tool_calls и tool_results в messages деградирует качество принятия решений агентом. Не потому что модель "забыла" — она технически всё видит, но внимание распределяется неравномерно.

---

### 6.2 Стратегии управления памятью в LangGraph

**Sliding Window** — самая простая стратегия: удалять сообщения старше N. Реализуется через `RemoveMessage`:

```python
messages = state["messages"]
# Идентифицируем старые сообщения (кроме SystemMessage)
to_delete = [RemoveMessage(id=m.id) for m in messages[:-10]
             if not isinstance(m, SystemMessage)]
return {"messages": to_delete}
```

Критический нюанс: `SystemMessage` нельзя удалять никогда. Системный промпт содержит инструкции модели: её роль, доступные инструменты, правила поведения. Потеря SystemMessage = модель "забыла" кто она. Результат непредсказуем.

`trim_messages` из `langchain_core.messages` предоставляет параметр `include_system=True` именно для этого.

**Явная суммаризация** — дополнительный узел в графе LangGraph:

```
[agent_node] → условие: если len(messages) > 20
                              │ да
                         [summarize_node]
                         "Сожми историю в 3 предложения"
                              │
                         Сохраняем summary в State
                         Удаляем старые messages
                         Добавляем SystemMessage с summary
                              │
		                 [agent_node] продолжает с summary
```

Компромисс: суммаризация стоит LLM-вызов (+500ms задержки), но сохраняет семантическую связность диалога, которую Sliding Window теряет.

**Разделение данных и контекста** — архитектурный паттерн:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # только текст диалога
    tool_results: dict                       # raw JSON от инструментов
    summary: str                             # накопленный контекст
```

Вместо того чтобы класть весь JSON-ответ от инструмента в `messages` (100-10 000 токенов), кладём краткий синтезированный текст ("SQL вернул 1000 записей, топ-3 по объёму: X, Y, Z"), а полные данные храним в `tool_results`. Это принципиально снижает рост контекста.

---

### 6.3 GraphRecursionError и fallback-архитектура

`recursion_limit=25` — это не тайм-аут, а счётчик шагов графа. Каждый переход от одного узла к другому инкрементирует счётчик. При достижении лимита — `GraphRecursionError`.

Почему 25 — разумный дефолт: нормальный агентный запрос занимает 3-10 шагов. При 25 шагах агент либо выполнил очень сложную задачу, либо зациклился. Второе гораздо вероятнее в production.

Повышение `recursion_limit` — не решение проблемы зацикливания, только откладывает её.

**Счётчик ошибок в State как паттерн надёжности:**

```python
class ResilientState(TypedDict):
    messages: Annotated[list, add_messages]
    error_count: int  # число consecutive ошибок инструментов
```

При ошибке инструмента: `error_count += 1`. При успехе: `error_count = 0`. При `error_count >= 3` — conditional edge направляет в `fallback_node` вместо `agent_node`.

`fallback_node` не должен молча возвращать пустоту. Best practice: возвращать пользователю конкретное сообщение с контактами ("Автоматическая обработка не удалась, обратитесь в поддержку: ext. 1234") и логировать событие в observability-систему для последующего анализа.

---

### 6.4 Human-in-the-Loop: паттерн прерывания

LangGraph поддерживает первоклассное прерывание: `compile(interrupt_before=["node_name"])`. Граф выполняется до указанного узла, сохраняет полный State в чекпоинтере, возвращает управление приложению.

Это критично для enterprise-сценариев с необратимыми действиями:

```
Агент принял решение: отправить email 500 клиентам.
→ INTERRUPT перед "send_email_node"
→ Оператор просматривает State: список получателей, текст письма
→ Оператор подтверждает или корректирует
→ graph.invoke(None, config)  # возобновление с текущего чекпоинта
→ email отправлен
```

Без interrupt — агент мог бы совершить необратимое действие автономно. С interrupt — человек остаётся в контуре для high-stakes операций. Это разрыв между демо-агентом и production-агентом.

Чекпоинтер (MemorySaver для тестов, SqliteSaver или PostgresSaver для production) сохраняет State после каждого узла. Это также обеспечивает fault tolerance: если процесс упал на 15-м шаге из 20, можно продолжить с 15-го, а не с нуля.

---

## БЛОК 7: МОНИТОРИНГ И PRODUCTION

### 7.1 Почему традиционный APM (Application Performance Monitoring) недостаточен для LLM

Традиционные системы мониторинга (Datadog, Prometheus/Grafana, New Relic) отлично покрывают инфраструктурные метрики: CPU, память, HTTP latency, error rate. Для LLM-приложений этого принципиально недостаточно.

HTTP-запрос к LLM-эндпоинту с точки зрения APM: один HTTP POST, 1.5 секунды, код 200. Никакой информации о:
- Каким был промпт и соответствовал ли он ожидаемому формату
- Какой версии промпт использован (v1 или v2)
- Сколько токенов потрачено (≠ latency при streaming)
- Вернул ли агент правильный инструмент
- Была ли галлюцинация в ответе
- Какие документы извлёк ретривер

LLM-специфичная observability оперирует понятием трейса как дерева вызовов с семантикой:

```
Trace: customer_support_request (1.8s)
├── Span: prompt_formatting (3ms)
│     input: {user_query: "VPN не работает"}
│     output: "System: ты IT-ассистент..."
├── Span: qdrant_retrieval (120ms)
│     input: query_vector
│     output: 5 chunks, top_score=0.89
├── Span: llm_call (1.6s)
│     model: qwen3:8b
│     input_tokens: 847, output_tokens: 123
│     output: "Проверьте настройки VPN..."
└── Span: output_parsing (2ms)
```

Этот трейс позволяет диагностировать: ретривал вернул релевантное? LLM выдал галлюцинацию? Какая часть latency — ретривал, какая — генерация?

---

### 7.2 Langfuse: архитектура и почему он лучше LangSmith для on-premise

LangSmith — проприетарный SaaS от LangChain Inc. Данные трейсов отправляются на серверы компании. Для корпораций с требованиями ISO 27001, 152-ФЗ, GDPR — это блокер.

Langfuse — open-core платформа, спроектированная для self-hosted развёртывания. Лицензирование двухуровневое: Python/TypeScript SDK — MIT; серверная часть — FSL (Functional Source License). FSL разрешает on-premise развёртывание для внутренних нужд, запрещает делать из Langfuse конкурирующий SaaS. Через два года после релиза каждой версии FSL автоматически конвертируется в Apache 2.0. Для корпоративной установки это приемлемо — но юристов предупредите заранее, что это не чистый MIT.

Архитектура v3 (2025) рассчитана на high-load трейсинг:

```
[SDK (Python/TypeScript)] → [OTLP Collector] → [ClickHouse (аналитика трейсов)]
                                             → [PostgreSQL (метаданные)]
                                             → [MinIO/S3 (blob/промпты)]
					                         → [Redis (очередь)]
```

Разделение ClickHouse (аналитика) + PostgreSQL (метаданные) — принципиальное архитектурное решение: колоночный ClickHouse держит миллиарды строк трейсов с быстрой агрегацией, PostgreSQL хранит конфигурации и метаданные с ACID-гарантиями.

**Prompt Management** — хранит промпты с версиями и метками. Два разных параметра в зависимости от операции:

```python
# Создание версии — labels (список)
langfuse.create_prompt(
    name="rag-system-prompt",
    prompt="Ты корпоративный ассистент...",
    labels=["production"]   # сразу помечаем как production
)

# Получение по метке — label (строка)
prompt = langfuse.get_prompt("rag-system-prompt", label="production")

# По умолчанию и так возвращается production-версия
prompt = langfuse.get_prompt("rag-system-prompt")

# Конкретная версия по номеру
prompt = langfuse.get_prompt("rag-system-prompt", version=2)
```

При откате достаточно переставить метку `production` в UI на нужную версию — без пересборки и деплоя кода.

**Datasets** — именованные наборы тест-кейсов `{input, expected_output, metadata}`. Каждый запуск eval-набора — «dataset run», который сохраняется с метриками. Основа для regression-тестирования: запускаем после каждого изменения промпта и сравниваем метрики с baseline.

**Scores** — произвольные числовые метрики, привязанные к конкретному трейсу: оценка LLM-judge, human annotation асессоров, автоматическая проверка регулярками. Агрегируются в дашборде по времени — видна деградация качества до того, как она станет заметна пользователям.

**`@observe()` декоратор** — автоматически создаёт span при вызове функции, записывает input (аргументы), output (return value) и latency. Важная деталь по обработке ошибок: если внутри функции возникает исключение, декоратор перехватывает его, помечает span статусом `level="ERROR"` с текстом ошибки в `status_message`, после чего **пробрасывает исключение дальше** стандартным `raise`. Функция падает как обычно — декоратор не «глотает» ошибки и не меняет поведение приложения:

```python
from langfuse import observe

@observe(as_type="generation")
def call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content   # output автоматически записан в span

@observe()
def rag_pipeline(question: str) -> str:
    docs = retriever.invoke(question)   # input записан в родительский span
    return call_llm(build_prompt(docs, question))
```

---

### 7.3 Prometheus: метрики как time-series

Prometheus — pull-based система сбора метрик. Раз в N секунд (scrape_interval, обычно 15s) обращается к эндпоинту `/metrics` приложения и забирает текущие значения.

Четыре типа метрик в `prometheus_client`:

**Counter** — только возрастает. Никогда не уменьшается (сброс только при рестарте). Используется: количество запросов, ошибок, токенов. `inc()` увеличивает на 1, `inc(n)` — на n.

**Histogram** — разбивает наблюдения по бакетам, вычисляет сумму и количество. Из него Grafana вычисляет перцентили через `histogram_quantile()`. Используется для latency: `observe(elapsed_seconds)`.

**Gauge** — может расти и убывать. Текущее значение. Используется: VRAM utilization, количество активных сессий.

**Summary** — клиентские перцентили (вычисляются в приложении, а не в Prometheus). Используется реже из-за невозможности агрегации между инстансами.

Почему перцентили важнее среднего для LLM:

```
Среднее latency = 1.2 сек → "всё хорошо"
P99 latency    = 12 сек   → 1% пользователей ждут 12 секунд
                           → это 1 из 100 запросов
                           → при 1000 RPS = 10 плохих запросов в секунду
```

Среднее маскирует хвосты. SLA-соглашения всегда определяются через P95 или P99.

---

### 7.4 OpenTelemetry: стандарт переносимости observability

OpenTelemetry (OTel) — CNCF-проект, вендор-нейтральный стандарт для сбора трейсов, метрик и логов. LangChain и LangGraph поддерживают OTel нативно через callbacks.

Архитектурная ценность: один раз инструментировать приложение через OTel, затем выгружать данные в любой backend — Langfuse, Jaeger, Zipkin, Tempo, Datadog, New Relic — меняя только конфигурацию экспортёра без изменения кода.

```
Приложение (OTel SDK) → OTLP → [Langfuse] (для МК)
                             → [Jaeger] (для on-premise tracing)
                             → [Datadog] (для enterprise)
```

Для корпоративного on-premise: Langfuse принимает трейсы по протоколу OTLP — стандартная интеграция.

---

### 7.5 Метрики качества специфичные для LLM

**Faithfulness** — доля утверждений в ответе, подтверждённых retrieved-контекстом. Измеряется через LLM-judge: "Это утверждение поддерживается следующим фрагментом: чанк? (да/нет)". Метрика от 0 до 1.

**Context Recall** — доля информации из эталонного ответа, которая нашлась в retrieved-чанках. Низкий recall → ретривер не нашёл нужное. Требует эталонного ответа — дорого в разметке.

**Context Precision** — доля retrieved-чанков, релевантных вопросу. Низкая precision → засорение контекста → модель отвлекается на нерелевантное.

**Answer Relevance** — семантическая близость ответа к вопросу. Можно измерить через cosine similarity эмбеддингов.

**Tool Selection Accuracy** — только для агентов. Доля запросов, для которых агент выбрал правильный инструмент. Требует аннотированного тест-набора с ожидаемым инструментом.

LLM-as-a-Judge: использование сильной LLM (GPT-4, Claude) как автоматического оценщика качества ответов другой LLM. Позволяет масштабировать eval без человеческой разметки. Bias: модель-судья может предпочитать ответы в своём стиле. Mitigation: использовать разные модели как судей и усреднять.

---

### 7.6 vLLM и PagedAttention: почему производительность важна

Стандартный inference (HuggingFace Transformers) хранит KV-кеш для каждого запроса в непрерывных блоках памяти. При обработке нескольких запросов параллельно возникает фрагментация: даже при наличии суммарно достаточной VRAM отдельные запросы не влезают. Результат: либо один запрос за раз (низкий throughput), либо OOM.

PagedAttention (vLLM, Kwon et al., 2023) заимствует идею виртуальной памяти из операционных систем. KV-кеш делится на страницы фиксированного размера, хранящиеся в нефиксированных блоках VRAM. Логически непрерывный контекст → физически разбросанные страницы. Это позволяет:

- Одновременно обрабатывать 10-50 запросов (continuous batching)
- Делиться страницами KV-кеша между запросами с общим префиксом (prefix caching)
- Практически устранить фрагментацию VRAM

Результат на RTX 4090: HuggingFace inference — 15-25 токенов/сек при одном запросе. vLLM — 80-120 токенов/сек при одном запросе, 400-600 токенов/сек при 10 параллельных запросах (суммарно).

Для production-API, обслуживающего нескольких пользователей одновременно, vLLM — не опция, а необходимость.

---

### 7.7 SGLang и RadixAttention: специализация для RAG

SGLang (Systems for Generative Language, 2024) вводит RadixAttention — расширение идеи prefix caching для структурированных программ над LLM.

В RAG-системе каждый запрос имеет структуру:

```
[системный промпт] [документ 1] [документ 2] [документ 3] [вопрос пользователя]
```

Системный промпт и часто — retrieved-документы из популярных разделов — повторяются в тысячах запросов. RadixAttention строит Radix Tree над токенами и кеширует вычисленные KV-состояния для общих префиксов. При следующем запросе с тем же системным промптом: вычисляем только хвост (уникальный вопрос).

**В чем проблема обычных LLM**

Когда пользователь задает вопрос RAG-системе, нейросети отправляют огромный "бутерброд" из текста, который всегда строится по одному шаблону:

1. **Системные правила:** "Ты умный корпоративный помощник, отвечай коротко..."
2. **Контекст:** Три страницы инструкции, которые база данных нашла по запросу.
3. **Вопрос:** "Как оформить отгул?"

Обычная модель (без умного кеширования) "читает" этот текст **каждый раз с самого начала**. Даже если 1000 сотрудников за день зададут вопросы по одной и той же инструкции, нейросеть тысячу раз заново перечитает правила и саму инструкцию, сжигая вычислительные мощности и время (тот самый $O(n^2)$).

**Что придумали в SGLang (RadixAttention)**

Эта технология работает как умная заготовка. Она умеет сохранять в оперативную память математические вычисления (KV-кеш) для тех кусков текста, которые находятся в начале запроса и часто повторяются.

* Если Петя спросил про отгул, нейросеть прочитала правила, регламент отпусков и вопрос Пети. SGLang **сохранил в память** вычисления для правил и регламента.
* Когда Маша через минуту задаст другой вопрос, но по **тому же самому регламенту**, нейросети не придется читать правила и регламент заново. Она достанет готовую базу из кеша (мгновенно) и потратит вычислительные силы только на обработку уникального вопроса Маши.

**Когда это дает турбо-ускорение (в 2-5 раз)**

Это блестяще работает на **статичных базах знаний** — например, документация продукта или корпоративная вики. Люди часто гуглят одни и те же популярные статьи. SGLang один раз "переваривает" популярную статью, а дальше отвечает на вопросы по ней в несколько раз быстрее других движков (например, vLLM).

**Когда это почти бесполезно**

Это не даст прироста скорости, если контент **генерируют сами пользователи**. Например, если это сервис, куда каждый юзер загружает *свой личный* 100-страничный PDF-договор и задает по нему вопросы.

В этом случае общим у всех запросов будет только короткое системное правило в начале, а сам длинный контекст всегда будет уникальным. Кешировать будет нечего, и нейросети придется честно читать каждый договор с нуля.

При corpus-based RAG (документация фиксирована) SGLang даёт 2-5× ускорение по сравнению с vLLM для одинаковых запросов, поскольку документация повторно используется из кеша.

При user-generated контенте (каждый retrieved-контекст уникален) преимущество минимально — только системный промпт кешируется.

---

### 7.8 Финальная связка архитектур: от МК1 к МК4

Три мастер-класса строят одну систему, добавляя слои:

```
МК1: Fine-tuning
  T-lite-dpo ← модель, знающая корпоративный стиль и безопасное поведение
  Проблема оставшаяся нерешённой: не знает конкретной документации
     ↓
МК2: LangChain Enterprise
  T-lite-dpo + RAG (Qdrant + BGE-M3) ← теперь знает документацию
  T-lite-dpo + LangGraph ← может выполнять многошаговые задачи
  Проблема оставшаяся нерешённой: нет автоматического контроля качества
     ↓
МК3: LLMOps
  + Guardrails (Guardrails AI + NeMo + Llama Guard) ← безопасность runtime
  + Langfuse ← трассировка и versioning промптов
  + Prometheus ← метрики для алертинга
  + pytest + promptfoo + GitHub Actions ← автоматическое качество в CI/CD
     ↓
МК4: RAG Advanced (план)  → углублённый RAG: GraphRAG, ColBERT, re-ranking

Результат: production-ready корпоративная AI-система
с полным жизненным циклом от обучения до мониторинга.
```

Этот стек в разных вариантах развёртывают компании из Fortune 500 в 2026 году. Реальные примеры: Klarna (85M пользователей) использует LangGraph для агентного обслуживания клиентов; LinkedIn — LangGraph для профессиональных рекомендаций; Replit — для агентного кодинга. Harvey AI обслуживает 97% Am Law 100 на RAG-архитектуре. MUFG Bank (Япония) использует LangChain LCEL в production с миграцией Python → TypeScript для frontend.