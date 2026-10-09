

# Docker Compose и Podman Compose

Источники:
- https://www.redhat.com/en/blog/podman-compose-docker-compose
- https://oneuptime.com/blog/post/2026-03-18-choose-between-quadlet-docker-compose-systemd/view

---

## 1. Docker Compose V2 — что это и как работает

**Docker Compose** — инструмент декларативного описания многоконтейнерных приложений в одном YAML-файле. Один файл заменяет десятки `docker run` команд с флагами.

В 2021 году проект Docker Compose был полностью переписан с Python на Go — это Docker Compose V2. Важные отличия от V1:

- V1: `docker-compose` (отдельный бинарник, Python, **deprecated**, удалён в 2024)
- V2: `docker compose` (без дефиса, встроен в Docker CLI как плагин)

```text
  Docker Compose V2: архитектура
  ─────────────────────────────────────────────

  compose.yaml                  Docker CLI
  ─────────────────   ──────────────────────────────
  services:        →  docker compose up
    web:               │
    db:                ▼
    cache:         Docker Daemon (dockerd)
                       │
                       ├── контейнер: web
                       ├── контейнер: db
                       └── контейнер: cache
                       │
                  общая сеть (автосоздание)
                  named volumes (автосоздание)
```

Официальное имя файла с 2023 года — `compose.yaml` (также принимаются `compose.yml`, `docker-compose.yaml`, `docker-compose.yml` в порядке приоритета).

---

## 2. Структура compose.yaml и практический пример

```yaml
# compose.yaml — типовой стек: nginx + app + postgres + redis
name: myapp

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app
    restart: unless-stopped

  app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  cache:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

---

## 3. Основные команды Docker Compose

```bash
# запуск / остановка
docker compose up -d                     # запустить в фоне
docker compose up --build                # пересобрать образы перед запуском
docker compose down                      # остановить и удалить контейнеры
docker compose down -v                   # + удалить volumes

# состояние и логи
docker compose ps                        # список сервисов
docker compose ps -a                     # включая остановленные
docker compose logs -f                   # логи всех сервисов
docker compose logs -f app               # логи конкретного сервиса
docker compose top                       # процессы внутри контейнеров

# управление конкретным сервисом
docker compose restart app
docker compose exec app /bin/bash        # войти в контейнер
docker compose run --rm app sh           # запустить одноразово

# сборка и обновление
docker compose build --no-cache
docker compose pull                      # обновить образы из registry
docker compose up -d --force-recreate   # пересоздать контейнеры
docker compose up -d --scale app=3      # масштабировать сервис

# профили (запускать не все сервисы сразу)
docker compose --profile debug up        # запустить только debug-профиль
docker compose --profile monitoring up

# конфигурация
docker compose config                    # показать итоговый compose.yaml
docker compose convert                   # конвертация форматов
```

**Watch mode** (Docker Compose v2.22+, live reload при изменении файлов):

```bash
docker compose watch
```

```yaml
# в compose.yaml
services:
  app:
    develop:
      watch:
        - action: sync
          path: ./src
          target: /app/src
        - action: rebuild
          path: package.json
```

---

## 4. Podman Compose — три пути

Docker Compose V2 в Podman поддерживается с версии Podman v4.1 (2022), хотя поддержка Buildkit API всё ещё в процессе.

```text
  Три способа использовать Compose с Podman:
  ────────────────────────────────────────────────────────────────

  Способ 1: Docker Compose → Podman socket (максимальная совместимость)
  ──────────────────────────────────────────────────────────────────────
  docker compose  ──►  Podman socket  ──►  Podman (daemonless)
  (стандартный CLI)    (эмулирует       (rootless, без daemon)
                        Docker API)

  systemctl --user enable --now podman.socket

  Способ 2: podman-compose (community CLI)
  ──────────────────────────────────────────
  podman-compose  ──►  Podman CLI напрямую
  (Python, FOSS)       (rootless by default)

  pip install podman-compose

  Способ 3: Quadlet (production, systemd-native)
  ──────────────────────────────────────────────
  .container unit  ──►  systemd --user  ──►  Podman
  (декларативно)       (lifecycle       (rootless)
                         management)
```

### Способ 1: Docker Compose + Podman socket

```bash
# включить Podman socket (эмулирует Docker API)
systemctl --user enable --now podman.socket

# указать Docker Compose использовать Podman socket
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock

# теперь docker compose работает через Podman
docker compose up -d
docker compose ps
```

### Способ 2: podman-compose

```bash
pip install podman-compose

# синтаксис идентичен docker compose
podman-compose up -d
podman-compose down
podman-compose logs -f
podman-compose exec app /bin/bash
```

### Способ 3: Quadlet — production-замена Compose на RHEL

```bash
# ~/.config/containers/systemd/webapp.network
[Network]
Driver=bridge

# ~/.config/containers/systemd/db.container
[Unit]
Description=PostgreSQL Database
After=network.target

[Container]
Image=postgres:16-alpine
Environment=POSTGRES_USER=user
Environment=POSTGRES_PASSWORD=pass
Environment=POSTGRES_DB=mydb
Volume=pgdata.volume:/var/lib/postgresql/data
Network=webapp.network

[Service]
Restart=always

[Install]
WantedBy=default.target

# ~/.config/containers/systemd/app.container
[Unit]
Description=Application
After=db.service

[Container]
Image=myapp:latest
Environment=DATABASE_URL=postgres://user:pass@db:5432/mydb
Network=webapp.network
PublishPort=8080:8080

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user start db app
systemctl --user status app
journalctl --user -u app -f      # логи через journald
```

---

## 5. Архитектурное сравнение: Docker Compose vs Podman Compose vs Quadlet

```text
               Docker Compose       podman-compose        Quadlet
               ──────────────       ──────────────        ───────
  Язык/тип     Go плагин CLI        Python FOSS           systemd units

  Daemon       dockerd              нет (daemonless)      нет (daemonless)

  Rootless     нет (по умолчанию)   да                    да

  Lifecycle    Compose manages      Compose manages       systemd manages
  management   (не systemd)         (не systemd)          (restart, logs,
                                                           boot order)

  Логи         docker logs          podman logs           journald (journalctl)

  Автозапуск   restart:             restart:              WantedBy=
  при ребуте   unless-stopped       unless-stopped        default.target
               (через dockerd)      (через systemd unit)  (нативно)

  Совместимость compose.yaml        compose.yaml          свой формат
  формата       полная              ~95% (без Buildkit)   (.container)

  Зрелость     высокая              community, stable     production RHEL

  CI/CD        широкая поддержка    rootless в CI         не для CI
               (GitHub Actions,
               GitLab CI)

  Production   single-host ok,      single-host ok,       single-host
  применение   не для HA            не для HA             production best
```

---

## 6. Enterprise best practices: точки применения

```text
  ┌──────────────────────────────────────────────────────────────────┐
  │  Сценарий                  Рекомендация              Почему      │
  ├──────────────────────────────────────────────────────────────────┤
  │  Local development         docker compose            Зрелость,   │
  │  (любая платформа)         + compose.yaml            экосистема, │
  │                                                      Watch mode  │
  ├──────────────────────────────────────────────────────────────────┤
  │  CI/CD pipeline            podman-compose или        Rootless,   │
  │  (Linux runner)            Docker Compose            нет daemon, │
  │                            + Podman socket           безопасно   │
  ├──────────────────────────────────────────────────────────────────┤
  │  Single-host production    Quadlet                  systemd-     │
  │  RHEL / CentOS Stream       (Podman 4.4+)            native,     │
  │                                                      journald,   │
  │                                                      rootless    │
  ├──────────────────────────────────────────────────────────────────┤
  │  Single-host production    docker compose +          Простота,   │
  │  Ubuntu / Debian           Watchtower (autoupdate)   зрелость    │
  ├──────────────────────────────────────────────────────────────────┤
  │  Multi-host / HA /         Kubernetes + Helm         Compose не  │
  │  масштабирование           (не Compose!)             для этого   │
  ├──────────────────────────────────────────────────────────────────┤
  │  Rapid prototyping /       docker compose            Fastest     │
  │  demo стенды               (любой ОС)                time-to-run │
  └──────────────────────────────────────────────────────────────────┘
```

**Ключевое ограничение Compose в 2026:** для production single-server деплоев, где важна системная интеграция, Quadlet является более естественным выбором. Для dev-воркфлоу и multi-container приложений Docker Compose остаётся удобнее.

Compose — это **инструмент одного хоста**. Как только появляется требование HA, failover, rolling update или балансировка — нужен Kubernetes. Compose не масштабируется горизонтально: `--scale app=3` создаёт три контейнера на одном хосте без распределения и без автовосстановления при падении ноды.

Production Linux-серверы, self-hosting, security-sensitive нагрузки — Podman с Quadlet является более сильным дефолтом: daemonless, rootless по умолчанию, systemd-интеграция и нулевые лицензионные затраты.