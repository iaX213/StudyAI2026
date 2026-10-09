
# Docker

Источники:
- https://docs.docker.com/engine/install/debian/
- https://hub.docker.com/_/ubuntu
- https://www.youtube.com/watch?v=X48VuDVv0do&t=1s&ab_channel=TechWorldwithNana
- https://www.youtube.com/watch?v=7EuZwvNROs4

---

## 1. Контекст — DevOps, контейнеризация, оркестрация

**DevOps** — набор практик, объединяющих разработку (Dev) и эксплуатацию (Ops) с целью сокращения цикла разработки и обеспечения непрерывной поставки программного обеспечения.

**Docker** (альтернатива: **Podman**) — платформа контейнеризации, использующая виртуализацию на уровне ОС для упаковки приложения со всеми его зависимостями и конфигурацией в изолированный пакет — **контейнер**.

**Kubernetes** — система оркестрации контейнеров, автоматизирующая деплой, масштабирование и управление контейнеризированными приложениями. Работает с Docker, Containerd, CRI-O.

**OpenShift** (Red Hat) — enterprise-платформа на базе Kubernetes (fork) для on-premises контейнеризованных рабочих нагрузок.

---

## 2. Контейнеры vs Виртуальные машины

```text
      Virtual Machines                      Containers
+----------+----------+----------+   +----------+----------+----------+
|  App A   |  App B   |  App C   |   |  App A   |  App B   |  App C   |
+----------+----------+----------+   +----------+----------+----------+
| Bins/Lib | Bins/Lib | Bins/Lib |   | Bins/Lib | Bins/Lib | Bins/Lib |
+----------+----------+----------+   +--------------------------------+
| Guest OS | Guest OS | Guest OS |   |         Docker Engine          |
+----------+----------+----------+   +--------------------------------+
|         Hypervisor             |   |            Host OS             |
+--------------------------------+   +--------------------------------+
|            Host OS             |   |         Infrastructure         |
+--------------------------------+   +--------------------------------+
|          Infrastructure        |
+--------------------------------+

Каждая VM несёт полный Guest OS (~GBs).    Контейнеры делят ядро хоста (~MBs).
Изоляция сильнее, накладные расходы выше.  Быстрый старт, низкий оверхед.
```

---

## 3. Архитектура Docker

```text
                                DOCKER HOST
+------------------+  +--------------------------------+   +-------------------+
|  Docker Client   |  |         Docker Daemon          |   |  Docker Registry  |
|  (docker CLI)    |  |            (dockerd)           |   |  (Docker Hub)     |
|                  |  |                                |   |                   |
|  docker build  ----->  Builds image from Dockerfile  |   |  [nginx:latest]   |
|  docker pull   <-----  Downloads image from registry <---> [ubuntu:22.04]    |
|  docker run    ----->  Creates & runs container      |   |  [alpine:3.18]    |
|  docker ps     <-----  Returns container list        |   |                   |
+------------------+  +--------------------------------+   +-------------------+
                                     |
                         +-----------+-----------+
                         |                       |
                    +--------+             +----------+
                    | Images |             |Containers|
                    +--------+             +----------+
```

---

## 4. Установка Docker

**Debian / Ubuntu:**

```bash
# официальная документация:
# https://docs.docker.com/engine/install/debian/
curl -fsSL https://get.docker.com | sh
```

**CentOS / RHEL:**

```bash
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
docker run hello-world
```

**Astra Linux:**
- https://wiki.astralinux.ru/pages/viewpage.action?pageId=158601444

Пакеты для Debian-based (ручная установка):
- https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/

---

## 5. Основные команды Docker

**Образы:**

```bash
docker images                               # список локальных образов
docker pull alpine                          # скачать образ из registry
docker pull ubuntu                          # https://hub.docker.com/_/ubuntu
```

**Запуск контейнеров:**

```bash
docker run -it ubuntu /bin/bash             # интерактивный запуск
docker run -d -p 8080:80 nginx              # фоновый режим, пробросить порт
docker run -p 80:80 nginx:08_06_2022        # конкретный тег
docker run -it -p 80:80 nginx:08_06_2022 /bin/bash -c "nginx -g 'daemon off;'"
docker run --name my_nginx -p 80:80 nginx:08_06_2022
docker run --name <name> -it <image_id>:<tag> /bin/bash

# volume mount
docker run -v ${pwa}/index.html:/usr/share/nginx/html/index.html nginx

# автозапуск после перезагрузки
docker run -dit --restart unless-stopped <image>
# https://www.digitalocean.com/community/questions/how-to-start-docker-containers-automatically-after-a-reboot
```

**Управление запущенными контейнерами:**

```bash
docker ps -a                                # список контейнеров (включая stopped)
docker stats                                # realtime потребление ресурсов
docker exec -it <container_id> /bin/bash    # подключиться к работающему контейнеру
docker exec -it <container_id> /bin/bash -c "nginx -g 'daemon off;'"

docker stop <container_id>
docker stop $(docker ps -a -q)              # остановить все контейнеры
docker kill $(docker ps -qa)               # kill все контейнеры

docker rm $(docker ps -a -q)               # удалить все остановленные контейнеры
docker container rm [OPTIONS] CONTAINER     # удалить конкретный

docker commit <container_id> <image>:<tag> # сохранить состояние контейнера как образ
docker cp <container_id>:/file/path/within/container /host/path/target
```

**Проверка:**

```bash
curl localhost
```

---

## 6. Dockerfile — описание образа

Каждая инструкция в Dockerfile создаёт новый **layer** образа, который кэшируется локально. Это ускоряет повторные сборки при изменении только нижних слоёв.

```dockerfile
# Each instruction in this file generates a new layer that gets pushed
# to your local image cache
#
# Lines preceded by # are regarded as comments and ignored
#
# The line below states we will base our new image on the Latest Official Ubuntu
FROM ubuntu:24.04
#
# Identify the maintainer of an image
LABEL maintainer="myname@somecompany.com"
#
# Update the image to the latest packages
RUN apt-get update && apt-get upgrade -y
#
# Install NGINX to test
RUN apt-get install nginx -y
#
# Expose port 80
EXPOSE 80
#
# Last is the actual command to start up NGINX within our Container
CMD ["nginx", "-g", "daemon off;"]
```

Редактирование Dockerfile:

```bash
nano Dockerfile
```

Альтернативно — несколько Dockerfile для разных сервисов:

```bash
docker build Dockerfile.nginx  .
docker build Dockerfile.php    .
```

---

## 7. Сборка образа

```bash
docker build -t <name_of_rep>:<tag> .       # собрать образ из Dockerfile в текущей директории
docker build -t nginx:08_06_2026 .          # пример с конкретным тегом
docker images                               # проверить результат сборки
```

---

## 8. Linux Namespaces — механизм изоляции контейнеров

Docker использует **Linux namespaces** для изоляции контейнеров. Каждый контейнер получает собственные namespaces, видя только своё «окружение».

```text
  Вне контейнера ("host" namespace)      Внутри контейнера
  ─────────────────────────────────      ──────────────────────────────
  ps aux показывает все процессы         ps aux показывает только
  системы                                процессы своего PID namespace

  Один общий network namespace           Отдельный network namespace
  (eth0, lo, ...)                        (eth0 = veth внутри контейнера)

  Общая файловая система                 Изолированный mnt namespace
                                         (rootfs контейнера)
```

Типы namespaces в Linux:

```bash
$ lsns -p 273
NS          TYPE
4026531835  cgroup
4026531836  pid
4026531837  user
4026531838  uts
4026531839  ipc
4026531840  mnt
4026532009  net
```

Просмотр namespaces текущего процесса:

```bash
ls -l /proc/$$/ns
ls -l /proc/273/ns
```

**PID namespace** — дерево процессов внутри контейнера начинается с PID 1:

```text
  Host PID namespace          Container PID namespace
  ──────────────────          ───────────────────────
     PID 1                         PID 1  (= PID 7 снаружи)
     PID 2                         PID 2  (= PID 8 снаружи)
     PID 3                         PID 3  (= PID 9 снаружи)
     PID 4                         PID 4  (= PID 10 снаружи)
     PID 5
     PID 6
      └── [дочерний PID namespace]
           PID 7
           PID 8
           PID 9
           PID 10
```

**Создание отдельного UTS namespace вручную** (пример как это работает без Docker):

```bash
unshare --uts zsh    # запустить shell с отдельным UTS namespace (hostname изолирован)
```

**Вход в namespace работающего контейнера** (аналог `docker exec`):

```bash
nsenter -t 6666 --mount --uts --ipc --net --cgroup /bin/sh
# где 6666 — PID целевого процесса (контейнера)
```

Смотреть также: `firejail` — пример sandbox-изоляции на базе namespaces.

---

## 9. Очистка Docker

```bash
docker system prune -f                         # удалить все неиспользуемые объекты
docker builder prune -a -f                     # очистить build cache
docker image prune -a -f                       # удалить все неиспользуемые образы

# очистить логи контейнеров
sudo sh -c "truncate -s 0 /var/lib/docker/containers/*/*-json.log"

# проверить занятое место
sudo du -hs /var/log
sudo du -hs /var/lib/docker/
```

---

## 10. Установка утилит в контейнер (для диагностики)

```bash
docker run -it ubuntu /bin/bash
whoami                                         # root (внутри контейнера)
apt install procps-ng inetutils-ping -y        # установить ps, ping
```

Утилиты для работы с namespaces и processes:

```bash
lsns                                           # список всех namespaces
nsenter -t <PID>                               # войти в namespace процесса
```