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