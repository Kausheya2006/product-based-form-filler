FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --default-timeout=1000

COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

COPY src /app/src

ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]
