FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --default-timeout=1000

COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

COPY src /app/src

ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]
