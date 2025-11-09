FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/ ./services/
COPY templates/ ./templates/
COPY benchmark/ ./benchmark/
COPY *.py ./
COPY model/ ./model/

EXPOSE 8080

CMD ["python", "-m", "services.api.app"]

