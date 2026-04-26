FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copier les données ML dans le container
COPY ../docsML /data/docsML

ENV DATA_DIR=/data/docsML
ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
