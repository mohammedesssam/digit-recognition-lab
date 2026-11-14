FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
COPY app ./app
COPY static ./static
COPY artifacts ./artifacts
COPY reports/metrics.json ./reports/metrics.json
COPY data/optdigits.tes ./data/optdigits.tes
RUN useradd --create-home appuser
USER appuser
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
