FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.lock pyproject.toml ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock

COPY . .
RUN python -m compileall -q src tests

CMD ["python", "-m", "src.data.prepare_data", "--help"]
