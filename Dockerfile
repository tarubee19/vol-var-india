# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional extras, e.g. --build-arg EXTRAS="anthropic>=0.40" for the live LLM backend.
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then pip install $EXTRAS; fi

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY docs/kb ./docs/kb
COPY .streamlit ./.streamlit
ENV PYTHONPATH=/app/src \
    VOLVAR_RESULTS_DIR=/app/results

# Serve the committed real-data results (refreshed every weekday by .github/workflows/refresh.yml).
# For an offline demo on simulated data instead, build with --build-arg SIMULATED=1.
COPY results ./results
ARG SIMULATED=""
RUN if [ -n "$SIMULATED" ]; then \
        rm -rf results && python scripts/make_synthetic.py --out data/synthetic \
        && python scripts/run_pipeline.py --data-dir data/synthetic --results-dir results \
            --n-jobs 2 --no-figures \
        && rm -rf data; \
    fi

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "volvar.api:app", "--host", "0.0.0.0", "--port", "8000"]
