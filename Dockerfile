FROM python:3.11-slim AS baseline
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY crypto_predictor ./crypto_predictor
COPY examples ./examples
RUN useradd --create-home runner && mkdir /app/artifacts && chown runner /app/artifacts
USER runner
ENTRYPOINT ["python", "-m", "crypto_predictor"]
CMD ["--csv", "examples/synthetic.csv", "--model", "persistence", "--lookback", "10"]

FROM baseline AS research
USER root
COPY requirements-online.txt .
RUN pip install --no-cache-dir -r requirements-online.txt
USER runner

FROM research AS full
USER root
COPY requirements-optional.txt .
RUN pip install --no-cache-dir -r requirements-optional.txt
USER runner
