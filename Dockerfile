FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN python -m pip install --no-cache-dir --timeout 120 ".[torch]" \
    && rm -rf build src/traceml_ai.egg-info

CMD ["traceml", "run", "examples/diagnosis/dataloader_bottleneck_demo.py", "--mode=summary", "--args", "--scenario", "slow"]
