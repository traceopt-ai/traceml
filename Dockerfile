FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN python -m pip install --no-cache-dir --timeout 120 ".[torch,dashboard]" \
    && rm -rf build src/traceml_ai.egg-info

# Let the dashboard bind outside the container so `docker run -p 8765:8765` works.
ENV TRACEML_DASHBOARD_HOST=0.0.0.0

EXPOSE 8765

CMD ["traceml", "run", "examples/diagnosis/dataloader_bottleneck_demo.py", "--mode=summary", "--args", "--scenario", "slow"]
