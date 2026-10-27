# syntax=docker/dockerfile:1
# System 1 (Rust) and System 2 + the sim + the API (Python) run as two processes in ONE container: they talk through POSIX shared
# memory, which only works between processes that share /dev/shm.

FROM rust:1.90-slim-bookworm AS system1
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM python:3.11-slim-bookworm
# MuJoCo renders the camera frame offscreen; with no GPU in the container that goes through OSMesa (software GL).
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 POLICYSERVE_SYSTEM1=/app/bin/system1
WORKDIR /app
COPY py ./py
RUN pip install --no-cache-dir "./py[llm]"
COPY --from=system1 /src/target/release/system1 ./bin/system1
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["python", "-m", "policyserve.api", "--host", "0.0.0.0", "--port", "8000"]
