# Smart Tomato Garden — Rust/Axum inference server
# Multi-stage: builder (Rust + OpenCV dev) → runtime (OpenCV libs only, ~120 MB)
# ONNX Runtime is statically linked — no extra .so needed.
#
# Build from repo root:
#   docker build -f server/Dockerfile -t garden-ml-rust:latest .

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    build-essential pkg-config \
    clang libclang-dev \
    libopencv-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --no-modify-path

WORKDIR /build

# Cache dependency compilation (includes opencv bindgen + ort static link)
COPY server/Cargo.toml server/Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs
RUN PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig \
    cargo build --release \
    && rm -f target/release/garden-ml-server \
              target/release/deps/garden_ml_server*

# Build actual application
COPY server/src ./src
RUN touch src/main.rs && \
    PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig \
    cargo build --release

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Only OpenCV runtime libs + CA certs (libopencv-imgcodecs pulls imgproc + core)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-imgcodecs406t64 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/garden-ml-server ./

ENV GML_HOST=0.0.0.0
ENV GML_PORT=5002
ENV GML_ARTIFACTS_DIR=/app/ml/artifacts/model_registry/v0005
ENV RUST_LOG=garden_ml_server=info,tower_http=info

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:5002/health > /dev/null

ENTRYPOINT ["./garden-ml-server"]
