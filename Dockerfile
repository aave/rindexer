FROM rust:1.88.0-bookworm AS builder
ARG TARGETPLATFORM
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Build dependencies. Only exercised by the build-from-source fallback below; the
# CI workflow supplies a pre-built binary and skips compilation entirely.
# node is required by the core build.rs (embeds the GraphQL binary).
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    cmake \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# The CI workflow builds the binary natively per-arch and drops it in docker-binary/.
# If it's there, use it; otherwise build from source for the target platform.
# reth is intentionally excluded (not needed for the published image).
COPY docker-binary/ /tmp/docker-binary/
RUN if [ -f /tmp/docker-binary/rindexer_cli ]; then \
        echo "Using pre-built binary"; \
        mkdir -p /app/target/release; \
        cp /tmp/docker-binary/rindexer_cli /app/target/release/rindexer_cli; \
        chmod +x /app/target/release/rindexer_cli; \
    else \
        echo "Building from source for ${TARGETPLATFORM}"; \
        if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
            RUSTFLAGS='-C target-cpu=x86-64-v2' cargo build --release -p rindexer_cli --features jemalloc; \
        else \
            RUSTFLAGS='-C target-cpu=neoverse-n1' cargo build --release -p rindexer_cli; \
        fi; \
    fi

# trixie (glibc 2.41) — must be >= the glibc the binary was linked against. CI builds the
# binary natively on ubuntu-24.04 (glibc 2.39); bookworm-slim (2.36) is too old and the
# binary fails to load with "GLIBC_2.39 not found".
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Foundry (forge) — used by `rindexer phantom` commands. foundryup installs the
# binaries matching the final image's architecture.
RUN curl -L https://foundry.paradigm.xyz | bash
RUN /root/.foundry/bin/foundryup

COPY --from=builder /app/target/release/rindexer_cli /app/rindexer
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

ENTRYPOINT ["/app/rindexer"]
