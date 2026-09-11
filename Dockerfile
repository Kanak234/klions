# Stage 1: Build binaries
FROM rust:1-slim-bookworm AS builder

WORKDIR /usr/src/klions

# Copy workspace manifests, docs, and member crates
COPY Cargo.toml Cargo.lock README.md LICENSE ./
COPY crates ./crates

# Build release binaries with locked dependencies
RUN cargo build --release --locked -p klions-cli -p klions-lsp

# Stage 2: Runtime image
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /home/appuser -m -s /bin/false appuser

# Copy compiled binaries from builder stage
COPY --from=builder /usr/src/klions/target/release/klions /usr/local/bin/klions
COPY --from=builder /usr/src/klions/target/release/klions-lsp /usr/local/bin/klions-lsp

USER appuser:appuser
WORKDIR /home/appuser

# Built-in container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["klions", "health"] || exit 1

ENTRYPOINT ["klions"]
CMD ["--help"]
