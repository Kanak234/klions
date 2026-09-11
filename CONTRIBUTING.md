# Contributing to KLIONS

Thank you for your interest in contributing to the KLIONS compiler and toolchain!

## Development Setup

KLIONS is organized as a Cargo workspace with 14 member crates and a VS Code language extension.

### Prerequisites

- Rust stable toolchain (`cargo`, `rustc` 1.75+)
- Node.js & npm (for editor extension in `editors/vscode`)

### Verification Workflow

Before submitting a pull request, run the following checks locally:

1. **Check workspace compilation:**
   ```bash
   cargo check --workspace --all-targets --all-features
   ```

2. **Run all workspace tests:**
   ```bash
   cargo test --workspace --all-features
   ```

3. **Verify compiler self-health check:**
   ```bash
   cargo run -p klions-cli -- health
   ```

4. **Lint with Clippy:**
   ```bash
   cargo clippy --workspace --all-targets --all-features
   ```

## Pull Request Guidelines

- Branch from `master`.
- Keep changes atomic, focused, and well-tested.
- Ensure all CI gates pass before requesting review.
