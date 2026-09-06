# Changelog

All notable changes to KLIONS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-01 — "Ember"

First release. A complete toolchain for dense feed-forward networks on CPU.

### Language

- Functions, constants, imports, and the `model` declaration
- `let` bindings with optional `mut`, type annotations, and `_` discard
- Full expression grammar with `@` for matrix multiplication, binding tighter
  than `*`
- `if` / `else`, `while`, `for ... in` over ranges, arrays, tensors, and
  datasets
- `train <model> using <dataset> { ... }` as a statement
- `no_grad { ... }` to suspend gradient tracking
- `Result<T, E>` with `?` propagation, `unwrap`, `is_ok`, `is_err`
- Optional semicolons: a newline terminates a statement where that is
  unambiguous
- Reserved words for deferred features, each reporting its target release

### Type system

- Tensor dimensions carried in the type; `_` defers one to run time
- Static shape checking on `+`, `-`, `*`, `/`, and `@`, reporting both shapes
  and the first disagreeing axis
- NumPy right-alignment broadcasting, modelled precisely
- Model layer chains verified for width consistency at declaration
- No implicit numeric conversion between values; literals take their type from
  context
- Return-path analysis, mutability checking, unused-binding warnings
- "Did you mean" suggestions within edit distance two

### Numerics

- Strided `f32` tensors of rank 0–4 with views that alias a shared buffer
- Cache-blocked matrix multiplication, batched for rank 3
- Reverse-mode automatic differentiation over 23 operations
- Every gradient rule verified against central finite differences to 1e-3
- Deterministic PCG-XSH-RR generator; `--seed` reproduces a run bit-for-bit
- Allocation ceiling checked before allocation is attempted

### Neural networks

- Layers: `Dense`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`, `Dropout`
- Losses: `mse`, `mae`, `cross_entropy`, `binary_cross_entropy`
- `cross_entropy` consumes raw logits and applies log-sum-exp internally, so a
  preceding `Softmax` is neither required nor wanted
- Optimizers: SGD with momentum, Adam, RMSProp, with published-paper defaults
- Glorot uniform initialization, zero biases
- IDX and RFC-4180 CSV loaders, hardened against malformed input
- Versioned `.klm` model format with a magic number and a shape manifest

### Tooling

- `klions run`, `check`, `fmt`, `repl`, `explain`, `version`
- Diagnostics with stable codes, spans, notes, and help lines
- `--json` for newline-delimited machine-readable diagnostics
- `--seed`, `--time`, `--color` / `--no-color`, honouring `NO_COLOR`
- Exit codes 0 / 2 / 101 / 130
- `klions-lsp`: LSP 3.17 over stdio with diagnostics, shape-aware hover,
  completion, document symbols, and go-to-definition
- VS Code extension with a distinct `keyword.ai` scope for AI constructs

### Notes

- The toolchain builds with `cargo build --release` and **no external
  dependencies**. The JSON codec and the random number generator are
  first-party, so the build needs no network.
- Deferred features are errors, not silent no-ops. `Conv2D` reports 0.2.0;
  GPU devices report 0.3.0.
