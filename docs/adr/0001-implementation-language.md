# ADR 0001 — Rust as the implementation language

**Status:** accepted · **Date:** 2026-08-01

## Context

KLIONS needs a compiler front end, a tensor engine with hand-written numerical
kernels, a reverse-mode autodiff tape, an interpreter, a CLI, and a language
server. The SRS sets three constraints that bear directly on this choice:

- DC-003: the toolchain builds with one command and no system dependencies.
- DC-007: no BLAS, no external ML library. Every kernel is first-party.
- NFR-S-005: loaders must be hardened against malformed input.

## Decision

Rust, edition 2021, with a Cargo workspace of thirteen crates and **zero
external dependencies**.

## Rationale

**Memory safety without a runtime.** The tensor engine does pointer-like work
— strided views aliasing a shared buffer, offsets, broadcast index arithmetic.
In C or C++ a stride bug is a silent out-of-bounds read that corrupts a
gradient; the training run continues and produces plausible nonsense. In Rust
it is a bounds-checked panic or, more often, a compile error. For a language
whose selling point is catching numerical mistakes, being written in something
that permits silent numerical corruption would be an awkward contradiction.

**Zero dependencies is achievable.** This mattered more than expected. The
JSON needed by the language server is ~350 lines. The PCG32 generator is ~80.
Writing them means `cargo build --release` works on a machine that has never
seen the internet — which is the actual deployment target for a teaching
language in a place with intermittent connectivity. It also means there is no
supply chain to audit and no version skew to debug two years from now.

**Enums and exhaustive matching.** The AST, the value type, and the diagnostic
set are all closed sums. Adding a token kind and forgetting to handle it
somewhere is a compile error rather than a runtime surprise found by a user.

**Cargo is the whole build system.** No CMake, no autotools, no
`LD_LIBRARY_PATH`. One command, as DC-003 requires.

## Alternatives considered

**C++ with LLVM.** The obvious choice for a compiler, and where KLIONS will go
when it needs native code generation. Rejected for v0.1.0 because LLVM is a
large system dependency that would break DC-003 outright, and because the MVP
needs a tree-walking interpreter, not a code generator. The interpreter's job
is to sequence tensor kernels; the kernels carry the numerical work, so
interpretation overhead is not on the critical path.

**Python.** Fastest to prototype, and the ecosystem is where the users are.
Rejected because the tensor kernels would then have to be NumPy — violating
DC-007 — and because distributing a Python toolchain to students means
distributing a Python environment, which is the problem KLIONS is partly
trying to avoid.

**Go.** Good build story, decent performance, much simpler than Rust.
Rejected mainly for the type system: modelling the AST and the diagnostic
taxonomy without sum types would mean interface dispatch and runtime type
assertions in exactly the places where exhaustiveness matters most.

## Consequences

- Contributors need Rust. The build needs 1.75 or newer.
- Compile times are real: a clean release build takes a few minutes.
- Some borrow-checker friction in the interpreter, resolved with `Rc<RefCell<>>`
  where values are genuinely shared. This is a fair trade for not having to
  reason about lifetimes at run time.
- The autodiff tape uses boxed closures for backward rules, which costs an
  allocation per node. Measured against training time, it does not register —
  the matmul kernel dominates by orders of magnitude.
- LLVM remains the plan for native compilation. Nothing here forecloses it:
  the front end is already a separate crate that produces a typed AST.
