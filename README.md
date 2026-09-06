# KLIONS

**An AI-first systems programming language where tensor shapes are part of the type.**

Version 0.1.0 "Ember" · Apache-2.0 WITH LLVM-exception

---

## The problem

You write a neural network. Twelve minutes into training, it dies:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 64x10)
  File ".../torch/nn/modules/linear.py", line 114, in forward
```

The traceback names files you did not write and tensors you did not name. The
information needed to catch this — `outputs=128` on one line, `inputs=64` on
the next — was sitting in your source the entire time.

KLIONS reads it.

```
error[K1007]: layer `output` expects 64 inputs but `hidden` produces 128 outputs
  --> mnist.kl:3:14
   |
 2 |     hidden = Dense(inputs=784, outputs=128, activation="relu")
   |              ---------------------------------------------- `hidden` produces 128 outputs
 3 |     output = Dense(inputs=64, outputs=10)
   |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: layers run in declaration order, so `output` receives `hidden`'s output
   = help: set `inputs=128` on `output`
```

No tensor was allocated. No epoch ran. This appears in your editor as you type.

## A complete program

```klions
model DigitNet {
    hidden = Dense(inputs=784, outputs=128, activation="relu")
    drop   = Dropout(p=0.2)
    output = Dense(inputs=128, outputs=10)
}

fn main() {
    let train_data = Dataset::from_idx("data/train-images.idx",
                                       "data/train-labels.idx")
        .unwrap()
        .normalize()

    train DigitNet using train_data {
        epochs = 12,
        batch_size = 64,
        optimizer = Adam(0.001),
        loss = "cross_entropy",
        metrics = ["accuracy"]
    }

    let result = DigitNet.evaluate(test)
    println("accuracy: {:.4}", result.accuracy)
}
```

```
model DigitNet {
    hidden     Dense(inputs=784, outputs=128, activation="relu")   100480 params
    drop       Dropout(p=0.2)                                           0 params
    output     Dense(inputs=128, outputs=10, activation="none")      1290 params
}
total parameters: 101770

  epoch   1/12   loss 0.8715  accuracy 0.7833
  epoch   6/12   loss 0.0195  accuracy 0.9993
  epoch  12/12   loss 0.0069  accuracy 1.0000
training finished in 3.92s

held-out accuracy: 0.9983 over 600 samples
```

## Build it

```bash
cargo build --release
```

One command. **No external dependencies** — not one crate beyond the standard
library. No BLAS, no LLVM, no Python, no network. Every kernel, the random
number generator, and the language server's JSON are all first-party.

Two binaries land in `target/release`:

| Binary | Purpose |
|---|---|
| `klions` | compiler and interpreter |
| `klions-lsp` | language server |

## Use it

```bash
klions run examples/xor.kl          # compile and execute
klions check src/*.kl               # analyze without running
klions check src/*.kl --json        # machine-readable diagnostics
klions fmt src/*.kl                 # canonical formatting
klions repl                         # interactive session
klions explain K1001                # what a diagnostic means, at length
klions run model.kl --seed 7 --time # reproducible, timed
```

Exit codes: `0` success · `2` compilation failed · `101` run-time fault ·
`130` interrupted.

## What the language gives you

**Shapes in types.** `tensor<f32, 64, 784>`. Dimensions written as integers are
checked at compile time; `_` defers one to run time. Broadcasting follows the
NumPy right-alignment rule, and the checker models it precisely.

**Native AI constructs.** `model`, `train ... using ...`, and `no_grad` are
syntax, not library calls, so they can be checked and highlighted as what they
are.

**Diagnostics that name the fix.** Every error carries a stable code, a span,
and — nearly always — a `help:` line saying what to do. Names get "did you
mean" suggestions within edit distance two.

**Nothing is silently ignored.** Ask for `Conv2D` and you get an error naming
the release that will carry it. Accepting an option and doing nothing would
mean your model trained on something other than what you wrote.

**Reproducibility by default.** One seeded generator, `--seed 42` unless you
say otherwise. The same seed gives bit-identical results.

**No implicit numeric conversion.** `i32 + f32` is an error; write `as f32`.
Literals are exempt, because a literal has no type until context gives it one.

**Editor support from the same code.** `klions-lsp` calls the same `analyze()`
the compiler calls. An editor squiggle and a `klions check` error are the same
computation, not two implementations that drift.

## Repository layout

```
crates/
  klions-diagnostics   codes, spans, rendering, JSON output
  klions-lexer         tokenizer, virtual statement terminators
  klions-ast           syntax tree
  klions-parser        recursive descent + Pratt, error recovery
  klions-types         name resolution, inference, shape checking
  klions-frontend      the façade the compiler and the LSP share
  klions-tensor        strided f32 tensors, blocked matmul, PCG32
  klions-autodiff      reverse-mode tape, finite-difference verification
  klions-nn            layers, losses, optimizers, datasets, serialization
  klions-runtime       interpreter and training loop
  klions-platform      TTY detection, SIGINT
  klions-cli           the `klions` binary
  klions-lsp           the `klions-lsp` binary
docs/                  reference, tutorial, diagnostics, ADRs, grammar
examples/              hello, xor, mnist
editors/vscode/        syntax, snippets, LSP client
tests/                 golden programs and a diagnostic corpus
tools/                 dataset generation and MNIST fetch
```

## Testing

```bash
cargo test --workspace
```

178 tests. Of those, 23 verify **every differentiable operation** against
central finite differences to 1e-3 — the release gate that says the autodiff
engine is correct rather than merely plausible. A further corpus of 25 files
asserts that each diagnostic still fires with the code it promises.

## Status and scope

v0.1.0 is an MVP. It does dense feed-forward networks on CPU, and it does them
correctly and reproducibly.

**Here now:** dense layers, dropout, four losses, three optimizers, IDX and CSV
loading, model save/load, compile-time shape checking, a language server, a
formatter, a REPL.

**Deferred, and it says so:** convolutions and normalization (0.2.0), GPU
execution (0.3.0), recurrent and attention layers (0.4.0), distributed
training (0.5.0). Ask for one and the error names the release.

## Documentation

- [Getting started](docs/tutorial/getting-started.md) — empty file to trained model
- [Language reference](docs/reference/language.md)
- [Standard library](docs/reference/stdlib.md)
- [Diagnostic index](docs/diagnostics/index.md)
- [Grammar](docs/grammar.ebnf)
- ADRs: [implementation language](docs/adr/0001-implementation-language.md) ·
  [shapes in types](docs/adr/0002-shapes-in-the-type-system.md)

## Author

Kanak Prabhakar
