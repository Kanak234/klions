# ADR 0002 — Tensor shapes belong in the type system

**Status:** accepted · **Date:** 2026-08-01

## Context

The failure that motivates KLIONS is specific and common: a shape mismatch in
a neural network surfaces minutes into a training run, as a traceback through
library code the user did not write, naming tensors the user did not name. The
information needed to catch it — the layer widths, the batch dimension — was
present in the source the whole time.

## Decision

Tensor dimensions are part of the type. `tensor<f32, 64, 784>` and
`tensor<f32, 64, 10>` are different types, and the checker rejects
`a + b` between them before any code runs. Model layer chains are checked at
declaration.

An underscore, `tensor<f32, _, 784>`, defers one dimension to run time.

## Rationale

**The information exists at compile time.** `Dense(inputs=784, outputs=128)`
is a literal. So is the next layer's `inputs`. Comparing two integers that are
both written in the source is not an advanced type-system feature; declining to
compare them is a choice, and it is the wrong one.

**Diagnostics can name the axis.** Because the checker holds both shapes, it
can say "axis 1 disagrees: 128 on the left, 10 on the right" rather than
"shapes not aligned". The difference between those two messages is roughly the
difference between fixing the bug in ten seconds and fixing it in ten minutes.

**Partial information is still useful.** A dataset's row count is unknown until
the file is read, so requiring every dimension to be static would make the
feature useless in practice. `_` marks exactly the dimensions that genuinely
cannot be known, and everything else stays checked. In the MNIST example the
batch dimension is `_` and the feature dimension is 784 — and 784 is where the
mistakes happen.

## Alternatives considered

**Full dependent types.** Dimensions as arbitrary compile-time expressions,
so `concat` could be typed precisely. Rejected as far too much machinery for
the value: it would demand a solver, and the error messages a solver produces
are much worse than the ones a direct comparison produces. The whole point is
better messages.

**Runtime-only checking, as in PyTorch and TensorFlow.** Rejected — it is the
status quo this language exists to improve on.

**A separate linter.** Rejected because a separate tool is a tool people
forget to run, and because it would need its own parser and its own type
model, which would then drift from the compiler's.

## Consequences

- Shape errors that would have taken minutes of training to surface now appear
  in the editor as you type, via the same checker the compiler runs.
- Programs that reshape dynamically must use `_` and accept run-time checking.
  The run-time errors carry the same codes and the same message quality.
- The checker must model broadcasting precisely, including the right-alignment
  rule and the stretching of unit dimensions. This is implemented once, in
  `klions-types::ty::broadcast`, and shared with the tensor engine's own
  runtime check so the two cannot disagree about what is legal.
- Adding an operator means teaching the checker its shape rule. This is real
  work, and it is the work that makes the language worth using.
