# KLIONS Standard Library — v0.1.0

Everything here is built into the toolchain. There is nothing to install.

## Output

| Function | Returns | Notes |
|---|---|---|
| `print(fmt, ...)` | `void` | no trailing newline |
| `println(fmt, ...)` | `void` | trailing newline |
| `format(fmt, ...)` | `string` | builds a string instead of printing |
| `assert(cond, msg?)` | `void` | stops the program when `cond` is false |

Placeholders are `{}`, or `{:.N}` for N decimal places. Write `{{` for a
literal brace. The placeholder count is checked at compile time.

## Tensor construction

| Function | Result |
|---|---|
| `zeros(d...)` | every element 0 |
| `ones(d...)` | every element 1 |
| `full(d..., v)` | every element `v` |
| `eye(n)` | n×n identity |
| `arange(start, end, step?)` | evenly spaced values |
| `rand(d...)` | uniform on [0, 1) |
| `randn(d...)` | standard normal |
| `tensor([...])` | from a nested array literal |
| `one_hot(labels, classes)` | indicator rows |

`rand` and `randn` draw from the single global generator, seeded by `--seed`.
The same seed reproduces a run exactly.

## Tensor methods

| Method | Result |
|---|---|
| `.shape()` | dimensions as a list |
| `.rank()`, `.len()`, `.numel()` | integers |
| `.item()` | the single element of a size-1 tensor |
| `.reshape(d...)` | one dimension may be `-1` and is inferred |
| `.flatten()` | rank-1 view |
| `.t()`, `.transpose(a, b)` | axis swap |
| `.sum()`, `.mean()`, `.max()`, `.min()` | scalar, or per-axis with an argument |
| `.argmax()`, `.argmin()` | index of the extreme element |
| `.relu()`, `.sigmoid()`, `.tanh()`, `.softmax()` | element-wise |
| `.exp()`, `.ln()`, `.sqrt()`, `.abs()` | element-wise |
| `.to_vec()` | contents as a flat array |

Free-function forms exist too: `sum(t)` and `t.sum()` are the same.

### Broadcasting

Element-wise operations align shapes from the right. A dimension of 1 is
stretched; anything else must match.

```
[64, 128] + [128]     -> [64, 128]
[64, 1]   + [64, 10]  -> [64, 10]
[64, 128] + [64, 10]  -> error[K1001], axis 1
```

## Datasets

| Constructor | Result |
|---|---|
| `Dataset::from_idx(images, labels)` | `Result<Dataset, DatasetError>` |
| `Dataset::from_csv(path, label_column, has_header?)` | `Result<Dataset, DatasetError>` |

Paths resolve relative to the source file, not the shell's working directory,
and may not escape the project root.

| Method | Result |
|---|---|
| `.batch(n)` | sets the batch size |
| `.shuffle()` | permutes using the global generator |
| `.normalize()` | scales features into [0, 1] |
| `.standardize()` | zero mean, unit variance |
| `.one_hot(classes)` | widens integer labels |
| `.split(fraction)` | the leading fraction |
| `.len()` | sample count |
| `.features()`, `.labels()` | the underlying tensors |
| `.feature_width()`, `.label_width()` | integers |
| `.sample(i)`, `.label(i)` | one row |

Datasets are loaded once and batches are cut from that copy; a batch never
re-reads the file.

## Models

| Method | Result |
|---|---|
| `.predict(x)` | forward pass, gradients off |
| `.evaluate(dataset)` | `Metrics` with `.loss`, `.accuracy`, `.samples` |
| `.save(path)` | `Result<void, ModelError>` |
| `.load(path)` | `Result<void, ModelError>` |
| `.summary()` | printable layer table |
| `.parameter_count()` | total trainable parameters |

Saved models use a versioned format with a magic number and a shape manifest,
so loading into a differently-shaped model fails loudly.

## Optimizers

| Constructor | Defaults |
|---|---|
| `SGD(lr?, momentum?)` | lr 0.01, momentum 0.0 |
| `Adam(lr?)` | lr 0.001, β₁ 0.9, β₂ 0.999, ε 1e-8 |
| `RMSProp(lr?)` | lr 0.001, decay 0.9, ε 1e-8 |

## Maths

`abs`, `sqrt`, `exp`, `ln`, `pow`, `floor`, `ceil`, `round` work on numbers and
element-wise on tensors.

## Conversion and timing

`int(x)`, `float(x)`, `str(x)`, `len(x)`, `shape(t)`, `rank(t)`, `time_ms()`.
