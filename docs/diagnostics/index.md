# Diagnostic Index

Every KLIONS diagnostic has a stable code. A code, once assigned, is never
reused for a different error class — so `K1001` will mean "shape mismatch"
for as long as the language exists.

Run `klions explain <code>` for a longer discussion of the common ones.

## Ranges

| Range | Class |
|---|---|
| `K0001`–`K0019` | lexical |
| `K0020`–`K0039` | syntax |
| `K0040`–`K0079` | names and types |
| `K0080`–`K0099` | run time |
| `K1xxx` | `ShapeError` |
| `K2xxx` | `DatasetError` |
| `K3xxx` | `TrainingError` |
| `K4xxx` | `ModelError` |
| `K5xxx` | `DeviceError` |
| `K6xxx` | deferred feature |
| `K9xxx` | internal compiler error |

AI-specific classes have their own ranges deliberately. When a run fails, the
first digit tells you whether the problem is in your shapes, your data, your
training configuration, or your saved model — before you read a word of the
message.

## Lexical

| Code | Meaning |
|---|---|
| `K0001` | source file is not valid UTF-8 |
| `K0002` | unterminated string literal |
| `K0003` | unterminated block comment |
| `K0004` | unknown escape sequence |
| `K0005` | unexpected character |
| `K0006` | non-ASCII identifier |
| `K0007` | reserved keyword |
| `K0008` | malformed number |
| `K0009` | invalid `\u{...}` escape |

## Syntax

| Code | Meaning |
|---|---|
| `K0020` | unexpected token |
| `K0021` | expected a specific token |
| `K0022` | unclosed delimiter |
| `K0023` | unknown layer type |
| `K0024` | unknown training option |
| `K0025` | positional argument after a named one |
| `K0026` | unknown module in an import |
| `K0027` | malformed tensor type |
| `K0028` | expected an expression |
| `K0029` | expected a statement or item |

## Names and types

| Code | Meaning |
|---|---|
| `K0040` | undefined name |
| `K0041` | duplicate definition |
| `K0042` | type mismatch |
| `K0043` | implicit numeric conversion required |
| `K0044` | invalid cast |
| `K0045` | value is not callable |
| `K0046` | wrong argument count |
| `K0047` | unknown named argument |
| `K0048` | not every path returns a value |
| `K0049` | return type mismatch |
| `K0050` | assignment to an immutable binding |
| `K0051` | value is not indexable |
| `K0052` | no such field or method |
| `K0053` | `?` outside a `Result`-returning function |
| `K0054` | `?` error type mismatch |
| `K0055` | condition is not `bool` |
| `K0056` | `break` outside a loop |
| `K0057` | `continue` outside a loop |
| `K0058` | value is not iterable |
| `K0059` | unused variable (warning) |
| `K0060` | unused parameter (warning) |
| `K0061` | format placeholder count mismatch |
| `K0062` | operator not defined on this type |
| `K0063` | no `main` function |
| `K0064` | value is not a model |
| `K0065` | value is not a dataset |
| `K0066` | required training option missing |
| `K0067` | value does not match its shape annotation |

## Run time

| Code | Meaning |
|---|---|
| `K0080` | call-stack limit exceeded |
| `K0081` | integer division by zero |
| `K0082` | integer overflow |
| `K0083` | index out of bounds |
| `K0084` | assertion failed |
| `K0085` | `unwrap` on an error value |
| `K0086` | run-time fault |
| `K0087` | I/O error |

## ShapeError

| Code | Meaning |
|---|---|
| `K1001` | element-wise shape mismatch |
| `K1002` | matrix multiplication shape mismatch |
| `K1003` | reshape element-count mismatch |
| `K1004` | rank out of range |
| `K1005` | axis out of range |
| `K1006` | broadcast failed |
| `K1007` | layer chain width mismatch |
| `K1008` | model input width does not match the dataset |
| `K1009` | negative dimension |

## DatasetError

| Code | Meaning |
|---|---|
| `K2001` | dataset file not found |
| `K2002` | IDX magic number invalid |
| `K2003` | IDX file truncated |
| `K2004` | feature and label counts differ |
| `K2005` | malformed CSV |
| `K2006` | label column out of range |
| `K2007` | dataset is empty |
| `K2008` | path escapes the project root |

## TrainingError

| Code | Meaning |
|---|---|
| `K3001` | non-finite loss or gradient |
| `K3002` | hyperparameter out of range |
| `K3003` | model has no trainable parameters |
| `K3004` | unknown loss function |
| `K3005` | unknown metric |
| `K3006` | loss does not reach any parameter |
| `K3007` | batch larger than the dataset |

## ModelError

| Code | Meaning |
|---|---|
| `K4001` | recursive model reference |
| `K4002` | model has no layers |
| `K4003` | model file magic number invalid |
| `K4004` | unsupported model file version |
| `K4005` | shape manifest mismatch |

## DeviceError

| Code | Meaning |
|---|---|
| `K5001` | unsupported device |
| `K5002` | allocation would exceed the memory ceiling |
| `K5003` | out of memory |

## Deferred features

`K6001` marks a construct the language recognizes but this release does not
implement. The message names the release that will carry it.

These are errors rather than warnings on purpose. Accepting `Conv2D` and
quietly doing nothing would mean your model trained on something other than
what you wrote — and you would have no way to tell.
