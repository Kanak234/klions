# KLIONS Language Reference — v0.1.0 "Ember"

This is the reference for the language as implemented. Where the reference and
the compiler disagree, the compiler is the bug — please report it.

The normative grammar is [`docs/grammar.ebnf`](../grammar.ebnf).

---

## 1. Program structure

Every program has a `main` function. Execution starts there.

```klions
fn main() {
    println("Hello, KLIONS!")
}
```

A file holds imports, then items. The three item kinds are `fn`, `model`, and
`const`. Items may appear in any order — a function may call one declared
below it.

`main` takes no parameters and returns either nothing or `i32`. A returned
`i32` becomes the process exit code.

## 2. Statement terminators

A semicolon ends a statement. So does a newline, when the previous token could
end a statement and the next token could not continue one. Both of these are
valid and mean the same thing:

```klions
let a = 1
let b = 2;
```

An expression can span lines freely, because a trailing operator signals that
more is coming:

```klions
let total = first_term +
            second_term +
            third_term
```

## 3. Bindings

```klions
let x = 5            // immutable
let mut y = 5        // assignable
let z: f32 = 2.5     // annotated
let _ = compute()    // evaluated, not bound
```

Assigning to a name declared without `mut` is an error. Shadowing in an inner
scope is allowed; redefining a name in the same scope is not.

A name that is never read produces a warning. Prefix it with `_` to say the
omission is deliberate.

## 4. Types

| Type | Meaning |
|---|---|
| `i32`, `i64` | signed integers |
| `f32`, `f64` | floating point |
| `bool` | `true` or `false` |
| `string` | UTF-8 text |
| `void` | no value |
| `tensor<f32, D...>` | dense array, dimensions in the type |
| `Result<T, E>` | success or failure |
| `Model`, `Dataset`, `Optimizer`, `Metrics` | library types |

### Numeric conversion is explicit

```klions
let a: i32 = 3
let b: f32 = 1.5
let c = a as f32 + b        // fine
let d = a + b               // error[K0043]
```

Literals are exempt, because a literal has no type of its own until context
gives it one. `let x: i32 = 1` is exact, not a conversion.

### Tensor types carry shape

```klions
let a: tensor<f32, 64, 784> = zeros(64, 784)
let b: tensor<f32, _, 10>   = zeros(32, 10)   // `_` is decided at run time
```

Dimensions written as integers are checked at compile time. `_` defers to run
time — use it when a size depends on a file you have not read yet.

## 5. Operators

Lowest precedence first:

| Level | Operators | Associativity |
|---|---|---|
| 1 | `\|\|` | left |
| 2 | `&&` | left |
| 3 | `==` `!=` | left |
| 4 | `<` `<=` `>` `>=` | left |
| 5 | `+` `-` | left |
| 6 | `*` `/` `%` | left |
| 7 | `@` | left |
| 8 | unary `-` `!`, postfix `.` `()` `[]` `?` | — |

`@` is matrix multiplication and binds tighter than `*`, so `a @ b * c` means
`(a @ b) * c`. `*` on tensors is element-wise.

`&&` and `||` short-circuit: the right operand is not evaluated when the left
decides the result.

Integer overflow and integer division by zero are errors, not silent wrapping.
Float division by zero follows IEEE 754 and yields infinity or NaN.

## 6. Control flow

```klions
if x > 10 {
    println("big")
} else if x > 5 {
    println("medium")
} else {
    println("small")
}

while running { step() }

for i in 0..10 { println("{}", i) }
for row in matrix { process(row) }
```

Conditions must be `bool`. There is no truthiness: write `if n != 0`, not
`if n`.

## 7. Functions

```klions
fn area(width: f64, height: f64) -> f64 {
    return width * height
}
```

Parameter types and the return type are required when there is a return value.
Every path through a function with a return type must return.

Arguments may be passed by name, and named arguments come after positional
ones:

```klions
let a = area(3.0, height=4.0)
```

## 8. Models

A model is an ordered chain of layers. Declaration order is forward-pass order.

```klions
model Classifier {
    hidden = Dense(inputs=784, outputs=128, activation="relu")
    drop   = Dropout(p=0.2)
    output = Dense(inputs=128, outputs=10)
}
```

Layer widths are checked against each other when the model is declared. If
`hidden` produces 128 outputs, the next layer with an `inputs` must ask for
128.

### Layers in v0.1.0

| Layer | Arguments |
|---|---|
| `Dense` | `inputs`, `outputs`, `activation` (optional) |
| `ReLU`, `Sigmoid`, `Tanh`, `Softmax` | none |
| `Flatten` | none |
| `Dropout` | `p` |

Activations: `none`, `relu`, `sigmoid`, `tanh`, `softmax`.

Layers scheduled for later releases (`Conv2D`, `LSTM`, `TransformerBlock`, and
others) are recognized and rejected with the release that will carry them.
They are never silently ignored.

## 9. Training

```klions
train Classifier using training_data {
    epochs = 12,
    batch_size = 64,
    optimizer = Adam(0.001),
    loss = "cross_entropy",
    metrics = ["accuracy"],
    validation_split = 0.1,
    verbose = true
}
```

`epochs` is required — there is no default worth guessing. Everything else has
one.

Losses: `mse`, `mae`, `cross_entropy`, `binary_cross_entropy`.
Optimizers: `SGD`, `Adam`, `RMSProp`, with the defaults from their papers.

`cross_entropy` takes raw logits and applies log-sum-exp internally. Do **not**
put a `Softmax` layer in front of it — that is the single most common way to
build a quietly wrong model, and here it is unnecessary.

## 10. Gradients

Gradient tracking is on during training and off everywhere else. `no_grad`
suspends it explicitly:

```klions
no_grad {
    let prediction = Classifier.predict(sample)
}
```

`predict` already runs without tracking, so `no_grad` is for cases where you
compute something by hand.

## 11. Errors

Operations that can fail return `Result<T, E>`.

```klions
let data = Dataset::from_csv("train.csv", 2).unwrap()
```

`unwrap` stops the program with the original error message. `?` propagates the
error to the caller, and requires the enclosing function to return a `Result`.
`is_ok` and `is_err` let you branch.

## 12. Formatting output

```klions
println("{} scored {:.2}", name, score)
println("{{literal braces}}")
```

The number of `{}` placeholders must match the number of arguments; a mismatch
is a compile error, not a run-time surprise.

## 13. Reserved words

`struct`, `enum`, `trait`, `impl`, `match`, `pub`, `async`, `await`, `unsafe`,
`device`, and `spawn` are reserved. Using one as a name reports the release
that will give it meaning.
