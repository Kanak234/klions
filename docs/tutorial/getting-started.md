# Getting Started with KLIONS

This walks from an empty file to a trained classifier. It assumes no prior
machine-learning experience, and about twenty minutes.

## 1. Build the toolchain

```bash
cargo build --release
```

One command, no system dependencies, no BLAS, no Python. Two binaries land in
`target/release`: `klions` and `klions-lsp`.

## 2. Hello

Put this in `hello.kl`:

```klions
fn main() {
    println("Hello, KLIONS!")
}
```

```bash
klions run hello.kl
```

## 3. A first tensor

A tensor is a grid of numbers. Its shape is part of its type, which is what
lets the compiler catch mistakes before anything runs.

```klions
fn main() {
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    println("shape: {}", a.shape())
    println("sum:   {}", a.sum())
    println("a @ a:\n{}", a @ a)
}
```

`@` is matrix multiplication. `*` multiplies element by element — they are
different operations, so they get different symbols.

## 4. Watch the compiler catch a shape error

Try this deliberately wrong program:

```klions
fn main() {
    let a: tensor<f32, 4, 3> = zeros(4, 3)
    let b: tensor<f32, 4, 5> = zeros(4, 5)
    let c = a + b
}
```

```
error[K1001]: shape mismatch in `+`: left is [4, 3], right is [4, 5]
  --> shapes.kl:4:13
   |
 4 |     let c = a + b
   |             ^^^^^
   |
   = note: axis 1 disagrees: 3 on the left, 5 on the right
   = help: element-wise operations compare shapes from the right; a dimension
           of 1 is stretched, anything else must match exactly
```

No tensor was allocated. No epoch ran. This is the whole point of the
language: in most stacks that error surfaces minutes into training, as a
traceback through library code you did not write.

## 5. A model

A model is a list of layers. They run top to bottom.

```klions
model Net {
    hidden = Dense(inputs=2, outputs=8, activation="tanh")
    output = Dense(inputs=8, outputs=1, activation="sigmoid")
}
```

`Dense` means every input connects to every output. `inputs` must match what
the previous layer produces — and the compiler checks that too:

```
error[K1007]: layer `output` expects 4 inputs but `hidden` produces 8 outputs
   = help: set `inputs=8` on `output`
```

## 6. Data

Put this in `data/xor.csv`:

```
x1,x2,label
0,0,0
0,1,1
1,0,1
1,1,0
```

```klions
let data = Dataset::from_csv("data/xor.csv", 2).unwrap()
```

The `2` says which column holds the label. The path is relative to your source
file, not to wherever you happen to run the command from.

## 7. Training

```klions
train Net using data {
    epochs = 400,
    batch_size = 4,
    optimizer = Adam(0.05),
    loss = "mse"
}
```

That is the whole loop. No gradient bookkeeping, no optimizer wiring, no
manual zeroing of gradients between steps.

## 8. The complete program

```klions
model XorNet {
    hidden = Dense(inputs=2, outputs=8, activation="tanh")
    output = Dense(inputs=8, outputs=1, activation="sigmoid")
}

fn main() {
    let data = Dataset::from_csv("data/xor.csv", 2).unwrap()

    train XorNet using data {
        epochs = 400,
        batch_size = 4,
        optimizer = Adam(0.05),
        loss = "mse",
        verbose = false
    }

    for i in 0..4 {
        let x = data.sample(i)
        let y = XorNet.predict(x)
        println("  {} XOR {} -> {:.3}  (expected {})",
                x[0, 0], x[0, 1], y[0, 0], data.label(i))
    }
}
```

```
  0.0 XOR 0.0 -> 0.006  (expected 0.0)
  0.0 XOR 1.0 -> 0.993  (expected 1.0)
  1.0 XOR 0.0 -> 0.993  (expected 1.0)
  1.0 XOR 1.0 -> 0.007  (expected 0.0)
```

XOR cannot be solved by a single layer. The hidden layer is what makes it
work, and getting these four numbers right is the classic proof that
backpropagation is functioning.

## 9. Reproducibility

```bash
klions run xor.kl --seed 42
```

The same seed gives bit-identical results. When you report a number, others can
reproduce it.

## 10. Where to go next

- `examples/mnist.kl` — digit classification with dropout and Adam
- `docs/reference/language.md` — the full language
- `docs/reference/stdlib.md` — every built-in function
- `klions explain K1001` — any diagnostic, explained at length
