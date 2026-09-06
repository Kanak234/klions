//! The `train` statement — FR-AI-008, FR-AI-014 … FR-AI-018, FR-RT-011.
//!
//! One epoch is: shuffle, iterate batches, forward, loss, backward, step,
//! clear the tape. The tape is cleared after every step (FR-AI-005) so memory
//! stays flat across an arbitrarily long run.

use std::collections::HashMap;
use std::rc::Rc;

use klions_ast::{Expr, ExprKind, TrainOption};
use klions_autodiff::{self as ad, Var};
use klions_diagnostics::{codes, Span};
use klions_nn::data::Dataset;
use klions_nn::loss::{accuracy, Loss};
use klions_nn::optim::{default_adam, Optimizer, OptimizerKind};
use klions_nn::{EvalResult, Model};
use klions_tensor::Tensor;

use crate::builtins;
use crate::value::{RuntimeError, Value};
use crate::Interpreter;

struct Settings {
    epochs: usize,
    batch_size: usize,
    loss: Loss,
    optimizer: OptimizerKind,
    metrics: Vec<String>,
    validation_split: f32,
    verbose: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            epochs: 1,
            batch_size: 32,
            loss: Loss::Mse,
            optimizer: default_adam(),
            metrics: vec!["loss".to_string()],
            validation_split: 0.0,
            verbose: true,
        }
    }
}

pub fn run_training(
    interp: &mut Interpreter,
    model_name: &str,
    model_span: Span,
    dataset_name: &str,
    dataset_span: Span,
    options: &[TrainOption],
    span: Span,
) -> Result<(), RuntimeError> {
    let model = interp
        .model_named(model_name)
        .or_else(|| match interp.eval(&ident(model_name, model_span)).ok() {
            Some(Value::Model(m)) => Some(m),
            _ => None,
        })
        .ok_or_else(|| {
            RuntimeError::new(
                codes::NOT_A_MODEL,
                model_span,
                format!("`{}` is not a model", model_name),
            )
        })?;

    let dataset = match interp.eval(&ident(dataset_name, dataset_span))? {
        Value::Dataset(d) => d,
        other => {
            return Err(RuntimeError::new(
                codes::NOT_A_DATASET,
                dataset_span,
                format!("`{}` is a `{}`, not a dataset", dataset_name, other.type_name()),
            ))
        }
    };

    let settings = read_settings(interp, options)?;
    train_model(interp, &model, &dataset, &settings, span, model_name)
}

fn ident(name: &str, span: Span) -> Expr {
    Expr { kind: ExprKind::Ident(name.to_string()), span }
}

fn read_settings(
    interp: &mut Interpreter,
    options: &[TrainOption],
) -> Result<Settings, RuntimeError> {
    let mut s = Settings::default();
    let mut given: HashMap<&str, ()> = HashMap::new();

    for o in options {
        let v = interp.eval(&o.value)?;
        match o.key.as_str() {
            "epochs" => {
                s.epochs = v.as_int(o.value.span)?.max(0) as usize;
                given.insert("epochs", ());
            }
            "batch_size" => s.batch_size = v.as_int(o.value.span)?.max(1) as usize,
            "loss" => s.loss = builtins::loss_from_name(&v.as_str(o.value.span)?, o.value.span)?,
            "optimizer" => match v {
                Value::Optimizer(k) => s.optimizer = (*k).clone(),
                Value::Str(name) => {
                    s.optimizer = match name.as_str() {
                        "Adam" | "adam" => default_adam(),
                        "SGD" | "sgd" => klions_nn::optim::default_sgd(),
                        "RMSProp" | "rmsprop" => klions_nn::optim::default_rmsprop(),
                        other => {
                            return Err(RuntimeError::new(
                                codes::UNDEFINED_NAME,
                                o.value.span,
                                format!("unknown optimizer `{}`", other),
                            )
                            .with_help("available: SGD, Adam, RMSProp"))
                        }
                    }
                }
                other => {
                    return Err(RuntimeError::new(
                        codes::TYPE_MISMATCH,
                        o.value.span,
                        format!("`optimizer` expects an optimizer, found `{}`", other.type_name()),
                    ))
                }
            },
            "metrics" => {
                if let Value::Array(items) = v {
                    s.metrics = items.iter().map(|x| x.display()).collect();
                }
            }
            "validation_split" => s.validation_split = v.as_float(o.value.span)? as f32,
            "verbose" => s.verbose = v.truthy(o.value.span)?,
            _ => {}
        }
    }
    Ok(s)
}

fn train_model(
    interp: &mut Interpreter,
    model: &Rc<std::cell::RefCell<Model>>,
    dataset: &Rc<Dataset>,
    s: &Settings,
    span: Span,
    model_name: &str,
) -> Result<(), RuntimeError> {
    let params = model.borrow().params();
    if params.is_empty() {
        return Err(RuntimeError::new(
            codes::NO_PARAMETERS,
            span,
            format!("model `{}` has no trainable parameters", model_name),
        )
        .with_help("a model needs at least one Dense layer to learn anything"));
    }

    // FR-TYP-013 at run time: the dataset's width is only known now.
    if let Some(w) = model.borrow().input_width() {
        let dw = dataset.feature_width();
        if w != dw {
            return Err(RuntimeError::new(
                codes::MODEL_INPUT_MISMATCH,
                span,
                format!(
                    "model `{}` expects {} input features but dataset `{}` provides {}",
                    model_name, w, dataset.name, dw
                ),
            )
            .with_help(format!(
                "set the first layer's `inputs={}`, or reshape the dataset before training",
                dw
            )));
        }
    }

    // Cross-entropy needs one-hot targets shaped like the model's output.
    let prepared = prepare_targets(model, dataset, s, span)?;
    let (train_set, valid_set) = if s.validation_split > 0.0 {
        let (v, t) = prepared.split(s.validation_split);
        (t, Some(v))
    } else {
        (prepared, None)
    };

    let mut opt = Optimizer::new(s.optimizer.clone());
    let started = klions_platform::now_ms();

    if s.verbose {
        let _ = writeln!(
            interp.out,
            "training {} — {} samples, {} parameters, {}",
            model_name,
            train_set.len(),
            model.borrow().param_count(),
            s.optimizer.describe()
        );
    }

    for epoch in 1..=s.epochs {
        // FR-AI-014: shuffling draws from the seeded global generator, so a
        // fixed seed reproduces the run exactly.
        let shuffled = train_set.shuffle(&mut interp.rng).batch(s.batch_size);
        let batches = shuffled.batch_count();
        let mut epoch_loss = 0.0f64;
        let mut epoch_acc = 0.0f64;
        let mut seen = 0usize;

        for bi in 0..batches {
            // FR-RT-011: interruption unwinds between batches, never mid-update.
            if klions_platform::interrupted() {
                let _ = writeln!(interp.out);
                let _ = writeln!(
                    interp.out,
                    "interrupted after {} epochs; parameters retain their last completed update",
                    epoch - 1
                );
                ad::clear_tape();
                return Err(RuntimeError::interrupted(span));
            }

            let (x, y) = shuffled
                .get_batch(bi)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;

            let step = run_batch(model, &x, &y, s, &params, &mut opt, &mut interp.rng, span)?;
            epoch_loss += step.loss as f64 * x.dim(0) as f64;
            epoch_acc += step.accuracy as f64 * x.dim(0) as f64;
            seen += x.dim(0);

            if s.verbose && klions_platform::stdout_is_tty() && batches > 1 {
                let _ = write!(
                    interp.out,
                    "\r  epoch {}/{}  batch {}/{}  loss {:.4}",
                    epoch, s.epochs, bi + 1, batches, step.loss
                );
                let _ = interp.out.flush();
            }
        }

        if seen == 0 {
            continue;
        }
        let mean_loss = epoch_loss / seen as f64;
        let mean_acc = epoch_acc / seen as f64;

        if s.verbose {
            if klions_platform::stdout_is_tty() && batches > 1 {
                let _ = write!(interp.out, "\r");
            }
            let mut line = format!(
                "  epoch {:>3}/{:<3}  loss {:.4}",
                epoch, s.epochs, mean_loss
            );
            if s.metrics.iter().any(|m| m == "accuracy") {
                line.push_str(&format!("  accuracy {:.4}", mean_acc));
            }
            if let Some(v) = &valid_set {
                let r = evaluate(interp, &model.borrow(), &Rc::new(v.clone()), span)?;
                line.push_str(&format!("  val_loss {:.4}", r.loss));
                if s.metrics.iter().any(|m| m == "accuracy") {
                    line.push_str(&format!("  val_accuracy {:.4}", r.accuracy));
                }
            }
            let _ = writeln!(interp.out, "{}{}", line, " ".repeat(8));
        }
    }

    if s.verbose {
        let elapsed = klions_platform::now_ms() - started;
        let _ = writeln!(
            interp.out,
            "training finished in {:.2}s",
            elapsed as f64 / 1000.0
        );
    }
    ad::clear_tape();
    Ok(())
}

struct StepResult {
    loss: f32,
    accuracy: f32,
}

#[allow(clippy::too_many_arguments)]
fn run_batch(
    model: &Rc<std::cell::RefCell<Model>>,
    x: &Tensor,
    y: &Tensor,
    s: &Settings,
    params: &[ad::Param],
    opt: &mut Optimizer,
    rng: &mut klions_tensor::Rng,
    span: Span,
) -> Result<StepResult, RuntimeError> {
    // FR-AI-005: a fresh tape per step; nothing survives into the next one.
    ad::clear_tape();
    let prev = ad::set_tracking(true);

    let result = (|| -> Result<StepResult, RuntimeError> {
        let input = Var::constant(x.clone());
        let pred = model
            .borrow()
            .forward(&input, true, rng)
            .map_err(|e| RuntimeError::from_tensor(e, span))?;

        let l = s
            .loss
            .compute(&pred, y)
            .map_err(|e| RuntimeError::from_tensor(e, span))?;
        let loss_value = l.value.item().map_err(|e| RuntimeError::from_tensor(e, span))?;

        // FR-AI-018: a non-finite loss stops the run with an explanation
        // rather than silently poisoning every parameter with NaN.
        if !loss_value.is_finite() {
            return Err(RuntimeError::new(
                codes::NON_FINITE_GRADIENT,
                span,
                format!("loss became {} during training", klions_tensor::fmt_f32(loss_value)),
            )
            .with_note("this usually means the learning rate is too high, or the inputs are unscaled")
            .with_help(
                "try a smaller learning rate, or normalize the dataset with `.normalize()`",
            ));
        }

        let root = l.node.ok_or_else(|| {
            RuntimeError::new(
                codes::NO_GRAD_PATH,
                span,
                "the loss does not depend on any trainable parameter",
            )
            .with_help(
                "check that the model's layers are reached by the forward pass, and that the \
                 computation is not inside `no_grad`",
            )
        })?;

        let grads = ad::backward(root, None).map_err(|e| RuntimeError::from_tensor(e, span))?;

        // Collect each parameter's gradient by the leaf node it registered.
        let mut collected: Vec<Option<Tensor>> = Vec::with_capacity(params.len());
        for p in params {
            let node = *p.node.borrow();
            let g = node.and_then(|n| grads.get(n).cloned().flatten());
            if let Some(g) = &g {
                if !ad::all_finite(g) {
                    return Err(RuntimeError::new(
                        codes::NON_FINITE_GRADIENT,
                        span,
                        format!("gradient for `{}` contains non-finite values", p.name),
                    )
                    .with_note("NaN or infinity in a gradient makes every later update meaningless")
                    .with_help("lower the learning rate, or normalize the inputs"));
                }
            }
            collected.push(g);
        }

        opt.step(params, &collected)
            .map_err(|e| RuntimeError::from_tensor(e, span))?;

        let acc = accuracy(&pred.value, y).unwrap_or(0.0);
        Ok(StepResult { loss: loss_value, accuracy: acc })
    })();

    ad::set_tracking(prev);
    ad::clear_tape();
    result
}

/// Widen integer labels to one-hot when the loss needs a distribution.
fn prepare_targets(
    model: &Rc<std::cell::RefCell<Model>>,
    dataset: &Rc<Dataset>,
    s: &Settings,
    span: Span,
) -> Result<Dataset, RuntimeError> {
    let out_width = model.borrow().output_width().unwrap_or(0);
    let needs_one_hot = matches!(s.loss, Loss::CrossEntropy)
        && dataset.label_width() == 1
        && out_width > 1;
    if needs_one_hot {
        return dataset
            .one_hot(out_width)
            .map_err(|e| RuntimeError::from_tensor(e, span));
    }
    if dataset.label_width() != out_width && out_width > 0 && dataset.label_width() > 1 {
        return Err(RuntimeError::new(
            codes::SHAPE_MISMATCH,
            span,
            format!(
                "model `{}` produces {} outputs but the dataset's labels are {} wide",
                model.borrow().name,
                out_width,
                dataset.label_width()
            ),
        )
        .with_help("match the final layer's `outputs` to the label width"));
    }
    // A single-output model with scalar labels needs the label column shaped
    // [n, 1] so it lines up with the prediction.
    if out_width == 1 && dataset.label_width() == 1 {
        let mut d = (**dataset).clone();
        if d.labels.rank() == 1 {
            d.labels = d
                .labels
                .reshape(&[d.labels.dim(0) as isize, 1])
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
        }
        return Ok(d);
    }
    Ok((**dataset).clone())
}

/// FR-AI-016: `model.evaluate(dataset)`.
pub fn evaluate(
    interp: &mut Interpreter,
    model: &Model,
    dataset: &Rc<Dataset>,
    span: Span,
) -> Result<EvalResult, RuntimeError> {
    let out_width = model.output_width().unwrap_or(0);
    let prepared = if dataset.label_width() == 1 && out_width > 1 {
        dataset
            .one_hot(out_width)
            .map_err(|e| RuntimeError::from_tensor(e, span))?
    } else if out_width == 1 && dataset.labels.rank() == 1 {
        let mut d = (**dataset).clone();
        d.labels = d
            .labels
            .reshape(&[d.labels.dim(0) as isize, 1])
            .map_err(|e| RuntimeError::from_tensor(e, span))?;
        d
    } else {
        (**dataset).clone()
    };

    let batched = prepared.batch(256);
    let mut total_loss = 0.0f64;
    let mut total_acc = 0.0f64;
    let mut seen = 0usize;

    // Inference runs with tracking off, so no tape is built (FR-AI-004).
    ad::no_grad(|| -> Result<(), RuntimeError> {
        for bi in 0..batched.batch_count() {
            let (x, y) = batched
                .get_batch(bi)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
            let pred = model
                .forward(&Var::constant(x.clone()), false, &mut interp.rng)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
            let l = Loss::Mse
                .compute(&pred, &y)
                .map_err(|e| RuntimeError::from_tensor(e, span))?;
            let lv = l.value.item().map_err(|e| RuntimeError::from_tensor(e, span))?;
            let n = x.dim(0);
            total_loss += lv as f64 * n as f64;
            total_acc += accuracy(&pred.value, &y).unwrap_or(0.0) as f64 * n as f64;
            seen += n;
        }
        Ok(())
    })?;

    let n = seen.max(1) as f64;
    Ok(EvalResult {
        loss: (total_loss / n) as f32,
        accuracy: (total_acc / n) as f32,
        samples: seen,
    })
}

use std::io::Write;
