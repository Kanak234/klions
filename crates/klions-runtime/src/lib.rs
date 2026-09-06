//! KLIONS runtime — FR-RT-001 … FR-RT-012.
//!
//! A tree-walking interpreter. Speed here is not the point: the tensor kernels
//! carry the numerical work, and the interpreter only sequences them.

pub mod builtins;
pub mod train;
pub mod value;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use klions_ast::*;
use klions_diagnostics::{codes, Span};
use klions_nn::{Activation, Layer, Model};
use klions_tensor::Rng;

pub use value::{RuntimeError, Value};

/// FR-RT-005: the call-stack limit, reported rather than crashing.
pub const MAX_FRAMES: usize = 4096;

/// Native stack reserved for the interpreter thread.
///
/// A tree-walking interpreter consumes native stack proportional to KLIONS
/// call depth. Running on the default thread stack would let a runaway
/// recursion abort the process before [`MAX_FRAMES`] could report it, which
/// is exactly the failure FR-RT-005 exists to prevent. Reserving the stack
/// up front makes the frame limit the thing that actually triggers.
pub const INTERPRETER_STACK_BYTES: usize = 512 * 1024 * 1024;

/// Run `f` on a thread with [`INTERPRETER_STACK_BYTES`] of stack.
pub fn on_interpreter_stack<F, T>(f: F) -> std::thread::Result<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .name("klions-interpreter".to_string())
        .stack_size(INTERPRETER_STACK_BYTES)
        .spawn(f)
        .expect("failed to spawn the interpreter thread")
        .join()
}

pub struct Interpreter {
    program: Rc<Program>,
    globals: HashMap<String, Value>,
    scopes: Vec<HashMap<String, Value>>,
    models: HashMap<String, Rc<std::cell::RefCell<Model>>>,
    functions: HashMap<String, Rc<FunctionDecl>>,
    pub rng: Rng,
    frames: usize,
    /// Directory of the source file, for EIR-017 dataset resolution.
    pub source_dir: PathBuf,
    pub project_root: Option<PathBuf>,
    pub out: Box<dyn std::io::Write>,
    pub verbose: bool,
}

/// Non-local control flow inside the interpreter.
pub enum Flow {
    Normal,
    Break,
    Continue,
    Return(Value),
}

pub type ExecResult = Result<Flow, RuntimeError>;
pub type EvalResult = Result<Value, RuntimeError>;

impl Interpreter {
    pub fn new(program: Program, seed: u64, source_dir: PathBuf) -> Interpreter {
        Interpreter {
            program: Rc::new(program),
            globals: HashMap::new(),
            scopes: vec![HashMap::new()],
            models: HashMap::new(),
            functions: HashMap::new(),
            rng: Rng::new(seed),
            frames: 0,
            source_dir,
            project_root: None,
            out: Box::new(std::io::stdout()),
            verbose: true,
        }
    }

    /// FR-RT-001: build the global environment, then call `main`.
    pub fn run(&mut self) -> Result<i32, RuntimeError> {
        klions_platform::install_interrupt_handler();
        let program = Rc::clone(&self.program);

        // Models first: constants may not reference them, but functions may.
        for item in &program.items {
            if let Item::Model(m) = item {
                let built = self.build_model(m)?;
                self.models
                    .insert(m.name.clone(), Rc::new(std::cell::RefCell::new(built)));
            }
        }
        for item in &program.items {
            match item {
                Item::Function(f) => {
                    self.functions.insert(f.name.clone(), Rc::new(f.clone()));
                }
                Item::Const(c) => {
                    let v = self.eval(&c.value)?;
                    self.globals.insert(c.name.clone(), v);
                }
                Item::Model(_) => {}
            }
        }

        let main = self
            .functions
            .get("main")
            .cloned()
            .ok_or_else(|| RuntimeError::new(codes::MISSING_MAIN, Span::new(0, 0), "no `main` function"))?;

        let v = self.call_function(&main, Vec::new(), main.span)?;
        // FR-RT-004: an i32 return becomes the process exit code.
        Ok(match v {
            Value::Int(i) => i as i32,
            _ => 0,
        })
    }

    // ---------------- environment ----------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, v: Value) {
        if name == "_" {
            return;
        }
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn get(&self, name: &str) -> Option<Value> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        self.globals.get(name).cloned()
    }

    fn assign(&mut self, name: &str, v: Value) -> bool {
        for s in self.scopes.iter_mut().rev() {
            if s.contains_key(name) {
                s.insert(name.to_string(), v);
                return true;
            }
        }
        if self.globals.contains_key(name) {
            self.globals.insert(name.to_string(), v);
            return true;
        }
        false
    }

    // ---------------- models ----------------

    fn build_model(&mut self, decl: &ModelDecl) -> Result<Model, RuntimeError> {
        let mut model = Model::new(decl.name.clone());
        for binding in &decl.layers {
            let l = &binding.layer;
            let mut ints: HashMap<&str, i64> = HashMap::new();
            let mut floats: HashMap<&str, f64> = HashMap::new();
            let mut strs: HashMap<&str, String> = HashMap::new();
            let positional = ["inputs", "outputs", "activation"];

            for (i, a) in l.args.iter().enumerate() {
                let key: &str = match &a.name {
                    Some(n) => match positional.iter().find(|p| *p == n) {
                        Some(p) => p,
                        None if l.kind == "Dropout" && n == "p" => "p",
                        None => continue,
                    },
                    None => {
                        if l.kind == "Dropout" {
                            "p"
                        } else {
                            match positional.get(i) {
                                Some(p) => p,
                                None => continue,
                            }
                        }
                    }
                };
                match &a.value.kind {
                    ExprKind::Int(v) => {
                        ints.insert(key, *v);
                        floats.insert(key, *v as f64);
                    }
                    ExprKind::Float(v) => {
                        floats.insert(key, *v);
                    }
                    ExprKind::Str(s) => {
                        strs.insert(key, s.clone());
                    }
                    _ => {
                        let v = self.eval(&a.value)?;
                        match v {
                            Value::Int(i) => {
                                ints.insert(key, i);
                                floats.insert(key, i as f64);
                            }
                            Value::Float(f) => {
                                floats.insert(key, f);
                            }
                            Value::Str(s) => {
                                strs.insert(key, s);
                            }
                            _ => {}
                        }
                    }
                }
            }

            let layer = match l.kind.as_str() {
                "Dense" => {
                    let inputs = ints.get("inputs").copied().unwrap_or(0) as usize;
                    let outputs = ints.get("outputs").copied().unwrap_or(0) as usize;
                    let act = strs
                        .get("activation")
                        .and_then(|s| Activation::parse(s))
                        .unwrap_or(Activation::None);
                    Layer::dense(&binding.name, inputs, outputs, act, &mut self.rng)
                        .map_err(|e| RuntimeError::from_tensor(e, l.span))?
                }
                "ReLU" => Layer::ReLU,
                "Sigmoid" => Layer::Sigmoid,
                "Tanh" => Layer::Tanh,
                "Softmax" => Layer::Softmax,
                "Flatten" => Layer::Flatten,
                "Dropout" => Layer::Dropout {
                    p: floats.get("p").copied().unwrap_or(0.5) as f32,
                },
                other => {
                    return Err(RuntimeError::new(
                        codes::UNKNOWN_LAYER,
                        l.kind_span,
                        format!("unknown layer `{}`", other),
                    ))
                }
            };
            model.push(binding.name.clone(), layer);
        }
        Ok(model)
    }

    // ---------------- functions ----------------

    fn call_function(
        &mut self,
        f: &Rc<FunctionDecl>,
        args: Vec<Value>,
        span: Span,
    ) -> EvalResult {
        // FR-RT-005: bounded recursion with a readable error.
        self.frames += 1;
        if self.frames > MAX_FRAMES {
            self.frames -= 1;
            return Err(RuntimeError::new(
                codes::STACK_OVERFLOW,
                span,
                format!(
                    "call stack exceeded {} frames while calling `{}`",
                    MAX_FRAMES, f.name
                ),
            )
            .with_help("this is usually unbounded recursion; check the base case"));
        }

        let saved = std::mem::take(&mut self.scopes);
        self.scopes = vec![HashMap::new()];
        for (p, v) in f.params.iter().zip(args) {
            self.define(&p.name, v);
        }
        let flow = self.exec_block_inner(&f.body);
        self.scopes = saved;
        self.frames -= 1;

        match flow? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Value::Void),
        }
    }

    // ---------------- statements ----------------

    pub fn exec_block(&mut self, b: &Block) -> ExecResult {
        self.push_scope();
        let r = self.exec_block_inner(b);
        self.pop_scope();
        r
    }

    fn exec_block_inner(&mut self, b: &Block) -> ExecResult {
        for s in &b.stmts {
            match self.exec_stmt(s)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    pub fn exec_stmt(&mut self, s: &Stmt) -> ExecResult {
        if klions_platform::interrupted() {
            return Err(RuntimeError::interrupted(s.span));
        }
        match &s.kind {
            StmtKind::Error => Ok(Flow::Normal),
            StmtKind::Let { name, value, .. } => {
                let v = self.eval(value)?;
                self.define(name, v);
                Ok(Flow::Normal)
            }
            StmtKind::Assign { target, op, value } => {
                let rhs = self.eval(value)?;
                let new = match op.to_binop() {
                    None => rhs,
                    Some(b) => {
                        let cur = self.eval(target)?;
                        value::binary(b, &cur, &rhs, s.span)?
                    }
                };
                self.store(target, new)?;
                Ok(Flow::Normal)
            }
            StmtKind::Expr(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            StmtKind::Block(b) => self.exec_block(b),
            StmtKind::NoGrad(b) => {
                // FR-AI-004: suspend tracking for the whole block.
                let prev = klions_autodiff::set_tracking(false);
                let r = self.exec_block(b);
                klions_autodiff::set_tracking(prev);
                r
            }
            StmtKind::If { cond, then_block, else_branch } => {
                let c = self.eval(cond)?;
                if c.truthy(cond.span)? {
                    self.exec_block(then_block)
                } else if let Some(e) = else_branch {
                    self.exec_stmt(e)
                } else {
                    Ok(Flow::Normal)
                }
            }
            StmtKind::While { cond, body } => {
                loop {
                    if klions_platform::interrupted() {
                        return Err(RuntimeError::interrupted(s.span));
                    }
                    let c = self.eval(cond)?;
                    if !c.truthy(cond.span)? {
                        break;
                    }
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            StmtKind::For { var, iter, body, .. } => self.exec_for(var, iter, body, s.span),
            StmtKind::Return(v) => {
                let val = match v {
                    Some(e) => self.eval(e)?,
                    None => Value::Void,
                };
                Ok(Flow::Return(val))
            }
            StmtKind::Break => Ok(Flow::Break),
            StmtKind::Continue => Ok(Flow::Continue),
            StmtKind::Train { model, dataset, options, model_span, dataset_span } => {
                train::run_training(
                    self,
                    model,
                    *model_span,
                    dataset,
                    *dataset_span,
                    options,
                    s.span,
                )?;
                Ok(Flow::Normal)
            }
        }
    }

    fn exec_for(&mut self, var: &str, iter: &IterExpr, body: &Block, span: Span) -> ExecResult {
        let items: Vec<Value> = match iter {
            IterExpr::Range(a, b) => {
                let start = self.eval(a)?.as_int(a.span)?;
                let end = self.eval(b)?.as_int(b.span)?;
                if end < start {
                    Vec::new()
                } else {
                    (start..end).map(Value::Int).collect()
                }
            }
            IterExpr::Value(e) => {
                let v = self.eval(e)?;
                match v {
                    Value::Array(items) => (*items).clone(),
                    Value::Tensor(t) => {
                        let n = t.dim(0);
                        let mut out = Vec::with_capacity(n);
                        for i in 0..n {
                            out.push(Value::Tensor(Rc::new(
                                t.row(i).map_err(|e| RuntimeError::from_tensor(e, e2span(span)))?,
                            )));
                        }
                        out
                    }
                    Value::Dataset(d) => {
                        let n = d.len();
                        let mut out = Vec::with_capacity(n);
                        for i in 0..n {
                            let x = d
                                .get_sample(i)
                                .map_err(|er| RuntimeError::from_tensor(er, span))?;
                            out.push(Value::Tensor(Rc::new(x)));
                        }
                        out
                    }
                    other => {
                        return Err(RuntimeError::new(
                            codes::NOT_ITERABLE,
                            e.span,
                            format!("`{}` is not iterable", other.type_name()),
                        ))
                    }
                }
            }
        };

        for item in items {
            if klions_platform::interrupted() {
                return Err(RuntimeError::interrupted(span));
            }
            self.push_scope();
            self.define(var, item);
            let r = self.exec_block_inner(body);
            self.pop_scope();
            match r? {
                Flow::Break => break,
                Flow::Return(v) => return Ok(Flow::Return(v)),
                _ => {}
            }
        }
        Ok(Flow::Normal)
    }

    fn store(&mut self, target: &Expr, v: Value) -> Result<(), RuntimeError> {
        match &target.kind {
            ExprKind::Ident(name) => {
                if self.assign(name, v) {
                    Ok(())
                } else {
                    Err(RuntimeError::new(
                        codes::UNDEFINED_NAME,
                        target.span,
                        format!("cannot find `{}` in this scope", name),
                    ))
                }
            }
            ExprKind::Index { base, indices } => {
                let b = self.eval(base)?;
                let mut idx = Vec::with_capacity(indices.len());
                for i in indices {
                    idx.push(self.eval(i)?.as_int(i.span)? as usize);
                }
                match b {
                    Value::Tensor(t) => {
                        let x = v.as_float(target.span)? as f32;
                        t.set(&idx, x).map_err(|e| RuntimeError::from_tensor(e, target.span))?;
                        Ok(())
                    }
                    other => Err(RuntimeError::new(
                        codes::NOT_INDEXABLE,
                        target.span,
                        format!("cannot index into `{}`", other.type_name()),
                    )),
                }
            }
            _ => Err(RuntimeError::new(
                codes::UNEXPECTED_TOKEN,
                target.span,
                "invalid assignment target",
            )),
        }
    }

    // ---------------- expressions ----------------

    pub fn eval(&mut self, e: &Expr) -> EvalResult {
        match &e.kind {
            ExprKind::Error => Ok(Value::Void),
            ExprKind::Int(v) => Ok(Value::Int(*v)),
            ExprKind::Float(v) => Ok(Value::Float(*v)),
            ExprKind::Str(s) => Ok(Value::Str(s.clone())),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Null => Ok(Value::Void),
            ExprKind::Ident(name) => {
                if let Some(v) = self.get(name) {
                    return Ok(v);
                }
                if let Some(m) = self.models.get(name) {
                    return Ok(Value::Model(Rc::clone(m)));
                }
                Err(RuntimeError::new(
                    codes::UNDEFINED_NAME,
                    e.span,
                    format!("cannot find `{}` in this scope", name),
                ))
            }
            ExprKind::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(self.eval(it)?);
                }
                Ok(Value::Array(Rc::new(out)))
            }
            ExprKind::Unary(op, inner) => {
                let v = self.eval(inner)?;
                value::unary(*op, &v, e.span)
            }
            ExprKind::Binary(op, a, b) => {
                // FR-RT-007: `&&` and `||` short-circuit.
                if op.is_logical() {
                    let left = self.eval(a)?.truthy(a.span)?;
                    return Ok(Value::Bool(match op {
                        BinOp::And => {
                            if !left {
                                false
                            } else {
                                self.eval(b)?.truthy(b.span)?
                            }
                        }
                        _ => {
                            if left {
                                true
                            } else {
                                self.eval(b)?.truthy(b.span)?
                            }
                        }
                    }));
                }
                let av = self.eval(a)?;
                let bv = self.eval(b)?;
                value::binary(*op, &av, &bv, e.span)
            }
            ExprKind::Cast(inner, ty) => {
                let v = self.eval(inner)?;
                value::cast(&v, ty, e.span)
            }
            ExprKind::Index { base, indices } => {
                let b = self.eval(base)?;
                let mut idx = Vec::with_capacity(indices.len());
                for i in indices {
                    idx.push(self.eval(i)?.as_int(i.span)?);
                }
                value::index(&b, &idx, e.span)
            }
            ExprKind::Field { base, name, name_span } => {
                let b = self.eval(base)?;
                builtins::field(self, &b, name, *name_span)
            }
            ExprKind::Path { base, name, span } => Err(RuntimeError::new(
                codes::UNDEFINED_NAME,
                *span,
                format!("`{}::{}` must be called, not referenced", base, name),
            )),
            ExprKind::Call { callee, args } => self.eval_call(callee, args, e.span),
            ExprKind::Try(inner) => {
                let v = self.eval(inner)?;
                match v {
                    Value::Ok(x) => Ok((*x).clone()),
                    // FR-ERR-001: `?` returns early on Err.
                    Value::Err(err) => Err(RuntimeError::propagated(*err, e.span)),
                    other => Ok(other),
                }
            }
            ExprKind::Ok(inner) => {
                let v = match inner {
                    Some(x) => self.eval(x)?,
                    None => Value::Void,
                };
                Ok(Value::Ok(Box::new(v)))
            }
            ExprKind::Err(inner) => {
                let v = self.eval(inner)?;
                Ok(Value::Err(Box::new(v)))
            }
        }
    }

    fn eval_call(&mut self, callee: &Expr, args: &[Arg], span: Span) -> EvalResult {
        // Method call.
        if let ExprKind::Field { base, name, name_span } = &callee.kind {
            let recv = self.eval(base)?;
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(self.eval(&a.value)?);
            }
            return builtins::method(self, &recv, name, argv, *name_span, span);
        }
        // Associated function.
        if let ExprKind::Path { base, name, span: pspan } = &callee.kind {
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(self.eval(&a.value)?);
            }
            return builtins::path_call(self, base, name, argv, *pspan);
        }
        // Plain call.
        if let ExprKind::Ident(name) = &callee.kind {
            if let Some(f) = self.functions.get(name).cloned() {
                let mut argv = Vec::with_capacity(args.len());
                for a in args {
                    argv.push(self.eval(&a.value)?);
                }
                return self.call_function(&f, argv, span);
            }
            // Builtins that need the unevaluated format string keep their args.
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(self.eval(&a.value)?);
            }
            return builtins::call(self, name, argv, span);
        }
        Err(RuntimeError::new(
            codes::NOT_CALLABLE,
            span,
            "this expression is not callable",
        ))
    }

    pub fn model_named(&self, name: &str) -> Option<Rc<std::cell::RefCell<Model>>> {
        self.models.get(name).cloned()
    }

    /// EIR-017 / NFR-S-006 dataset path resolution.
    pub fn resolve_path(&self, raw: &str, span: Span) -> Result<PathBuf, RuntimeError> {
        klions_nn::data::resolve_dataset_path(
            &self.source_dir,
            raw,
            self.project_root.as_deref(),
        )
        .map_err(|e| RuntimeError::from_dataset(e, span))
    }
}

fn e2span(s: Span) -> Span {
    s
}

/// Load, analyze, and run a source file. Used by the CLI and the tests.
pub fn run_source(
    source: &str,
    path: &Path,
    seed: u64,
    out: Box<dyn std::io::Write>,
) -> Result<i32, RuntimeError> {
    let analysis = klions_frontend::analyze(source);
    if analysis.has_errors() {
        return Err(RuntimeError::new(
            codes::INTERNAL_ERROR,
            Span::new(0, 0),
            "refusing to run a program that failed analysis",
        ));
    }
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut interp = Interpreter::new(analysis.program, seed, dir);
    interp.out = out;
    interp.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    /// Run a program on the interpreter stack and capture what it printed.
    fn execute(src: &str) -> (Result<i32, RuntimeError>, String) {
        let owned = src.to_string();
        let buf = SharedBuf::new();
        let sink = buf.clone();
        let r = on_interpreter_stack(move || {
            let analysis = klions_frontend::analyze(&owned);
            let errs: Vec<String> = analysis
                .errors()
                .map(|d| format!("[{}] {}", d.code, d.message))
                .collect();
            assert!(errs.is_empty(), "analysis failed: {:?}", errs);
            let mut interp = Interpreter::new(analysis.program, 42, PathBuf::from("."));
            interp.out = Box::new(sink);
            interp.run()
        })
        .expect("interpreter thread panicked");
        (r, buf.contents())
    }

    fn run(src: &str) -> Result<String, RuntimeError> {
        let (r, out) = execute(src);
        r?;
        Ok(out)
    }

    fn run_err(src: &str) -> RuntimeError {
        execute(src).0.expect_err("expected a runtime error")
    }

    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl SharedBuf {
        fn new() -> SharedBuf {
            SharedBuf(Arc::new(Mutex::new(Vec::new())))
        }
        fn contents(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
        }
    }
    impl std::io::Write for SharedBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn hello_world() {
        assert_eq!(run("fn main() { println(\"Hello, KLIONS!\") }").unwrap(), "Hello, KLIONS!\n");
    }

    #[test]
    fn arithmetic_and_formatting() {
        let out = run("fn main() { let a = 6\n let b = 7\n println(\"{}\", a * b) }").unwrap();
        assert_eq!(out, "42\n");
    }

    #[test]
    fn conditionals_choose_a_branch() {
        let out = run(
            "fn main() {
                 let x = 10
                 if x > 5 { println(\"big\") } else { println(\"small\") }
             }",
        )
        .unwrap();
        assert_eq!(out, "big\n");
    }

    #[test]
    fn for_loop_over_a_range() {
        let out = run("fn main() { for i in 0..4 { print(\"{} \", i) } }").unwrap();
        assert_eq!(out, "0 1 2 3 ");
    }

    #[test]
    fn while_loop_with_break() {
        let out = run(
            "fn main() {
                 let mut i = 0
                 while true {
                     if i >= 3 { break }
                     print(\"{}\", i)
                     i = i + 1
                 }
             }",
        )
        .unwrap();
        assert_eq!(out, "012");
    }

    #[test]
    fn functions_and_recursion() {
        let out = run(
            "fn fact(n: i64) -> i64 {
                 if n <= 1 { return 1 }
                 return n * fact(n - 1)
             }
             fn main() { println(\"{}\", fact(10)) }",
        )
        .unwrap();
        assert_eq!(out, "3628800\n");
    }

    #[test]
    fn short_circuit_avoids_the_second_operand() {
        // If `||` did not short-circuit, the division would trap.
        let out = run(
            "fn main() {
                 let d = 0
                 if d == 0 || 10 / d > 1 { println(\"safe\") }
             }",
        )
        .unwrap();
        assert_eq!(out, "safe\n");
    }

    #[test]
    fn tensors_evaluate_and_print_their_shape() {
        let out = run(
            "fn main() {
                 let a = zeros(2, 3)
                 println(\"{}\", a.shape())
             }",
        )
        .unwrap();
        assert!(out.contains('2') && out.contains('3'), "got {}", out);
    }

    #[test]
    fn matmul_produces_the_right_values() {
        let out = run(
            "fn main() {
                 let a = ones(2, 3)
                 let b = ones(3, 4)
                 let c = a @ b
                 println(\"{}\", c.sum())
             }",
        )
        .unwrap();
        // 2*4 entries, each 3.0 -> 24
        assert!(out.trim().starts_with("24"), "got {}", out);
    }

    #[test]
    fn integer_division_by_zero_is_an_error() {
        let e = run_err("fn main() { let z = 0\n println(\"{}\", 1 / z) }");
        assert_eq!(e.code, codes::DIVISION_BY_ZERO);
    }

    #[test]
    fn float_division_by_zero_yields_infinity() {
        let out = run("fn main() { let z = 0.0\n println(\"{}\", 1.0 / z) }").unwrap();
        assert!(out.contains("inf"), "got {}", out);
    }

    #[test]
    fn tensor_index_out_of_bounds_is_reported() {
        let e = run_err("fn main() { let a = zeros(2, 2)\n println(\"{}\", a[0, 9]) }");
        assert_eq!(e.code, codes::INDEX_ERROR);
        // FR-TEN-014: the message names the offending index and the bound.
        assert!(e.message.contains('9') && e.message.contains('2'), "{}", e.message);
    }

    /// FR-RT-005: unbounded recursion produces a diagnostic, not a crash.
    #[test]
    fn deep_recursion_reports_stack_overflow_rather_than_crashing() {
        let e = run_err(
            "fn f(n: i64) -> i64 { return f(n + 1) }
             fn main() { let _ = f(0) }",
        );
        assert_eq!(e.code, codes::STACK_OVERFLOW);
        assert!(e.help.is_some(), "the error should suggest checking the base case");
    }

    #[test]
    fn main_return_value_becomes_the_exit_code() {
        assert_eq!(execute("fn main() -> i32 { return 3 }").0.unwrap(), 3);
    }

    #[test]
    fn identical_seeds_produce_identical_output() {
        let src = "fn main() { let a = rand(2, 2)\n println(\"{}\", a.sum()) }";
        let one = run(src).unwrap();
        let two = run(src).unwrap();
        assert_eq!(one, two);
    }

    #[test]
    fn models_run_a_forward_pass() {
        let out = run(
            "model Net {
                 h = Dense(inputs=4, outputs=8, activation=\"relu\")
                 o = Dense(inputs=8, outputs=2)
             }
             fn main() {
                 let x = ones(1, 4)
                 let y = Net.predict(x)
                 println(\"{}\", y.shape())
             }",
        )
        .unwrap();
        assert!(out.contains('1') && out.contains('2'), "got {}", out);
    }
}
