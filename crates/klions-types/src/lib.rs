//! KLIONS semantic analysis — FR-TYP-001 … FR-TYP-014.
//!
//! The pass that gives KLIONS its reason to exist: tensor shapes are part of
//! the type, so a rank or dimension error in `+ - * / @` is reported at
//! compile time, with both shapes and the disagreeing axis named
//! (FR-TYP-005, FR-TYP-006, FR-TYP-009).

pub mod builtins;
pub mod ty;

use std::collections::HashMap;

use klions_ast::*;
use klions_diagnostics::{codes, Diagnostic, Span};
use klions_parser::nearest;
pub use ty::{broadcast, matmul_shape, MatMulError, Ty};

#[derive(Clone, Debug)]
struct Binding {
    ty: Ty,
    mutable: bool,
    span: Span,
    used: bool,
    is_param: bool,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub input_width: Option<usize>,
    pub output_width: Option<usize>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FnInfo {
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub span: Span,
}

pub struct Checker<'a> {
    program: &'a Program,
    scopes: Vec<HashMap<String, Binding>>,
    functions: HashMap<String, FnInfo>,
    models: HashMap<String, ModelInfo>,
    consts: HashMap<String, Ty>,
    /// Names of `let`-bound datasets, so `train M using D` can be checked.
    pub diagnostics: Vec<Diagnostic>,
    current_ret: Ty,
    loop_depth: usize,
    in_function: bool,
}

pub struct CheckResult {
    pub diagnostics: Vec<Diagnostic>,
    pub models: HashMap<String, ModelInfo>,
    pub functions: HashMap<String, FnInfo>,
}

pub fn check(program: &Program) -> CheckResult {
    let mut c = Checker::new(program);
    c.run();
    CheckResult {
        diagnostics: c.diagnostics,
        models: c.models,
        functions: c.functions,
    }
}

impl<'a> Checker<'a> {
    fn new(program: &'a Program) -> Self {
        Checker {
            program,
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            models: HashMap::new(),
            consts: HashMap::new(),
            diagnostics: Vec::new(),
            current_ret: Ty::Void,
            loop_depth: 0,
            in_function: false,
        }
    }

    fn err(&mut self, d: Diagnostic) {
        self.diagnostics.push(d);
    }

    // ---------------- scope helpers ----------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            // FR-TYP-015: warn about bindings never read.
            let mut unused: Vec<(String, Binding)> = scope
                .into_iter()
                // `<error>` is the parser's placeholder for a name it could not
                // read. Warning that it is unused would be noise stacked on top
                // of the real diagnostic.
                .filter(|(n, b)| {
                    !b.used && !n.starts_with('_') && n != "_" && n != "<error>"
                })
                .collect();
            unused.sort_by_key(|(_, b)| b.span.start);
            for (name, b) in unused {
                let (code, kind) = if b.is_param {
                    (codes::UNUSED_PARAMETER, "parameter")
                } else {
                    (codes::UNUSED_VARIABLE, "variable")
                };
                self.err(
                    Diagnostic::warning(code, b.span, format!("unused {} `{}`", kind, name))
                        .with_help(format!(
                            "prefix it with an underscore (`_{}`) if that is intentional",
                            name
                        )),
                );
            }
        }
    }

    fn declare(&mut self, name: &str, ty: Ty, mutable: bool, span: Span, is_param: bool) {
        if name == "_" {
            return;
        }
        // FR-TYP-002: shadowing in an inner scope is allowed; redefinition in
        // the same scope is not.
        if let Some(prev) = self.scopes.last().and_then(|s| s.get(name)) {
            let prev_span = prev.span;
            self.err(
                Diagnostic::error(
                    codes::DUPLICATE_DEFINITION,
                    span,
                    format!("`{}` is already defined in this scope", name),
                )
                .with_secondary(prev_span, "first definition here")
                .with_help("rename one of them, or use an inner block to shadow deliberately"),
            );
            return;
        }
        self.scopes.last_mut().unwrap().insert(
            name.to_string(),
            Binding { ty, mutable, span, used: false, is_param },
        );
    }

    fn lookup(&mut self, name: &str) -> Option<Binding> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(b) = scope.get_mut(name) {
                b.used = true;
                return Some(b.clone());
            }
        }
        None
    }

    fn all_visible_names(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for s in &self.scopes {
            v.extend(s.keys().cloned());
        }
        v.extend(self.functions.keys().cloned());
        v.extend(self.models.keys().cloned());
        v.extend(self.consts.keys().cloned());
        v.extend(builtins::BUILTIN_NAMES.iter().map(|s| s.to_string()));
        v
    }

    /// FR-TYP-010: "did you mean" within edit distance 2.
    fn undefined_name(&mut self, name: &str, span: Span) {
        let names = self.all_visible_names();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let mut d = Diagnostic::error(
            codes::UNDEFINED_NAME,
            span,
            format!("cannot find `{}` in this scope", name),
        );
        if let Some(s) = nearest(name, &refs) {
            d = d.with_help(format!("did you mean `{}`?", s));
        }
        self.err(d);
    }

    // ---------------- program ----------------

    fn run(&mut self) {
        // Pass 1: collect signatures so order of declaration does not matter.
        for item in &self.program.items {
            match item {
                Item::Function(f) => {
                    if self.functions.contains_key(&f.name) {
                        let prev = self.functions[&f.name].span;
                        self.err(
                            Diagnostic::error(
                                codes::DUPLICATE_DEFINITION,
                                f.name_span,
                                format!("function `{}` is defined more than once", f.name),
                            )
                            .with_secondary(prev, "first definition here"),
                        );
                        continue;
                    }
                    let params = f
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), ty::from_ast(&p.ty)))
                        .collect();
                    let ret = f.ret.as_ref().map(ty::from_ast).unwrap_or(Ty::Void);
                    self.functions.insert(
                        f.name.clone(),
                        FnInfo { params, ret, span: f.name_span },
                    );
                }
                Item::Model(m) => {
                    if self.models.contains_key(&m.name) {
                        let prev = self.models[&m.name].span;
                        self.err(
                            Diagnostic::error(
                                codes::DUPLICATE_DEFINITION,
                                m.name_span,
                                format!("model `{}` is defined more than once", m.name),
                            )
                            .with_secondary(prev, "first definition here"),
                        );
                        continue;
                    }
                    let info = self.check_model(m);
                    self.models.insert(m.name.clone(), info);
                }
                Item::Const(c) => {
                    self.consts.insert(c.name.clone(), ty::from_ast(&c.ty));
                }
            }
        }

        // FR-RT-001: a program needs `main`.
        if !self.functions.contains_key("main") {
            let span = self
                .program
                .items
                .first()
                .map(|i| i.span())
                .unwrap_or(Span::new(0, 1));
            self.err(
                Diagnostic::error(codes::MISSING_MAIN, span, "no `main` function found")
                    .with_help("every KLIONS program starts at `fn main() { ... }`"),
            );
        } else {
            let main = self.functions["main"].clone();
            if !main.params.is_empty() {
                let sp = main.span;
                self.err(
                    Diagnostic::error(
                        codes::WRONG_ARG_COUNT,
                        sp,
                        "`main` must not take parameters",
                    )
                    .with_help("write `fn main() { ... }` or `fn main() -> i32 { ... }`"),
                );
            }
            let ret = main.ret.clone();
            if !matches!(ret, Ty::Void | Ty::I32 | Ty::Unknown) {
                let sp = main.span;
                self.err(
                    Diagnostic::error(
                        codes::RETURN_TYPE_MISMATCH,
                        sp,
                        format!("`main` returns `{}`, which is not allowed", ret.display()),
                    )
                    .with_help("`main` must return `void` or `i32` (FR-RT-004)"),
                );
            }
        }

        // Pass 2: check constant initializers and function bodies.
        for item in &self.program.items {
            match item {
                Item::Const(c) => {
                    let want = ty::from_ast(&c.ty);
                    let got = self.expr_ty(&c.value);
                    let got = adapt_literal(&c.value, &want, &got);
                    if !want.accepts(&got) {
                        self.type_mismatch(&want, &got, c.value.span, "constant initializer");
                    }
                }
                Item::Function(f) => self.check_function(f),
                Item::Model(_) => {}
            }
        }
    }

    /// FR-TYP-012: verify the layer chain is width-consistent.
    fn check_model(&mut self, m: &ModelDecl) -> ModelInfo {
        let mut prev_out: Option<(usize, Span, String)> = None;
        let mut first_in = None;
        let mut last_out = None;
        let mut seen: HashMap<String, Span> = HashMap::new();

        for binding in &m.layers {
            if let Some(prev) = seen.get(&binding.name) {
                let prev = *prev;
                self.err(
                    Diagnostic::error(
                        codes::DUPLICATE_DEFINITION,
                        binding.name_span,
                        format!("layer `{}` is declared twice in model `{}`", binding.name, m.name),
                    )
                    .with_secondary(prev, "first declaration here"),
                );
            } else {
                seen.insert(binding.name.clone(), binding.name_span);
            }

            let l = &binding.layer;
            let (inputs, outputs) = self.check_layer_args(l);
            if first_in.is_none() {
                first_in = inputs;
            }
            if let Some(o) = outputs {
                last_out = Some(o);
            }

            // The consistency check itself.
            if let (Some((po, pspan, pname)), Some(i)) = (&prev_out, inputs) {
                if *po != i {
                    self.err(
                        Diagnostic::error(
                            codes::LAYER_CHAIN_MISMATCH,
                            l.span,
                            format!(
                                "layer `{}` expects {} inputs but `{}` produces {} outputs",
                                binding.name, i, pname, po
                            ),
                        )
                        .with_secondary(*pspan, format!("`{}` produces {} outputs", pname, po))
                        .with_note(format!(
                            "layers run in declaration order, so `{}` receives `{}`'s output",
                            binding.name, pname
                        ))
                        .with_help(format!("set `inputs={}` on `{}`", po, binding.name)),
                    );
                }
            }
            if let Some(o) = outputs {
                prev_out = Some((o, l.span, binding.name.clone()));
            }
        }

        ModelInfo {
            name: m.name.clone(),
            input_width: first_in,
            output_width: last_out,
            span: m.name_span,
        }
    }

    /// Validate one layer's arguments; return (inputs, outputs) where known.
    fn check_layer_args(&mut self, l: &LayerExpr) -> (Option<usize>, Option<usize>) {
        let spec = match builtins::layer_spec(&l.kind) {
            Some(s) => s,
            None => return (None, None), // parser already reported it
        };
        let mut values: HashMap<&str, (i64, Span)> = HashMap::new();
        let mut strings: HashMap<&str, (String, Span)> = HashMap::new();
        let mut floats: HashMap<&str, (f64, Span)> = HashMap::new();

        for (i, a) in l.args.iter().enumerate() {
            let key: Option<&str> = match &a.name {
                Some(n) => match spec.params.iter().find(|(pn, _)| pn == n) {
                    Some((pn, _)) => Some(pn),
                    None => {
                        let cands: Vec<&str> = spec.params.iter().map(|(n, _)| *n).collect();
                        let mut d = Diagnostic::error(
                            codes::UNKNOWN_NAMED_ARG,
                            a.name_span,
                            format!("`{}` has no argument named `{}`", l.kind, n),
                        );
                        d = match nearest(n, &cands) {
                            Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                            None if cands.is_empty() => {
                                d.with_help(format!("`{}` takes no arguments", l.kind))
                            }
                            None => d.with_help(format!("valid arguments: {}", cands.join(", "))),
                        };
                        self.err(d);
                        None
                    }
                },
                None => spec.params.get(i).map(|(n, _)| *n),
            };
            let Some(key) = key else { continue };

            match &a.value.kind {
                ExprKind::Int(v) => {
                    values.insert(key, (*v, a.value.span));
                    if *v <= 0 && matches!(key, "inputs" | "outputs") {
                        self.err(
                            Diagnostic::error(
                                codes::NEGATIVE_DIMENSION,
                                a.value.span,
                                format!("`{}` must be a positive integer, found {}", key, v),
                            )
                            .with_help("layer widths count neurons, so they start at 1"),
                        );
                    }
                }
                ExprKind::Float(v) => {
                    floats.insert(key, (*v, a.value.span));
                }
                ExprKind::Str(s) => {
                    strings.insert(key, (s.clone(), a.value.span));
                }
                _ => {
                    // FR-AI-007: layer arguments must be compile-time constants.
                    self.err(
                        Diagnostic::error(
                            codes::TYPE_MISMATCH,
                            a.value.span,
                            "layer arguments must be literal constants",
                        )
                        .with_help(
                            "model declarations are resolved at compile time so shapes can be \
                             checked; use a `const` if you need a named value",
                        ),
                    );
                }
            }
        }

        // Required arguments.
        for (pname, required) in spec.params {
            if *required
                && !values.contains_key(pname)
                && !strings.contains_key(pname)
                && !floats.contains_key(pname)
            {
                self.err(
                    Diagnostic::error(
                        codes::WRONG_ARG_COUNT,
                        l.span,
                        format!("`{}` requires the `{}` argument", l.kind, pname),
                    )
                    .with_help(format!("add `{}=...` to the layer", pname)),
                );
            }
        }

        // Activation must name a function that exists.
        if let Some((act, sp)) = strings.get("activation") {
            if builtins::ACTIVATIONS.iter().all(|a| a != act) {
                let mut d = Diagnostic::error(
                    codes::UNDEFINED_NAME,
                    *sp,
                    format!("unknown activation `{}`", act),
                );
                d = match nearest(act, builtins::ACTIVATIONS) {
                    Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                    None => d.with_help(format!(
                        "available activations: {}",
                        builtins::ACTIVATIONS.join(", ")
                    )),
                };
                self.err(d);
            }
        }

        // Dropout probability range.
        if l.kind == "Dropout" {
            let p = floats
                .get("p")
                .map(|(v, s)| (*v, *s))
                .or_else(|| values.get("p").map(|(v, s)| (*v as f64, *s)));
            if let Some((p, sp)) = p {
                if !(0.0..1.0).contains(&p) {
                    self.err(
                        Diagnostic::error(
                            codes::BAD_HYPERPARAMETER,
                            sp,
                            format!("dropout probability must be in [0, 1), found {}", p),
                        )
                        .with_help("`p` is the fraction of units dropped; 1.0 would drop all of them"),
                    );
                }
            }
        }

        (
            values.get("inputs").map(|(v, _)| *v as usize),
            values.get("outputs").map(|(v, _)| *v as usize),
        )
    }

    fn check_function(&mut self, f: &FunctionDecl) {
        self.push_scope();
        self.in_function = true;
        self.current_ret = f.ret.as_ref().map(ty::from_ast).unwrap_or(Ty::Void);

        for p in &f.params {
            let t = ty::from_ast(&p.ty);
            self.declare(&p.name, t, false, p.span, true);
        }
        self.check_block_inner(&f.body);

        // FR-TYP-011: every path must return a value.
        let ret = self.current_ret.clone();
        if !matches!(ret, Ty::Void | Ty::Unknown) && !block_always_returns(&f.body) {
            self.err(
                Diagnostic::error(
                    codes::MISSING_RETURN,
                    f.body.span,
                    format!(
                        "function `{}` declares `-> {}` but not every path returns a value",
                        f.name,
                        ret.display()
                    ),
                )
                .with_help("add a `return` at the end, or an `else` branch that returns"),
            );
        }
        self.pop_scope();
        self.in_function = false;
    }

    fn check_block(&mut self, b: &Block) {
        self.push_scope();
        self.check_block_inner(b);
        self.pop_scope();
    }

    fn check_block_inner(&mut self, b: &Block) {
        for s in &b.stmts {
            self.check_stmt(s);
        }
    }

    fn check_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Error => {}
            StmtKind::Let { mutable, name, name_span, ty, value } => {
                let got = self.expr_ty(value);
                let final_ty = match ty {
                    Some(annot) => {
                        let want = ty::from_ast(annot);
                        // A numeric *literal* has no fixed type until context
                        // gives it one, so `let x: i32 = 1` is exact, not a
                        // conversion. FR-TYP-003 governs values, not literals.
                        let got = adapt_literal(value, &want, &got);
                        if !want.accepts(&got) {
                            // FR-TYP-005: shape mismatches get the richer message.
                            if want.is_tensor() && got.is_tensor() {
                                self.shape_annotation_mismatch(&want, &got, value.span, annot.span);
                            } else {
                                self.type_mismatch(&want, &got, value.span, "`let` binding");
                            }
                        }
                        // Keep the more specific of the two.
                        if want.is_unknown() { got } else { want }
                    }
                    None => got,
                };
                self.declare(name, final_ty, *mutable, *name_span, false);
            }
            StmtKind::Assign { target, op, value } => {
                let tt = self.expr_ty(target);
                let vt = self.expr_ty(value);
                // FR-PAR-005: assignment requires `mut`.
                if let Some(root) = target.root_ident() {
                    let binding = self
                        .scopes
                        .iter()
                        .rev()
                        .find_map(|sc| sc.get(root))
                        .cloned();
                    if let Some(b) = binding {
                        if !b.mutable {
                            self.err(
                                Diagnostic::error(
                                    codes::ASSIGN_TO_IMMUTABLE,
                                    target.span,
                                    format!("cannot assign to immutable variable `{}`", root),
                                )
                                .with_secondary(b.span, "declared without `mut` here")
                                .with_help(format!("declare it as `let mut {} = ...`", root)),
                            );
                        }
                    }
                }
                match op.to_binop() {
                    None => {
                        let vt = adapt_literal(value, &tt, &vt);
                        if !tt.accepts(&vt) {
                            self.type_mismatch(&tt, &vt, value.span, "assignment");
                        }
                    }
                    Some(b) => {
                        let _ = self.binary_ty(b, &tt, &vt, target.span, value.span, s.span);
                    }
                }
            }
            StmtKind::Train { model, model_span, dataset, dataset_span, options } => {
                self.check_train(model, *model_span, dataset, *dataset_span, options);
            }
            StmtKind::If { cond, then_block, else_branch } => {
                let ct = self.expr_ty(cond);
                self.require_bool(&ct, cond.span, "`if` condition");
                self.check_block(then_block);
                if let Some(e) = else_branch {
                    self.check_stmt(e);
                }
            }
            StmtKind::While { cond, body } => {
                let ct = self.expr_ty(cond);
                self.require_bool(&ct, cond.span, "`while` condition");
                self.loop_depth += 1;
                self.check_block(body);
                self.loop_depth -= 1;
            }
            StmtKind::For { var, var_span, iter, body } => {
                let elem = match iter {
                    IterExpr::Range(a, b) => {
                        let at = self.expr_ty(a);
                        let bt = self.expr_ty(b);
                        for (t, sp) in [(&at, a.span), (&bt, b.span)] {
                            if !t.is_int() && !t.is_unknown() {
                                self.err(
                                    Diagnostic::error(
                                        codes::TYPE_MISMATCH,
                                        sp,
                                        format!(
                                            "range bounds must be integers, found `{}`",
                                            t.display()
                                        ),
                                    )
                                    .with_help("use `0..n` with integer endpoints"),
                                );
                            }
                        }
                        Ty::I64
                    }
                    IterExpr::Value(e) => {
                        let t = self.expr_ty(e);
                        match &t {
                            Ty::Array(inner) => (**inner).clone(),
                            Ty::Tensor { elem, dims } if !dims.is_empty() => Ty::Tensor {
                                elem: elem.clone(),
                                dims: dims[1..].to_vec(),
                            },
                            Ty::Dataset => Ty::tensor_wild(2),
                            Ty::Unknown => Ty::Unknown,
                            other => {
                                self.err(
                                    Diagnostic::error(
                                        codes::NOT_ITERABLE,
                                        e.span,
                                        format!("`{}` is not iterable", other.display()),
                                    )
                                    .with_help(
                                        "iterate over a range (`0..n`), an array, a tensor, or a dataset",
                                    ),
                                );
                                Ty::Unknown
                            }
                        }
                    }
                };
                self.push_scope();
                self.declare(var, elem, false, *var_span, false);
                self.loop_depth += 1;
                self.check_block_inner(body);
                self.loop_depth -= 1;
                self.pop_scope();
            }
            StmtKind::Return(v) => {
                let want = self.current_ret.clone();
                let got = match v {
                    Some(e) => {
                        let t = self.expr_ty(e);
                        adapt_literal(e, &want, &t)
                    }
                    None => Ty::Void,
                };
                if !want.accepts(&got) {
                    let sp = v.as_ref().map(|e| e.span).unwrap_or(s.span);
                    self.err(
                        Diagnostic::error(
                            codes::RETURN_TYPE_MISMATCH,
                            sp,
                            format!(
                                "this function returns `{}` but this expression is `{}`",
                                want.display(),
                                got.display()
                            ),
                        )
                        .with_help(if matches!(want, Ty::Void) {
                            "the function has no return type; write `return` with no value".to_string()
                        } else {
                            format!("produce a `{}` here, or change the signature", want.display())
                        }),
                    );
                }
            }
            StmtKind::Break => {
                if self.loop_depth == 0 {
                    self.err(
                        Diagnostic::error(
                            codes::BREAK_OUTSIDE_LOOP,
                            s.span,
                            "`break` outside a loop",
                        )
                        .with_help("`break` may only appear inside `while` or `for`"),
                    );
                }
            }
            StmtKind::Continue => {
                if self.loop_depth == 0 {
                    self.err(
                        Diagnostic::error(
                            codes::CONTINUE_OUTSIDE_LOOP,
                            s.span,
                            "`continue` outside a loop",
                        )
                        .with_help("`continue` may only appear inside `while` or `for`"),
                    );
                }
            }
            StmtKind::NoGrad(b) => self.check_block(b),
            StmtKind::Block(b) => self.check_block(b),
            StmtKind::Expr(e) => {
                self.expr_ty(e);
            }
        }
    }

    /// FR-TYP-013: `train M using D` — the model's input width must match the
    /// dataset's feature width where both are statically known.
    fn check_train(
        &mut self,
        model: &str,
        model_span: Span,
        dataset: &str,
        dataset_span: Span,
        options: &[TrainOption],
    ) {
        let minfo = self.models.get(model).cloned();
        if minfo.is_none() {
            // Might be a variable holding a Model.
            match self.lookup(model) {
                Some(b) if matches!(b.ty, Ty::Model | Ty::Unknown) => {}
                Some(b) => {
                    self.err(
                        Diagnostic::error(
                            codes::NOT_A_MODEL,
                            model_span,
                            format!("`{}` is a `{}`, not a model", model, b.ty.display()),
                        )
                        .with_help("`train` takes a `model` declaration or a `Model` value"),
                    );
                }
                None => self.undefined_name(model, model_span),
            }
        }

        let dty = match self.lookup(dataset) {
            Some(b) => b.ty,
            None => {
                self.undefined_name(dataset, dataset_span);
                Ty::Unknown
            }
        };
        if !matches!(dty, Ty::Dataset | Ty::Unknown) {
            self.err(
                Diagnostic::error(
                    codes::NOT_A_DATASET,
                    dataset_span,
                    format!("`{}` is a `{}`, not a dataset", dataset, dty.display()),
                )
                .with_help("load one with `Dataset::from_idx(...)` or `Dataset::from_csv(...)`"),
            );
        }

        // Option types.
        let mut seen: HashMap<&str, Span> = HashMap::new();
        for o in options {
            if let Some(spec) = builtins::train_option_spec(&o.key) {
                if let Some(prev) = seen.get(spec.name) {
                    let prev = *prev;
                    self.err(
                        Diagnostic::error(
                            codes::DUPLICATE_DEFINITION,
                            o.key_span,
                            format!("training option `{}` is given twice", o.key),
                        )
                        .with_secondary(prev, "first given here"),
                    );
                }
                seen.insert(spec.name, o.key_span);
                let got = self.expr_ty(&o.value);
                let got = adapt_literal(&o.value, &spec.ty, &got);
                if !spec.ty.accepts(&got) {
                    self.err(
                        Diagnostic::error(
                            codes::TYPE_MISMATCH,
                            o.value.span,
                            format!(
                                "`{}` expects `{}`, found `{}`",
                                o.key,
                                spec.ty.display(),
                                got.display()
                            ),
                        )
                        .with_help(spec.help.to_string()),
                    );
                }
                self.check_train_option_value(spec.name, &o.value);
            } else {
                self.expr_ty(&o.value);
            }
        }

        // FR-AI-008: `epochs` has no sensible default.
        if !seen.contains_key("epochs") {
            self.err(
                Diagnostic::error(
                    codes::MISSING_TRAIN_OPTION,
                    model_span,
                    "`train` requires an `epochs` option",
                )
                .with_help("add `epochs = 10` inside the `train` block"),
            );
        }

        // The flagship compile-time check.
        if let (Some(mi), Ty::Dataset) = (&minfo, &dty) {
            if let (Some(w), Some(dw)) = (mi.input_width, self.dataset_width(dataset)) {
                if w != dw {
                    self.err(
                        Diagnostic::error(
                            codes::MODEL_INPUT_MISMATCH,
                            dataset_span,
                            format!(
                                "model `{}` expects {} input features but dataset `{}` provides {}",
                                model, w, dataset, dw
                            ),
                        )
                        .with_secondary(mi.span, format!("`{}` declared here", model))
                        .with_help(format!(
                            "set the first layer's `inputs={}`, or reshape the dataset",
                            dw
                        )),
                    );
                }
            }
        }
    }

    fn check_train_option_value(&mut self, key: &str, e: &Expr) {
        match (key, &e.kind) {
            ("epochs", ExprKind::Int(v)) if *v <= 0 => {
                self.err(
                    Diagnostic::error(
                        codes::BAD_HYPERPARAMETER,
                        e.span,
                        format!("`epochs` must be at least 1, found {}", v),
                    )
                    .with_help("training for zero epochs would leave the model untrained"),
                );
            }
            ("batch_size", ExprKind::Int(v)) if *v <= 0 => {
                self.err(Diagnostic::error(
                    codes::BAD_HYPERPARAMETER,
                    e.span,
                    format!("`batch_size` must be at least 1, found {}", v),
                ));
            }
            ("validation_split", ExprKind::Float(v)) if !(0.0..1.0).contains(v) => {
                self.err(
                    Diagnostic::error(
                        codes::BAD_HYPERPARAMETER,
                        e.span,
                        format!("`validation_split` must be in [0, 1), found {}", v),
                    )
                    .with_help("a split of 1.0 would leave no training data"),
                );
            }
            ("loss", ExprKind::Str(s)) => {
                if !builtins::LOSSES.contains(&s.as_str()) {
                    let mut d = Diagnostic::error(
                        codes::UNKNOWN_LOSS,
                        e.span,
                        format!("unknown loss function `{}`", s),
                    );
                    d = match nearest(s, builtins::LOSSES) {
                        Some(x) => d.with_help(format!("did you mean `{}`?", x)),
                        None => d
                            .with_help(format!("available losses: {}", builtins::LOSSES.join(", "))),
                    };
                    self.err(d);
                }
            }
            _ => {}
        }
    }

    /// Where a dataset's width is statically known from its type annotation.
    fn dataset_width(&self, _name: &str) -> Option<usize> {
        // v0.1.0 learns dataset widths at run time (files are read then), so
        // this returns None unless a future release adds a manifest.
        None
    }

    // ---------------- expressions ----------------

    fn expr_ty(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Error => Ty::Unknown,
            ExprKind::Int(_) => Ty::I64,
            ExprKind::Float(_) => Ty::F64,
            ExprKind::Str(_) => Ty::Str,
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Null => Ty::Unknown,
            ExprKind::Ident(name) => {
                if let Some(b) = self.lookup(name) {
                    return b.ty;
                }
                if let Some(t) = self.consts.get(name) {
                    return t.clone();
                }
                if self.models.contains_key(name) {
                    return Ty::Model;
                }
                if let Some(f) = self.functions.get(name) {
                    return Ty::Function {
                        params: f.params.iter().map(|(_, t)| t.clone()).collect(),
                        ret: Box::new(f.ret.clone()),
                    };
                }
                if let Some(t) = builtins::builtin_ty(name) {
                    return t;
                }
                self.undefined_name(name, e.span);
                Ty::Unknown
            }
            ExprKind::Array(items) => {
                if items.is_empty() {
                    return Ty::Array(Box::new(Ty::Unknown));
                }
                let first = self.expr_ty(&items[0]);
                for it in &items[1..] {
                    let t = self.expr_ty(it);
                    if !first.accepts(&t) {
                        self.err(
                            Diagnostic::error(
                                codes::TYPE_MISMATCH,
                                it.span,
                                format!(
                                    "array elements must share a type: expected `{}`, found `{}`",
                                    first.display(),
                                    t.display()
                                ),
                            )
                            .with_secondary(items[0].span, "first element sets the type"),
                        );
                    }
                }
                Ty::Array(Box::new(first))
            }
            ExprKind::Unary(op, inner) => {
                let t = self.expr_ty(inner);
                match op {
                    UnOp::Neg => {
                        if !t.is_numeric() && !t.is_tensor() && !t.is_unknown() {
                            self.err(
                                Diagnostic::error(
                                    codes::BAD_OPERAND_TYPE,
                                    e.span,
                                    format!("cannot negate a value of type `{}`", t.display()),
                                )
                                .with_help("`-` applies to numbers and tensors"),
                            );
                            return Ty::Unknown;
                        }
                        t
                    }
                    UnOp::Not => {
                        self.require_bool(&t, inner.span, "`!`");
                        Ty::Bool
                    }
                }
            }
            ExprKind::Binary(op, a, b) => {
                let at = self.expr_ty(a);
                let bt = self.expr_ty(b);
                self.binary_ty(*op, &at, &bt, a.span, b.span, e.span)
            }
            ExprKind::Cast(inner, target) => {
                let from = self.expr_ty(inner);
                let to = ty::from_ast(target);
                // FR-TYP-004: explicit casts are numeric only.
                let ok = (from.is_numeric() && to.is_numeric())
                    || from.is_unknown()
                    || to.is_unknown()
                    || (from.is_tensor() && to.is_tensor());
                if !ok {
                    self.err(
                        Diagnostic::error(
                            codes::BAD_CAST,
                            e.span,
                            format!(
                                "cannot cast `{}` to `{}`",
                                from.display(),
                                to.display()
                            ),
                        )
                        .with_help("`as` converts between numeric types; it is not a reinterpretation"),
                    );
                }
                to
            }
            ExprKind::Call { callee, args } => self.call_ty(callee, args, e.span),
            ExprKind::Field { base, name, name_span } => {
                let bt = self.expr_ty(base);
                match builtins::field_ty(&bt, name) {
                    Some(t) => t,
                    None => {
                        if bt.is_unknown() {
                            return Ty::Unknown;
                        }
                        let cands = builtins::members_of(&bt);
                        let mut d = Diagnostic::error(
                            codes::NO_SUCH_FIELD,
                            *name_span,
                            format!("`{}` has no member named `{}`", bt.display(), name),
                        );
                        d = match nearest(name, &cands) {
                            Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                            None if cands.is_empty() => d,
                            None => d.with_help(format!("available members: {}", cands.join(", "))),
                        };
                        self.err(d);
                        Ty::Unknown
                    }
                }
            }
            ExprKind::Path { base, name, span } => match builtins::path_ty(base, name) {
                Some(t) => t,
                None => {
                    let cands = builtins::path_members(base);
                    let mut d = Diagnostic::error(
                        codes::UNDEFINED_NAME,
                        *span,
                        format!("`{}::{}` does not exist", base, name),
                    );
                    d = match nearest(name, &cands) {
                        Some(s) => d.with_help(format!("did you mean `{}::{}`?", base, s)),
                        None if cands.is_empty() => {
                            d.with_help(format!("`{}` is not a known type or module", base))
                        }
                        None => d.with_help(format!("`{}` provides: {}", base, cands.join(", "))),
                    };
                    self.err(d);
                    Ty::Unknown
                }
            },
            ExprKind::Index { base, indices } => {
                let bt = self.expr_ty(base);
                for i in indices {
                    let it = self.expr_ty(i);
                    if !it.is_int() && !it.is_unknown() {
                        self.err(
                            Diagnostic::error(
                                codes::TYPE_MISMATCH,
                                i.span,
                                format!("index must be an integer, found `{}`", it.display()),
                            )
                            .with_help("cast with `as i64` if you have a float"),
                        );
                    }
                }
                match &bt {
                    Ty::Array(inner) => (**inner).clone(),
                    Ty::Tensor { elem, dims } => {
                        let n = indices.len();
                        if !dims.is_empty() && n > dims.len() {
                            self.err(
                                Diagnostic::error(
                                    codes::BAD_RANK,
                                    e.span,
                                    format!(
                                        "{} indices given for a rank-{} tensor",
                                        n,
                                        dims.len()
                                    ),
                                )
                                .with_help("supply at most one index per axis"),
                            );
                            return Ty::Unknown;
                        }
                        if dims.is_empty() {
                            Ty::Unknown
                        } else if n == dims.len() {
                            (**elem).clone()
                        } else {
                            Ty::Tensor { elem: elem.clone(), dims: dims[n..].to_vec() }
                        }
                    }
                    Ty::Dataset => Ty::tensor_wild(1),
                    Ty::Unknown => Ty::Unknown,
                    other => {
                        self.err(
                            Diagnostic::error(
                                codes::NOT_INDEXABLE,
                                base.span,
                                format!("`{}` cannot be indexed", other.display()),
                            )
                            .with_help("indexing applies to tensors, arrays, and datasets"),
                        );
                        Ty::Unknown
                    }
                }
            }
            ExprKind::Try(inner) => {
                let t = self.expr_ty(inner);
                // FR-TYP-014: `?` needs a Result, inside a Result-returning fn.
                match &t {
                    Ty::Result(ok, err) => {
                        match &self.current_ret {
                            Ty::Result(_, fn_err) => {
                                if !fn_err.accepts(err) {
                                    self.err(
                                        Diagnostic::error(
                                            codes::QUESTION_ERROR_MISMATCH,
                                            e.span,
                                            format!(
                                                "`?` propagates `{}` but this function returns `{}`",
                                                err.display(),
                                                fn_err.display()
                                            ),
                                        )
                                        .with_help(
                                            "widen the function's error type, or handle the error here",
                                        ),
                                    );
                                }
                            }
                            other => {
                                self.err(
                                    Diagnostic::error(
                                        codes::QUESTION_OUTSIDE_RESULT,
                                        e.span,
                                        format!(
                                            "`?` used in a function returning `{}`",
                                            other.display()
                                        ),
                                    )
                                    .with_help(format!(
                                        "change the signature to `-> Result<{}, {}>`, \
                                         or handle the error explicitly",
                                        other.display(),
                                        err.display()
                                    )),
                                );
                            }
                        }
                        (**ok).clone()
                    }
                    Ty::Unknown => Ty::Unknown,
                    other => {
                        self.err(
                            Diagnostic::error(
                                codes::QUESTION_OUTSIDE_RESULT,
                                e.span,
                                format!("`?` applied to `{}`, which is not a Result", other.display()),
                            )
                            .with_help("`?` unwraps `Result<T, E>`; this value cannot fail"),
                        );
                        Ty::Unknown
                    }
                }
            }
            ExprKind::Ok(inner) => {
                let t = match inner {
                    Some(x) => self.expr_ty(x),
                    None => Ty::Void,
                };
                Ty::Result(Box::new(t), Box::new(Ty::Unknown))
            }
            ExprKind::Err(inner) => {
                let t = self.expr_ty(inner);
                Ty::Result(Box::new(Ty::Unknown), Box::new(t))
            }
        }
    }

    /// FR-TYP-005/006/009 — the shape-aware binary operator rules.
    fn binary_ty(
        &mut self,
        op: BinOp,
        at: &Ty,
        bt: &Ty,
        a_span: Span,
        b_span: Span,
        span: Span,
    ) -> Ty {
        if at.is_unknown() || bt.is_unknown() {
            return if op.is_comparison() || op.is_logical() { Ty::Bool } else { Ty::Unknown };
        }

        if op.is_logical() {
            self.require_bool(at, a_span, "logical operand");
            self.require_bool(bt, b_span, "logical operand");
            return Ty::Bool;
        }

        if op == BinOp::MatMul {
            return self.matmul_ty(at, bt, a_span, b_span, span);
        }

        // Tensor-involved arithmetic and comparison.
        if at.is_tensor() || bt.is_tensor() {
            if op.is_comparison() {
                self.err(
                    Diagnostic::error(
                        codes::BAD_OPERAND_TYPE,
                        span,
                        format!("`{}` is not defined on tensors", op.as_str()),
                    )
                    .with_help(
                        "compare reduced values instead, e.g. `a.sum() > b.sum()`, \
                         or use element-wise helpers",
                    ),
                );
                return Ty::Bool;
            }
            // A scalar on one side broadcasts freely.
            let (adims, bdims) = match (at.dims(), bt.dims()) {
                (Some(a), Some(b)) => (a.to_vec(), b.to_vec()),
                (Some(a), None) if bt.is_numeric() => return Ty::tensor(a.to_vec()),
                (None, Some(b)) if at.is_numeric() => return Ty::tensor(b.to_vec()),
                _ => {
                    let (bad, sp) = if at.is_tensor() { (bt, b_span) } else { (at, a_span) };
                    self.err(
                        Diagnostic::error(
                            codes::BAD_OPERAND_TYPE,
                            sp,
                            format!(
                                "cannot apply `{}` between a tensor and `{}`",
                                op.as_str(),
                                bad.display()
                            ),
                        )
                        .with_help("tensor arithmetic works with another tensor or a scalar number"),
                    );
                    return Ty::Unknown;
                }
            };
            if adims.is_empty() || bdims.is_empty() {
                return Ty::tensor(if adims.is_empty() { bdims } else { adims });
            }
            return match broadcast(&adims, &bdims) {
                Ok(dims) => Ty::tensor(dims),
                Err(axis) => {
                    self.shape_mismatch(op, at, bt, axis, span);
                    Ty::Unknown
                }
            };
        }

        // Scalars.
        if op.is_comparison() {
            if !self.scalars_comparable(at, bt) {
                self.err(
                    Diagnostic::error(
                        codes::TYPE_MISMATCH,
                        span,
                        format!(
                            "cannot compare `{}` with `{}`",
                            at.display(),
                            bt.display()
                        ),
                    )
                    .with_help("comparisons need operands of the same type"),
                );
            }
            return Ty::Bool;
        }

        if !at.is_numeric() || !bt.is_numeric() {
            // `string + string` is a common expectation; say so plainly.
            if matches!(at, Ty::Str) && matches!(bt, Ty::Str) && op == BinOp::Add {
                self.err(
                    Diagnostic::error(
                        codes::BAD_OPERAND_TYPE,
                        span,
                        "strings cannot be joined with `+` in v0.1.0",
                    )
                    .with_help("use `format(\"{}{}\", a, b)` instead"),
                );
                return Ty::Str;
            }
            let (bad, sp) = if at.is_numeric() { (bt, b_span) } else { (at, a_span) };
            self.err(
                Diagnostic::error(
                    codes::BAD_OPERAND_TYPE,
                    sp,
                    format!("`{}` is not defined on `{}`", op.as_str(), bad.display()),
                )
                .with_help("arithmetic applies to numbers and tensors"),
            );
            return Ty::Unknown;
        }

        // FR-TYP-003: no implicit numeric conversion.
        if at != bt {
            self.err(
                Diagnostic::error(
                    codes::NO_IMPLICIT_CONVERSION,
                    span,
                    format!(
                        "`{}` between `{}` and `{}` requires an explicit conversion",
                        op.as_str(),
                        at.display(),
                        bt.display()
                    ),
                )
                .with_note("KLIONS does not convert between numeric types silently")
                .with_help(format!(
                    "write `... as {}` on one side to make the conversion visible",
                    at.display()
                )),
            );
            return at.clone();
        }
        at.clone()
    }

    fn matmul_ty(&mut self, at: &Ty, bt: &Ty, a_span: Span, b_span: Span, span: Span) -> Ty {
        let (adims, bdims) = match (at.dims(), bt.dims()) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                let (bad, sp) = if at.is_tensor() { (bt, b_span) } else { (at, a_span) };
                self.err(
                    Diagnostic::error(
                        codes::BAD_OPERAND_TYPE,
                        sp,
                        format!("`@` needs two tensors, found `{}`", bad.display()),
                    )
                    .with_help("`@` is matrix multiplication; use `*` for element-wise products"),
                );
                return Ty::Unknown;
            }
        };
        if adims.is_empty() || bdims.is_empty() {
            return Ty::Tensor { elem: Box::new(Ty::F32), dims: vec![] };
        }
        match matmul_shape(adims, bdims) {
            Ok(dims) => Ty::tensor(dims),
            Err(MatMulError::RankTooLow { side, rank }) => {
                let sp = if side == "left" { a_span } else { b_span };
                self.err(
                    Diagnostic::error(
                        codes::BAD_RANK,
                        sp,
                        format!(
                            "the {} operand of `@` has rank {}, but matrix multiplication needs at least rank 2",
                            side, rank
                        ),
                    )
                    .with_help("reshape it, e.g. `x.reshape(1, -1)` to make a row vector"),
                );
                Ty::Unknown
            }
            Err(MatMulError::Inner { k_lhs, k_rhs }) => {
                self.err(
                    Diagnostic::error(
                        codes::MATMUL_SHAPE,
                        span,
                        format!(
                            "cannot matrix-multiply {} by {}",
                            at.shape_str(),
                            bt.shape_str()
                        ),
                    )
                    .with_secondary(a_span, format!("this is {}", at.shape_str()))
                    .with_note(format!(
                        "the inner dimensions must agree: {} on the left, {} on the right",
                        k_lhs, k_rhs
                    ))
                    .with_help(
                        "`[m, k] @ [k, n]` gives `[m, n]`; transpose one operand with `.t()` if that is what you meant",
                    ),
                );
                Ty::Unknown
            }
            Err(MatMulError::Batch { axis }) => {
                self.err(
                    Diagnostic::error(
                        codes::SHAPE_MISMATCH,
                        span,
                        format!(
                            "batch dimensions of `@` disagree: {} and {}",
                            at.shape_str(),
                            bt.shape_str()
                        ),
                    )
                    .with_note(format!("axis {} cannot be broadcast", axis)),
                );
                Ty::Unknown
            }
        }
    }

    fn call_ty(&mut self, callee: &Expr, args: &[Arg], span: Span) -> Ty {
        // Method call: `receiver.method(args)`
        if let ExprKind::Field { base, name, name_span } = &callee.kind {
            let recv = self.expr_ty(base);
            let arg_tys: Vec<(Ty, Span)> =
                args.iter().map(|a| (self.expr_ty(&a.value), a.value.span)).collect();
            return match builtins::method_ty(&recv, name, &arg_tys) {
                builtins::MethodLookup::Found(t) => t,
                builtins::MethodLookup::ArityMismatch { want, got } => {
                    self.err(
                        Diagnostic::error(
                            codes::WRONG_ARG_COUNT,
                            span,
                            format!(
                                "`{}.{}` takes {} argument{}, {} given",
                                recv.display(),
                                name,
                                want,
                                if want == 1 { "" } else { "s" },
                                got
                            ),
                        )
                        .with_help(builtins::method_signature(&recv, name)),
                    );
                    Ty::Unknown
                }
                builtins::MethodLookup::NotFound => {
                    if recv.is_unknown() {
                        return Ty::Unknown;
                    }
                    let cands = builtins::members_of(&recv);
                    let mut d = Diagnostic::error(
                        codes::NO_SUCH_FIELD,
                        *name_span,
                        format!("`{}` has no method named `{}`", recv.display(), name),
                    );
                    d = match nearest(name, &cands) {
                        Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                        None if cands.is_empty() => d,
                        None => d.with_help(format!("available methods: {}", cands.join(", "))),
                    };
                    self.err(d);
                    Ty::Unknown
                }
            };
        }

        // Associated function: `Type::function(args)`
        if let ExprKind::Path { base, name, span: pspan } = &callee.kind {
            let arg_tys: Vec<(Ty, Span)> =
                args.iter().map(|a| (self.expr_ty(&a.value), a.value.span)).collect();
            return match builtins::path_call_ty(base, name, &arg_tys) {
                Some(t) => t,
                None => {
                    let cands = builtins::path_members(base);
                    let mut d = Diagnostic::error(
                        codes::UNDEFINED_NAME,
                        *pspan,
                        format!("`{}::{}` does not exist", base, name),
                    );
                    d = match nearest(name, &cands) {
                        Some(s) => d.with_help(format!("did you mean `{}::{}`?", base, s)),
                        None if cands.is_empty() => d,
                        None => d.with_help(format!("`{}` provides: {}", base, cands.join(", "))),
                    };
                    self.err(d);
                    Ty::Unknown
                }
            };
        }

        // Plain function call.
        if let ExprKind::Ident(name) = &callee.kind {
            let arg_tys: Vec<(Ty, Span)> =
                args.iter().map(|a| (self.expr_ty(&a.value), a.value.span)).collect();

            if let Some(f) = self.functions.get(name).cloned() {
                if args.len() != f.params.len() {
                    self.err(
                        Diagnostic::error(
                            codes::WRONG_ARG_COUNT,
                            span,
                            format!(
                                "`{}` takes {} argument{}, {} given",
                                name,
                                f.params.len(),
                                if f.params.len() == 1 { "" } else { "s" },
                                args.len()
                            ),
                        )
                        .with_secondary(f.span, "declared here")
                        .with_help(format!(
                            "signature: fn {}({}) -> {}",
                            name,
                            f.params
                                .iter()
                                .map(|(n, t)| format!("{}: {}", n, t.display()))
                                .collect::<Vec<_>>()
                                .join(", "),
                            f.ret.display()
                        )),
                    );
                    return f.ret;
                }
                for (i, a) in args.iter().enumerate() {
                    let (raw, sp) = &arg_tys[i];
                    let (pname, want) = &f.params[i];
                    let got = &adapt_literal(&a.value, want, raw);
                    // FR-PAR-011: a named argument must match the parameter name.
                    if let Some(given) = &a.name {
                        if given != pname {
                            let cands: Vec<&str> =
                                f.params.iter().map(|(n, _)| n.as_str()).collect();
                            let mut d = Diagnostic::error(
                                codes::UNKNOWN_NAMED_ARG,
                                a.name_span,
                                format!("`{}` has no parameter named `{}`", name, given),
                            );
                            d = match nearest(given, &cands) {
                                Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                                None => d.with_help(format!(
                                    "parameters, in order: {}",
                                    cands.join(", ")
                                )),
                            };
                            self.err(d);
                        }
                    }
                    if !want.accepts(got) {
                        let mut d = Diagnostic::error(
                            codes::TYPE_MISMATCH,
                            *sp,
                            format!(
                                "argument `{}` expects `{}`, found `{}`",
                                pname,
                                want.display(),
                                got.display()
                            ),
                        );
                        if want.is_tensor() && got.is_tensor() {
                            d = d.with_note(format!(
                                "the parameter's shape is {}, the argument's is {}",
                                want.shape_str(),
                                got.shape_str()
                            ));
                        }
                        self.err(d.with_secondary(f.span, "declared here"));
                    }
                }
                return f.ret;
            }

            if let Some(t) = builtins::builtin_call_ty(name, &arg_tys) {
                self.check_builtin_call(name, args, &arg_tys, span);
                return t;
            }

            if self.models.contains_key(name) {
                self.err(
                    Diagnostic::error(
                        codes::NOT_CALLABLE,
                        span,
                        format!("`{}` is a model, not a function", name),
                    )
                    .with_help(format!(
                        "run it with `{}.predict(x)`, or train it with `train {} using ...`",
                        name, name
                    )),
                );
                return Ty::Unknown;
            }

            self.undefined_name(name, callee.span);
            return Ty::Unknown;
        }

        let t = self.expr_ty(callee);
        if !t.is_unknown() {
            self.err(
                Diagnostic::error(
                    codes::NOT_CALLABLE,
                    span,
                    format!("`{}` is not callable", t.display()),
                )
                .with_help("only functions and methods can be called"),
            );
        }
        Ty::Unknown
    }

    /// FR-RT-008: `print`/`println` placeholder count must match the arguments.
    fn check_builtin_call(
        &mut self,
        name: &str,
        args: &[Arg],
        _tys: &[(Ty, Span)],
        span: Span,
    ) {
        if !matches!(name, "print" | "println" | "format") {
            return;
        }
        let Some(first) = args.first() else { return };
        let ExprKind::Str(fmt) = &first.value.kind else { return };
        let placeholders = count_placeholders(fmt);
        let supplied = args.len() - 1;
        if placeholders != supplied {
            self.err(
                Diagnostic::error(
                    codes::FORMAT_ARG_COUNT,
                    span,
                    format!(
                        "format string has {} placeholder{} but {} argument{} supplied",
                        placeholders,
                        if placeholders == 1 { "" } else { "s" },
                        supplied,
                        if supplied == 1 { " was" } else { "s were" }
                    ),
                )
                .with_secondary(first.value.span, "this format string")
                .with_help("write `{}` once for each value; use `{{` for a literal brace"),
            );
        }
    }

    // ---------------- diagnostics helpers ----------------

    fn require_bool(&mut self, t: &Ty, span: Span, what: &str) {
        if matches!(t, Ty::Bool | Ty::Unknown) {
            return;
        }
        let mut d = Diagnostic::error(
            codes::CONDITION_NOT_BOOL,
            span,
            format!("{} must be a `bool`, found `{}`", what, t.display()),
        );
        d = if t.is_numeric() {
            d.with_help("KLIONS has no truthiness; compare explicitly, e.g. `x != 0`")
        } else {
            d.with_help("produce a boolean with a comparison or a logical operator")
        };
        self.err(d);
    }

    fn scalars_comparable(&self, a: &Ty, b: &Ty) -> bool {
        if a == b {
            return true;
        }
        a.is_unknown() || b.is_unknown()
    }

    fn type_mismatch(&mut self, want: &Ty, got: &Ty, span: Span, ctx: &str) {
        let mut d = Diagnostic::error(
            codes::TYPE_MISMATCH,
            span,
            format!(
                "{} expects `{}`, found `{}`",
                ctx,
                want.display(),
                got.display()
            ),
        );
        if want.is_numeric() && got.is_numeric() {
            d = d.with_help(format!(
                "add an explicit conversion: `... as {}`",
                want.display()
            ));
        }
        self.err(d);
    }

    /// FR-TYP-009: name both shapes and the first disagreeing axis.
    fn shape_mismatch(&mut self, op: BinOp, at: &Ty, bt: &Ty, axis: usize, span: Span) {
        let (ad, bd) = (at.dims().unwrap_or(&[]), bt.dims().unwrap_or(&[]));
        let r = ad.len().max(bd.len());
        let get = |d: &[Dim], i: usize| -> String {
            if i < r - d.len() {
                "1".to_string()
            } else {
                d[i - (r - d.len())].to_string()
            }
        };
        self.err(
            Diagnostic::error(
                codes::SHAPE_MISMATCH,
                span,
                format!(
                    "shape mismatch in `{}`: left is {}, right is {}",
                    op.as_str(),
                    at.shape_str(),
                    bt.shape_str()
                ),
            )
            .with_note(format!(
                "axis {} disagrees: {} on the left, {} on the right",
                axis,
                get(ad, axis),
                get(bd, axis)
            ))
            .with_help(
                "element-wise operations compare shapes from the right; a dimension of 1 is \
                 stretched, anything else must match exactly",
            ),
        );
    }

    fn shape_annotation_mismatch(&mut self, want: &Ty, got: &Ty, span: Span, annot_span: Span) {
        let mut d = Diagnostic::error(
            codes::ANNOTATION_MISMATCH,
            span,
            format!(
                "this expression has shape {}, but the annotation says {}",
                got.shape_str(),
                want.shape_str()
            ),
        )
        .with_secondary(annot_span, "declared shape");
        if want.rank() != got.rank() {
            d = d.with_note(format!(
                "ranks differ: {} vs {}",
                want.rank().unwrap_or(0),
                got.rank().unwrap_or(0)
            ));
        }
        self.err(d.with_help("use `_` in the annotation for a dimension known only at run time"));
    }
}

/// A numeric literal adopts the type demanded by its context. This is not an
/// implicit conversion: the literal simply has no independent type to convert
/// from. `let x: i32 = 1` is exact; `let x: i32 = some_i64` is not.
fn adapt_literal(e: &Expr, want: &Ty, got: &Ty) -> Ty {
    match &e.kind {
        ExprKind::Int(_) if want.is_int() => want.clone(),
        // An integer literal in a float context is still exact.
        ExprKind::Int(_) if want.is_float() => want.clone(),
        ExprKind::Float(_) if want.is_float() => want.clone(),
        ExprKind::Unary(UnOp::Neg, inner) => adapt_literal(inner, want, got),
        ExprKind::Array(items) => match (want, got) {
            (Ty::Array(w), Ty::Array(_)) if items.iter().all(is_numeric_literal) => {
                Ty::Array(w.clone())
            }
            _ => got.clone(),
        },
        _ => got.clone(),
    }
}

fn is_numeric_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) => true,
        ExprKind::Unary(UnOp::Neg, inner) => is_numeric_literal(inner),
        _ => false,
    }
}

/// FR-TYP-011 return-path analysis.
fn block_always_returns(b: &Block) -> bool {
    b.stmts.iter().any(stmt_always_returns)
}

fn stmt_always_returns(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Return(_) => true,
        StmtKind::Block(b) => block_always_returns(b),
        StmtKind::NoGrad(b) => block_always_returns(b),
        StmtKind::If { then_block, else_branch, .. } => match else_branch {
            Some(e) => block_always_returns(then_block) && stmt_always_returns(e),
            None => false,
        },
        // `while true { }` is not special-cased; the check stays conservative.
        _ => false,
    }
}

/// Count `{}` placeholders, honouring `{{` and `}}` escapes.
pub fn count_placeholders(fmt: &str) -> usize {
    let b: Vec<char> = fmt.chars().collect();
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        match b[i] {
            '{' if i + 1 < b.len() && b[i + 1] == '{' => i += 2,
            '}' if i + 1 < b.len() && b[i + 1] == '}' => i += 2,
            '{' => {
                // Scan to the closing brace; `{}` and `{:.3}` both count once.
                let mut j = i + 1;
                while j < b.len() && b[j] != '}' {
                    j += 1;
                }
                n += 1;
                i = j + 1;
            }
            _ => i += 1,
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_src(src: &str) -> Vec<Diagnostic> {
        let lexed = klions_lexer_shim::lex(src);
        let parsed = klions_parser::parse(lexed);
        let mut d = parsed.diagnostics;
        d.extend(check(&parsed.program).diagnostics);
        d
    }

    /// The parser crate re-exports what we need; this keeps the test terse.
    mod klions_lexer_shim {
        pub fn lex(src: &str) -> Vec<klions_lexer::Token> {
            klions_lexer::lex(src).tokens
        }
    }

    fn errors(src: &str) -> Vec<Diagnostic> {
        check_src(src)
            .into_iter()
            .filter(|d| d.severity == klions_diagnostics::Severity::Error)
            .collect()
    }

    fn assert_code(src: &str, code: &str) {
        let e = errors(src);
        assert!(
            e.iter().any(|d| d.code == code),
            "expected {} but got: {:?}",
            code,
            e.iter().map(|d| (d.code, &d.message)).collect::<Vec<_>>()
        );
    }

    fn assert_clean(src: &str) {
        let e = errors(src);
        assert!(
            e.is_empty(),
            "expected no errors, got: {:?}",
            e.iter().map(|d| (d.code, &d.message)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn minimal_program_checks() {
        assert_clean("fn main() { let x = 1\n let _ = x + 1 }");
    }

    #[test]
    fn missing_main_is_reported() {
        assert_code("fn helper() { }", codes::MISSING_MAIN);
    }

    #[test]
    fn undefined_name_is_reported() {
        assert_code("fn main() { let _ = undefined_thing }", codes::UNDEFINED_NAME);
    }

    #[test]
    fn no_implicit_numeric_conversion() {
        assert_code(
            "fn main() { let a: i32 = 1\n let b: f32 = 2.0\n let _ = a + b }",
            codes::NO_IMPLICIT_CONVERSION,
        );
    }

    /// The flagship check.
    #[test]
    fn tensor_shape_mismatch_is_caught_at_compile_time() {
        assert_code(
            "fn main() {
                 let a: tensor<f32, 64, 128> = zeros(64, 128)
                 let b: tensor<f32, 64, 10> = zeros(64, 10)
                 let _ = a + b
             }",
            codes::SHAPE_MISMATCH,
        );
    }

    #[test]
    fn compatible_shapes_pass() {
        assert_clean(
            "fn main() {
                 let a: tensor<f32, 64, 128> = zeros(64, 128)
                 let b: tensor<f32, 128> = zeros(128)
                 let _c = a + b
             }",
        );
    }

    #[test]
    fn matmul_inner_dimension_is_checked() {
        assert_code(
            "fn main() {
                 let a: tensor<f32, 64, 784> = zeros(64, 784)
                 let b: tensor<f32, 128, 10> = zeros(128, 10)
                 let _ = a @ b
             }",
            codes::MATMUL_SHAPE,
        );
    }

    #[test]
    fn matmul_valid_shapes_pass() {
        assert_clean(
            "fn main() {
                 let a: tensor<f32, 64, 784> = zeros(64, 784)
                 let b: tensor<f32, 784, 128> = zeros(784, 128)
                 let _c = a @ b
             }",
        );
    }

    #[test]
    fn wildcard_dimensions_defer_the_check() {
        assert_clean(
            "fn main() {
                 let a: tensor<f32, _, 784> = zeros(64, 784)
                 let b: tensor<f32, 784, 10> = zeros(784, 10)
                 let _c = a @ b
             }",
        );
    }

    #[test]
    fn layer_chain_mismatch_is_caught() {
        assert_code(
            "model Net {
                 h = Dense(inputs=784, outputs=128, activation=\"relu\")
                 o = Dense(inputs=64, outputs=10)
             }
             fn main() { }",
            codes::LAYER_CHAIN_MISMATCH,
        );
    }

    #[test]
    fn consistent_layer_chain_passes() {
        assert_clean(
            "model Net {
                 h = Dense(inputs=784, outputs=128, activation=\"relu\")
                 o = Dense(inputs=128, outputs=10)
             }
             fn main() { }",
        );
    }

    #[test]
    fn unknown_activation_is_reported_with_a_suggestion() {
        let e = errors(
            "model Net { h = Dense(inputs=4, outputs=2, activation=\"relju\") }
             fn main() { }",
        );
        let d = e.iter().find(|d| d.code == codes::UNDEFINED_NAME).expect("expected an error");
        assert!(d.help.as_deref().unwrap_or("").contains("relu"));
    }

    #[test]
    fn assignment_to_immutable_is_reported() {
        assert_code(
            "fn main() { let x = 1\n x = 2 }",
            codes::ASSIGN_TO_IMMUTABLE,
        );
    }

    #[test]
    fn mutable_assignment_passes() {
        assert_clean("fn main() { let mut x = 1\n x = 2\n let _ = x }");
    }

    #[test]
    fn non_bool_condition_is_reported() {
        assert_code("fn main() { if 1 { } }", codes::CONDITION_NOT_BOOL);
    }

    #[test]
    fn missing_return_is_reported() {
        assert_code(
            "fn f(x: i64) -> i64 { let _ = x }
             fn main() { }",
            codes::MISSING_RETURN,
        );
    }

    #[test]
    fn return_on_all_paths_passes() {
        assert_clean(
            "fn f(x: i64) -> i64 { if x > 0 { return 1 } else { return 2 } }
             fn main() { let _ = f(1) }",
        );
    }

    #[test]
    fn wrong_argument_count_is_reported() {
        assert_code(
            "fn f(a: i64, b: i64) -> i64 { return a + b }
             fn main() { let _ = f(1) }",
            codes::WRONG_ARG_COUNT,
        );
    }

    #[test]
    fn break_outside_a_loop_is_reported() {
        assert_code("fn main() { break }", codes::BREAK_OUTSIDE_LOOP);
    }

    #[test]
    fn format_placeholder_count_is_checked() {
        assert_code(
            "fn main() { println(\"{} and {}\", 1) }",
            codes::FORMAT_ARG_COUNT,
        );
    }

    #[test]
    fn placeholder_counting_handles_escapes() {
        assert_eq!(count_placeholders("{} {}"), 2);
        assert_eq!(count_placeholders("{{}} literal"), 0);
        assert_eq!(count_placeholders("{:.3} value"), 1);
        assert_eq!(count_placeholders("none"), 0);
    }

    #[test]
    fn train_requires_epochs() {
        assert_code(
            "model Net { o = Dense(inputs=4, outputs=1) }
             fn main() {
                 let d = Dataset::from_csv(\"x.csv\", 0)
                 train Net using d { loss = \"mse\" }
             }",
            codes::MISSING_TRAIN_OPTION,
        );
    }

    #[test]
    fn unknown_loss_is_reported() {
        assert_code(
            "model Net { o = Dense(inputs=4, outputs=1) }
             fn main() {
                 let d = Dataset::from_csv(\"x.csv\", 0)
                 train Net using d { epochs = 1, loss = \"crossentropy\" }
             }",
            codes::UNKNOWN_LOSS,
        );
    }

    #[test]
    fn duplicate_definition_in_one_scope_is_reported() {
        assert_code(
            "fn main() { let x = 1\n let x = 2\n let _ = x }",
            codes::DUPLICATE_DEFINITION,
        );
    }

    #[test]
    fn shadowing_in_an_inner_scope_is_allowed() {
        assert_clean("fn main() { let x = 1\n { let x = 2\n let _ = x }\n let _ = x }");
    }

    #[test]
    fn unused_variable_warns_but_does_not_error() {
        let all = check_src("fn main() { let unused_thing = 1 }");
        assert!(all.iter().any(|d| d.code == codes::UNUSED_VARIABLE));
        assert!(errors("fn main() { let unused_thing = 1 }").is_empty());
    }

    #[test]
    fn underscore_prefix_suppresses_the_unused_warning() {
        let all = check_src("fn main() { let _scratch = 1 }");
        assert!(!all.iter().any(|d| d.code == codes::UNUSED_VARIABLE));
    }

    #[test]
    fn question_mark_outside_result_is_reported() {
        assert_code(
            "fn main() { let _d = Dataset::from_csv(\"x.csv\", 0)? }",
            codes::QUESTION_OUTSIDE_RESULT,
        );
    }

    #[test]
    fn many_independent_errors_are_all_reported() {
        // FR-ERR-009: one run, many diagnostics.
        let e = errors(
            "fn main() {
                 let _a = undefined_one
                 let _b = undefined_two
                 let _c = undefined_three
             }",
        );
        assert!(e.len() >= 3, "expected at least 3 errors, got {}", e.len());
    }
}
