//! KLIONS abstract syntax tree.
//!
//! Every node carries a span (FR-PAR-010). This crate depends only on
//! `klions-diagnostics` for `Span`, keeping the front end layered (DC-004).

use klions_diagnostics::Span;

// ============================ Program ============================

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub imports: Vec<Import>,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug)]
pub struct Import {
    pub path: Vec<String>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Item {
    Function(FunctionDecl),
    Model(ModelDecl),
    Const(ConstDecl),
}

impl Item {
    pub fn name(&self) -> &str {
        match self {
            Item::Function(f) => &f.name,
            Item::Model(m) => &m.name,
            Item::Const(c) => &c.name,
        }
    }
    pub fn span(&self) -> Span {
        match self {
            Item::Function(f) => f.span,
            Item::Model(m) => m.span,
            Item::Const(c) => c.span,
        }
    }
    pub fn name_span(&self) -> Span {
        match self {
            Item::Function(f) => f.name_span,
            Item::Model(m) => m.name_span,
            Item::Const(c) => c.name_span,
        }
    }
    pub fn kind_str(&self) -> &'static str {
        match self {
            Item::Function(_) => "function",
            Item::Model(_) => "model",
            Item::Const(_) => "constant",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ModelDecl {
    pub name: String,
    pub name_span: Span,
    /// Declaration order is forward-pass order (FR-PAR-006).
    pub layers: Vec<LayerBinding>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LayerBinding {
    pub name: String,
    pub name_span: Span,
    pub layer: LayerExpr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LayerExpr {
    pub kind: String,
    pub kind_span: Span,
    pub args: Vec<Arg>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: String,
    pub name_span: Span,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

// ============================ Types ============================

#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeExprKind {
    Named(String),
    /// `tensor<f32, 64, 784>` — FR-PAR-008.
    Tensor { elem: String, dims: Vec<Dim> },
    Result(Box<TypeExpr>, Box<TypeExpr>),
    Function(Vec<TypeExpr>, Box<TypeExpr>),
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Fixed(usize),
    /// `_` — unknown until run time (FR-TYP-008).
    Wild,
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{}", n),
            Dim::Wild => write!(f, "_"),
        }
    }
}

// ============================ Statements ============================

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let {
        mutable: bool,
        name: String,
        name_span: Span,
        ty: Option<TypeExpr>,
        value: Expr,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// `train M using D { ... }` — FR-PAR-007.
    Train {
        model: String,
        model_span: Span,
        dataset: String,
        dataset_span: Span,
        options: Vec<TrainOption>,
    },
    If {
        cond: Expr,
        then_block: Block,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        cond: Expr,
        body: Block,
    },
    For {
        var: String,
        var_span: Span,
        iter: IterExpr,
        body: Block,
    },
    Return(Option<Expr>),
    Break,
    Continue,
    /// `no_grad { ... }` — FR-AI-004.
    NoGrad(Block),
    Expr(Expr),
    Block(Block),
    /// Emitted by error recovery (FR-PAR-004) so later passes skip it.
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
}

impl AssignOp {
    pub fn to_binop(self) -> Option<BinOp> {
        match self {
            AssignOp::Assign => None,
            AssignOp::Add => Some(BinOp::Add),
            AssignOp::Sub => Some(BinOp::Sub),
            AssignOp::Mul => Some(BinOp::Mul),
            AssignOp::Div => Some(BinOp::Div),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            AssignOp::Assign => "=",
            AssignOp::Add => "+=",
            AssignOp::Sub => "-=",
            AssignOp::Mul => "*=",
            AssignOp::Div => "/=",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainOption {
    pub key: String,
    pub key_span: Span,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum IterExpr {
    /// `for i in 0..n`
    Range(Expr, Expr),
    /// `for row in tensor`
    Value(Expr),
}

// ============================ Expressions ============================

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    Ident(String),
    Array(Vec<Expr>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, TypeExpr),
    Call {
        callee: Box<Expr>,
        args: Vec<Arg>,
    },
    /// `x.field` and method receivers.
    Field {
        base: Box<Expr>,
        name: String,
        name_span: Span,
    },
    /// `Type::func(...)` — associated path, e.g. `Dataset::from_idx`.
    Path {
        base: String,
        name: String,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// Postfix `?` — FR-ERR-001, FR-TYP-014.
    Try(Box<Expr>),
    Ok(Option<Box<Expr>>),
    Err(Box<Expr>),
    /// Parse-error placeholder.
    Error,
}

#[derive(Clone, Debug)]
pub struct Arg {
    /// `Some` for a named argument (FR-PAR-011).
    pub name: Option<String>,
    pub name_span: Span,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

impl UnOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
        }
    }
}

/// FR-PAR-003 precedence, lowest to highest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// Matrix multiply — binds tighter than `*`.
    MatMul,
}

impl BinOp {
    pub fn as_str(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::MatMul => "@",
        }
    }
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }
    pub fn is_logical(self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
        )
    }
}

// ============================ Helpers ============================

impl Expr {
    pub fn error(span: Span) -> Expr {
        Expr { kind: ExprKind::Error, span }
    }
    pub fn is_error(&self) -> bool {
        matches!(self.kind, ExprKind::Error)
    }
    /// Is this a valid assignment target (grammar `LValue`)?
    pub fn is_lvalue(&self) -> bool {
        match &self.kind {
            ExprKind::Ident(_) => true,
            ExprKind::Field { base, .. } => base.is_lvalue(),
            ExprKind::Index { base, .. } => base.is_lvalue(),
            _ => false,
        }
    }
    /// Root identifier of an lvalue chain, for mutability checks.
    pub fn root_ident(&self) -> Option<&str> {
        match &self.kind {
            ExprKind::Ident(n) => Some(n.as_str()),
            ExprKind::Field { base, .. } => base.root_ident(),
            ExprKind::Index { base, .. } => base.root_ident(),
            _ => None,
        }
    }
}

/// FR-AI-007: the layer set available in v0.1.0.
pub const LAYER_NAMES: &[&str] = &[
    "Dense", "ReLU", "Sigmoid", "Tanh", "Softmax", "Flatten", "Dropout",
];

/// Layers apportioned to later releases — FR-ERR-010.
pub const DEFERRED_LAYERS: &[(&str, &str)] = &[
    ("Conv2D", "0.2.0"),
    ("Conv1D", "0.2.0"),
    ("MaxPool", "0.2.0"),
    ("MaxPool2D", "0.2.0"),
    ("AvgPool", "0.2.0"),
    ("BatchNorm", "0.2.0"),
    ("LayerNorm", "0.2.0"),
    ("LSTM", "0.4.0"),
    ("GRU", "0.4.0"),
    ("RNN", "0.4.0"),
    ("TransformerBlock", "0.4.0"),
    ("Attention", "0.4.0"),
    ("Embedding", "0.4.0"),
];

/// FR-AI-008: accepted `train` option keys.
pub const TRAIN_KEYS: &[&str] = &[
    "epochs",
    "optimizer",
    "loss",
    "batch_size",
    "metrics",
    "validation_split",
    "verbose",
];

/// Options apportioned to later releases — FR-ERR-010.
pub const DEFERRED_TRAIN_KEYS: &[(&str, &str)] = &[
    ("explainability", "0.4.0"),
    ("device", "0.3.0"),
    ("distributed", "0.5.0"),
    ("checkpoint", "0.2.0"),
    ("early_stopping", "0.2.0"),
];
