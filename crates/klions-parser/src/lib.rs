//! KLIONS parser — FR-PAR-001 … FR-PAR-011.
//!
//! Hand-written recursive descent with Pratt (precedence-climbing) expression
//! parsing, per FR-PAR-002. Error recovery synchronizes at statement
//! boundaries so that *n* independent syntax errors yield *n* diagnostics
//! in a single run (FR-PAR-004).

use klions_ast::*;
use klions_diagnostics::{codes, Diagnostic, Span};
use klions_lexer::{Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    pub diagnostics: Vec<Diagnostic>,
    /// Guards against pathological nesting in fuzzed input (NFR-Q-002).
    depth: usize,
    panicking: bool,
}

const MAX_DEPTH: usize = 256;

pub struct ParseResult {
    pub program: Program,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn parse(tokens: Vec<Token>) -> ParseResult {
    let mut p = Parser::new(tokens);
    let program = p.parse_program();
    ParseResult { program, diagnostics: p.diagnostics }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0, diagnostics: Vec::new(), depth: 0, panicking: false }
    }

    // ---------------- token access ----------------

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }
    fn peek_at(&self, n: usize) -> &TokenKind {
        &self.tokens[(self.pos + n).min(self.tokens.len() - 1)].kind
    }
    fn span(&self) -> Span {
        self.tokens[self.pos.min(self.tokens.len() - 1)].span
    }
    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1).min(self.tokens.len() - 1)].span
    }
    fn at_eof(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }
    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos.min(self.tokens.len() - 1)].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn check(&self, k: &TokenKind) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(k)
    }
    fn eat(&mut self, k: TokenKind) -> bool {
        if self.check(&k) {
            self.advance();
            true
        } else {
            false
        }
    }
    /// Skip virtual newlines — used where a construct spans lines.
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::VirtualNewline) {
            self.advance();
        }
    }
    fn skip_terminators(&mut self) {
        while matches!(self.peek(), TokenKind::VirtualNewline | TokenKind::Semi) {
            self.advance();
        }
    }

    fn error(&mut self, code: &'static str, span: Span, msg: impl Into<String>) {
        if self.panicking {
            return; // suppress cascades until we resynchronize
        }
        self.panicking = true;
        self.diagnostics.push(Diagnostic::error(code, span, msg));
    }

    fn error_with_help(
        &mut self,
        code: &'static str,
        span: Span,
        msg: impl Into<String>,
        help: impl Into<String>,
    ) {
        if self.panicking {
            return;
        }
        self.panicking = true;
        self.diagnostics
            .push(Diagnostic::error(code, span, msg).with_help(help));
    }

    /// Emit a diagnostic that does not enter panic mode (recoverable, local).
    fn soft_error(&mut self, d: Diagnostic) {
        self.diagnostics.push(d);
    }

    fn expect(&mut self, k: TokenKind, ctx: &str) -> bool {
        if self.check(&k) {
            self.advance();
            true
        } else {
            let found = self.peek().describe();
            let span = self.span();
            self.error_with_help(
                codes::EXPECTED_TOKEN,
                span,
                format!("expected `{}` {}, found {}", k.lexeme(), ctx, found),
                format!("insert `{}` here", k.lexeme()),
            );
            false
        }
    }

    fn expect_ident(&mut self, ctx: &str) -> (String, Span) {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let sp = self.span();
                self.advance();
                (name, sp)
            }
            other => {
                let sp = self.span();
                // A keyword where a name belongs is nearly always someone using
                // a reserved word as a variable. Saying so is far more useful
                // than "expected an identifier".
                if is_keyword(&other) {
                    self.error_with_help(
                        codes::EXPECTED_TOKEN,
                        sp,
                        format!(
                            "`{}` is a keyword and cannot be used as a name",
                            other.lexeme()
                        ),
                        format!(
                            "rename it, e.g. `{}_data` or `my_{}`",
                            other.lexeme(),
                            other.lexeme()
                        ),
                    );
                    // Consume it. Leaving it in place would let the statement
                    // parser pick it up again and report a second, confusing
                    // error for the same mistake.
                    self.advance();
                } else {
                    self.error(
                        codes::EXPECTED_TOKEN,
                        sp,
                        format!("expected an identifier {}, found {}", ctx, other.describe()),
                    );
                }
                ("<error>".to_string(), sp)
            }
        }
    }

    /// FR-LEX-009 / grammar `Terminator`.
    fn expect_terminator(&mut self, ctx: &str) {
        match self.peek() {
            TokenKind::Semi | TokenKind::VirtualNewline => {
                self.advance();
                self.skip_terminators();
            }
            TokenKind::Eof | TokenKind::RBrace => {}
            other => {
                let found = other.describe();
                let sp = self.span();
                self.error_with_help(
                    codes::EXPECTED_TOKEN,
                    sp,
                    format!("expected end of statement after {}, found {}", ctx, found),
                    "end the statement with a newline or `;`",
                );
                self.synchronize();
            }
        }
    }

    /// FR-PAR-004: resynchronize at the next statement boundary.
    fn synchronize(&mut self) {
        self.panicking = false;
        loop {
            match self.peek() {
                TokenKind::Eof => return,
                TokenKind::Semi | TokenKind::VirtualNewline => {
                    self.advance();
                    self.skip_terminators();
                    return;
                }
                TokenKind::RBrace => return,
                TokenKind::Let
                | TokenKind::Fn
                | TokenKind::Model
                | TokenKind::Train
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return
                | TokenKind::Import
                | TokenKind::Const
                | TokenKind::NoGrad => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // ---------------- program ----------------

    pub fn parse_program(&mut self) -> Program {
        let mut program = Program::default();
        self.skip_terminators();

        while matches!(self.peek(), TokenKind::Import) {
            if let Some(i) = self.parse_import() {
                program.imports.push(i);
            }
            self.skip_terminators();
        }

        while !self.at_eof() {
            self.skip_terminators();
            if self.at_eof() {
                break;
            }
            match self.peek() {
                TokenKind::Import => {
                    // Late import: accepted, but note the convention.
                    let sp = self.span();
                    if let Some(i) = self.parse_import() {
                        program.imports.push(i);
                    }
                    self.soft_error(
                        Diagnostic::warning(
                            codes::BAD_IMPORT_PATH,
                            sp,
                            "`import` appears after the first item",
                        )
                        .with_help("move all imports to the top of the file"),
                    );
                }
                TokenKind::Fn => {
                    if let Some(f) = self.parse_function() {
                        program.items.push(Item::Function(f));
                    }
                }
                TokenKind::Model => {
                    if let Some(m) = self.parse_model() {
                        program.items.push(Item::Model(m));
                    }
                }
                TokenKind::Const => {
                    if let Some(c) = self.parse_const() {
                        program.items.push(Item::Const(c));
                    }
                }
                other => {
                    let found = other.describe();
                    let sp = self.span();
                    self.error_with_help(
                        codes::EXPECTED_STATEMENT,
                        sp,
                        format!("expected an item at the top level, found {}", found),
                        "top-level items are `fn`, `model`, `const`, and `import`; \
                         put statements inside `fn main() { ... }`",
                    );
                    self.advance();
                    self.synchronize();
                }
            }
            self.skip_terminators();
        }
        program
    }

    /// FR-PAR-009: `import klions::ai::nn`
    fn parse_import(&mut self) -> Option<Import> {
        let start = self.span();
        self.advance(); // `import`
        let mut path = Vec::new();
        let (first, _) = self.expect_ident("after `import`");
        path.push(first);
        while self.eat(TokenKind::ColonColon) {
            let (seg, _) = self.expect_ident("in import path");
            path.push(seg);
        }
        let span = start.to(self.prev_span());
        self.expect_terminator("import path");
        self.panicking = false;

        if path.first().map(|s| s.as_str()) != Some("klions") {
            self.soft_error(
                Diagnostic::error(
                    codes::BAD_IMPORT_PATH,
                    span,
                    format!("unknown module `{}`", path.join("::")),
                )
                .with_help(
                    "v0.1.0 resolves only the built-in `klions::` namespace; \
                     user modules arrive with the module system in 0.2.0",
                ),
            );
        } else if !is_known_module(&path) {
            self.soft_error(
                Diagnostic::error(
                    codes::BAD_IMPORT_PATH,
                    span,
                    format!("`{}` is not a built-in module", path.join("::")),
                )
                .with_help(
                    "available modules: klions::ai::nn, klions::ai::data, \
                     klions::ai::optim, klions::math, klions::io",
                ),
            );
        }
        Some(Import { path, span })
    }

    fn parse_function(&mut self) -> Option<FunctionDecl> {
        let start = self.span();
        self.advance(); // `fn`
        let (name, name_span) = self.expect_ident("after `fn`");
        self.expect(TokenKind::LParen, "after function name");

        let mut params = Vec::new();
        self.skip_newlines();
        if !self.check(&TokenKind::RParen) {
            loop {
                self.skip_newlines();
                let p_start = self.span();
                let (pname, _) = self.expect_ident("in parameter list");
                self.expect(TokenKind::Colon, "after parameter name");
                let ty = self.parse_type();
                params.push(Param { name: pname, ty, span: p_start.to(self.prev_span()) });
                self.skip_newlines();
                if !self.eat(TokenKind::Comma) {
                    break;
                }
                self.skip_newlines();
                if self.check(&TokenKind::RParen) {
                    break; // trailing comma
                }
            }
        }
        self.skip_newlines();
        self.expect(TokenKind::RParen, "after parameters");

        let ret = if self.eat(TokenKind::Arrow) { Some(self.parse_type()) } else { None };

        self.panicking = false;
        self.skip_newlines();
        let body = self.parse_block();
        let span = start.to(self.prev_span());
        Some(FunctionDecl { name, name_span, params, ret, body, span })
    }

    /// FR-PAR-006: layer bindings keep declaration order.
    fn parse_model(&mut self) -> Option<ModelDecl> {
        let start = self.span();
        self.advance(); // `model`
        let (name, name_span) = self.expect_ident("after `model`");
        self.skip_newlines();
        self.expect(TokenKind::LBrace, "to open the model body");
        self.panicking = false;

        let mut layers = Vec::new();
        loop {
            self.skip_terminators();
            if self.check(&TokenKind::RBrace) || self.at_eof() {
                break;
            }
            let l_start = self.span();
            let (lname, lname_span) = self.expect_ident("as a layer name");
            if !self.expect(TokenKind::Eq, "after the layer name") {
                self.synchronize();
                continue;
            }
            let layer = match self.parse_layer_expr() {
                Some(l) => l,
                None => {
                    self.synchronize();
                    continue;
                }
            };
            let span = l_start.to(self.prev_span());
            layers.push(LayerBinding { name: lname, name_span: lname_span, layer, span });
            self.expect_terminator("layer binding");
            self.panicking = false;
        }
        self.expect(TokenKind::RBrace, "to close the model body");
        let span = start.to(self.prev_span());

        if layers.is_empty() {
            self.soft_error(
                Diagnostic::error(codes::MODEL_EMPTY, span, format!("model `{}` has no layers", name))
                    .with_help("add at least one layer, e.g. `out = Dense(inputs=4, outputs=1)`"),
            );
        }
        self.panicking = false;
        Some(ModelDecl { name, name_span, layers, span })
    }

    fn parse_layer_expr(&mut self) -> Option<LayerExpr> {
        let start = self.span();
        let (kind, kind_span) = match self.peek().clone() {
            TokenKind::Ident(n) => {
                let sp = self.span();
                self.advance();
                (n, sp)
            }
            other => {
                let sp = self.span();
                self.error(
                    codes::EXPECTED_TOKEN,
                    sp,
                    format!("expected a layer type, found {}", other.describe()),
                );
                return None;
            }
        };

        // FR-ERR-010: deferred layers name their release rather than being ignored.
        if let Some((_, release)) = DEFERRED_LAYERS.iter().find(|(n, _)| *n == kind) {
            self.soft_error(
                Diagnostic::error(
                    codes::DEFERRED_FEATURE,
                    kind_span,
                    format!("layer `{}` is not available in v0.1.0", kind),
                )
                .with_note(format!("`{}` is scheduled for release {}", kind, release))
                .with_help("v0.1.0 supports dense feed-forward networks: Dense, ReLU, Sigmoid, Tanh, Softmax, Flatten, Dropout"),
            );
        } else if !LAYER_NAMES.contains(&kind.as_str()) {
            let sugg = nearest(&kind, LAYER_NAMES);
            let mut d = Diagnostic::error(
                codes::UNKNOWN_LAYER,
                kind_span,
                format!("unknown layer type `{}`", kind),
            );
            d = match sugg {
                Some(s) => d.with_help(format!("did you mean `{}`?", s)),
                None => d.with_help(format!("available layers: {}", LAYER_NAMES.join(", "))),
            };
            self.soft_error(d);
        }

        self.expect(TokenKind::LParen, "after the layer type");
        let args = self.parse_args();
        self.expect(TokenKind::RParen, "to close the layer arguments");
        Some(LayerExpr { kind, kind_span, args, span: start.to(self.prev_span()) })
    }

    fn parse_const(&mut self) -> Option<ConstDecl> {
        let start = self.span();
        self.advance(); // `const`
        let (name, name_span) = self.expect_ident("after `const`");
        self.expect(TokenKind::Colon, "after the constant name");
        let ty = self.parse_type();
        self.expect(TokenKind::Eq, "after the constant type");
        let value = self.parse_expr();
        let span = start.to(self.prev_span());
        self.expect_terminator("constant declaration");
        self.panicking = false;
        Some(ConstDecl { name, name_span, ty, value, span })
    }

    // ---------------- types ----------------

    pub fn parse_type(&mut self) -> TypeExpr {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::Fn => {
                self.advance();
                self.expect(TokenKind::LParen, "in function type");
                let mut params = Vec::new();
                if !self.check(&TokenKind::RParen) {
                    loop {
                        params.push(self.parse_type());
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen, "in function type");
                self.expect(TokenKind::Arrow, "in function type");
                let ret = self.parse_type();
                TypeExpr {
                    kind: TypeExprKind::Function(params, Box::new(ret)),
                    span: start.to(self.prev_span()),
                }
            }
            TokenKind::Ident(name) => {
                self.advance();
                match name.as_str() {
                    "tensor" => self.parse_tensor_type(start),
                    "Result" => {
                        self.expect(TokenKind::Lt, "after `Result`");
                        let ok = self.parse_type();
                        self.expect(TokenKind::Comma, "in `Result<T, E>`");
                        let err = self.parse_type();
                        self.expect(TokenKind::Gt, "to close `Result<T, E>`");
                        TypeExpr {
                            kind: TypeExprKind::Result(Box::new(ok), Box::new(err)),
                            span: start.to(self.prev_span()),
                        }
                    }
                    "auto" => TypeExpr { kind: TypeExprKind::Auto, span: start },
                    _ => TypeExpr { kind: TypeExprKind::Named(name), span: start },
                }
            }
            other => {
                let sp = self.span();
                self.error_with_help(
                    codes::EXPECTED_TOKEN,
                    sp,
                    format!("expected a type, found {}", other.describe()),
                    "types are i32, i64, f32, f64, bool, string, void, tensor<...>, Result<T, E>, Model, Dataset, Optimizer",
                );
                TypeExpr { kind: TypeExprKind::Named("<error>".into()), span: sp }
            }
        }
    }

    /// FR-PAR-008: `tensor<f32, 64, 784>` with `_` wildcards.
    fn parse_tensor_type(&mut self, start: Span) -> TypeExpr {
        if !self.eat(TokenKind::Lt) {
            let sp = self.span();
            self.soft_error(
                Diagnostic::error(
                    codes::BAD_TENSOR_TYPE,
                    sp,
                    "`tensor` requires an element type and dimensions",
                )
                .with_help("write it as `tensor<f32, 64, 784>`; use `_` for a dimension known only at run time"),
            );
            return TypeExpr {
                kind: TypeExprKind::Tensor { elem: "f32".into(), dims: vec![] },
                span: start,
            };
        }
        let elem = match self.peek().clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            other => {
                let sp = self.span();
                self.soft_error(
                    Diagnostic::error(
                        codes::BAD_TENSOR_TYPE,
                        sp,
                        format!("expected a tensor element type, found {}", other.describe()),
                    )
                    .with_help("v0.1.0 tensors are `f32`"),
                );
                "f32".to_string()
            }
        };
        let mut dims = Vec::new();
        while self.eat(TokenKind::Comma) {
            match self.peek().clone() {
                TokenKind::Int(v) => {
                    let sp = self.span();
                    self.advance();
                    if v < 0 {
                        self.soft_error(Diagnostic::error(
                            codes::NEGATIVE_DIMENSION,
                            sp,
                            format!("tensor dimension must be non-negative, found {}", v),
                        ));
                        dims.push(Dim::Fixed(0));
                    } else {
                        dims.push(Dim::Fixed(v as usize));
                    }
                }
                TokenKind::Underscore => {
                    self.advance();
                    dims.push(Dim::Wild);
                }
                other => {
                    let sp = self.span();
                    self.soft_error(
                        Diagnostic::error(
                            codes::BAD_TENSOR_TYPE,
                            sp,
                            format!("expected a dimension, found {}", other.describe()),
                        )
                        .with_help("dimensions are integer literals or `_`"),
                    );
                    self.advance();
                    dims.push(Dim::Wild);
                }
            }
        }
        self.expect(TokenKind::Gt, "to close the tensor type");
        TypeExpr {
            kind: TypeExprKind::Tensor { elem, dims },
            span: start.to(self.prev_span()),
        }
    }

    // ---------------- statements ----------------

    fn parse_block(&mut self) -> Block {
        let start = self.span();
        if !self.expect(TokenKind::LBrace, "to open a block") {
            self.synchronize();
            return Block { stmts: vec![], span: start };
        }
        self.panicking = false;
        let mut stmts = Vec::new();
        loop {
            self.skip_terminators();
            if self.check(&TokenKind::RBrace) || self.at_eof() {
                break;
            }
            let before = self.pos;
            let s = self.parse_stmt();
            stmts.push(s);
            if self.pos == before {
                self.advance(); // guarantee progress
            }
        }
        self.expect(TokenKind::RBrace, "to close the block");
        self.panicking = false;
        Block { stmts, span: start.to(self.prev_span()) }
    }

    fn parse_stmt(&mut self) -> Stmt {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            let sp = self.span();
            self.error(codes::UNEXPECTED_TOKEN, sp, "expression nests too deeply");
            self.synchronize();
            return Stmt { kind: StmtKind::Error, span: sp };
        }
        let s = self.parse_stmt_inner();
        self.depth -= 1;
        s
    }

    fn parse_stmt_inner(&mut self) -> Stmt {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::Let => self.parse_let(start),
            TokenKind::Train => self.parse_train(start),
            TokenKind::If => {
                let s = self.parse_if(start);
                self.skip_terminators();
                s
            }
            TokenKind::While => {
                self.advance();
                let cond = self.parse_expr_no_struct();
                self.skip_newlines();
                let body = self.parse_block();
                self.skip_terminators();
                Stmt {
                    kind: StmtKind::While { cond, body },
                    span: start.to(self.prev_span()),
                }
            }
            TokenKind::For => self.parse_for(start),
            TokenKind::NoGrad => {
                self.advance();
                self.skip_newlines();
                let b = self.parse_block();
                self.skip_terminators();
                Stmt { kind: StmtKind::NoGrad(b), span: start.to(self.prev_span()) }
            }
            TokenKind::Return => {
                self.advance();
                let value = if matches!(
                    self.peek(),
                    TokenKind::Semi | TokenKind::VirtualNewline | TokenKind::RBrace | TokenKind::Eof
                ) {
                    None
                } else {
                    Some(self.parse_expr())
                };
                let span = start.to(self.prev_span());
                self.expect_terminator("`return`");
                self.panicking = false;
                Stmt { kind: StmtKind::Return(value), span }
            }
            TokenKind::Break => {
                self.advance();
                let span = start.to(self.prev_span());
                self.expect_terminator("`break`");
                self.panicking = false;
                Stmt { kind: StmtKind::Break, span }
            }
            TokenKind::Continue => {
                self.advance();
                let span = start.to(self.prev_span());
                self.expect_terminator("`continue`");
                self.panicking = false;
                Stmt { kind: StmtKind::Continue, span }
            }
            TokenKind::LBrace => {
                let b = self.parse_block();
                self.skip_terminators();
                Stmt { kind: StmtKind::Block(b), span: start.to(self.prev_span()) }
            }
            TokenKind::Fn | TokenKind::Model => {
                let kw = self.peek().lexeme();
                self.error_with_help(
                    codes::EXPECTED_STATEMENT,
                    start,
                    format!("`{}` declarations are not allowed inside a function body", kw),
                    format!("move the `{}` declaration to the top level of the file", kw),
                );
                // Consume it anyway so the rest of the file still parses.
                if kw == "fn" {
                    let _ = self.parse_function();
                } else {
                    let _ = self.parse_model();
                }
                self.panicking = false;
                Stmt { kind: StmtKind::Error, span: start.to(self.prev_span()) }
            }
            _ => self.parse_expr_or_assign(start),
        }
    }

    fn parse_let(&mut self, start: Span) -> Stmt {
        self.advance(); // `let`
        let mutable = self.eat(TokenKind::Mut); // FR-PAR-005
        // `let _ = expr` discards the value without binding a name.
        let (name, name_span) = if matches!(self.peek(), TokenKind::Underscore) {
            let sp = self.span();
            self.advance();
            ("_".to_string(), sp)
        } else {
            self.expect_ident("after `let`")
        };
        let ty = if self.eat(TokenKind::Colon) { Some(self.parse_type()) } else { None };
        if !self.expect(TokenKind::Eq, "in a `let` binding") {
            self.synchronize();
            return Stmt { kind: StmtKind::Error, span: start.to(self.prev_span()) };
        }
        let value = self.parse_expr();
        let span = start.to(self.prev_span());
        self.expect_terminator("`let` binding");
        self.panicking = false;
        Stmt {
            kind: StmtKind::Let { mutable, name, name_span, ty, value },
            span,
        }
    }

    /// FR-PAR-007: `train <model> using <dataset> { key = value, ... }`
    fn parse_train(&mut self, start: Span) -> Stmt {
        self.advance(); // `train`
        let (model, model_span) = self.expect_ident("after `train`");
        if !self.expect(TokenKind::Using, "after the model name") {
            self.synchronize();
            return Stmt { kind: StmtKind::Error, span: start.to(self.prev_span()) };
        }
        let (dataset, dataset_span) = self.expect_ident("after `using`");

        let mut options = Vec::new();
        self.skip_newlines();
        if self.eat(TokenKind::LBrace) {
            self.panicking = false;
            loop {
                self.skip_terminators();
                if self.check(&TokenKind::RBrace) || self.at_eof() {
                    break;
                }
                let o_start = self.span();
                let (key, key_span) = self.expect_ident("as a training option");
                self.check_train_key(&key, key_span);
                if !self.expect(TokenKind::Eq, "after the option name") {
                    self.synchronize();
                    continue;
                }
                let value = self.parse_expr();
                options.push(TrainOption {
                    key,
                    key_span,
                    value,
                    span: o_start.to(self.prev_span()),
                });
                self.skip_newlines();
                if !self.eat(TokenKind::Comma) {
                    self.skip_terminators();
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                }
            }
            self.skip_terminators();
            self.expect(TokenKind::RBrace, "to close the training options");
        }
        let span = start.to(self.prev_span());
        self.expect_terminator("`train` statement");
        self.panicking = false;
        Stmt {
            kind: StmtKind::Train { model, model_span, dataset, dataset_span, options },
            span,
        }
    }

    fn check_train_key(&mut self, key: &str, span: Span) {
        if TRAIN_KEYS.contains(&key) {
            return;
        }
        // FR-ERR-010: deferred options error rather than being silently ignored.
        if let Some((_, release)) = DEFERRED_TRAIN_KEYS.iter().find(|(k, _)| *k == key) {
            self.soft_error(
                Diagnostic::error(
                    codes::DEFERRED_FEATURE,
                    span,
                    format!("training option `{}` is not available in v0.1.0", key),
                )
                .with_note(format!("`{}` is scheduled for release {}", key, release))
                .with_help("remove the option; it is not silently ignored because that would mislead you about what ran"),
            );
            return;
        }
        let mut d = Diagnostic::error(
            codes::UNKNOWN_TRAIN_OPTION,
            span,
            format!("unknown training option `{}`", key),
        );
        d = match nearest(key, TRAIN_KEYS) {
            Some(s) => d.with_help(format!("did you mean `{}`?", s)),
            None => d.with_help(format!("valid options: {}", TRAIN_KEYS.join(", "))),
        };
        self.soft_error(d);
    }

    fn parse_if(&mut self, start: Span) -> Stmt {
        self.advance(); // `if`
        let cond = self.parse_expr_no_struct();
        self.skip_newlines();
        let then_block = self.parse_block();
        self.skip_newlines();
        let else_branch = if self.eat(TokenKind::Else) {
            self.skip_newlines();
            let e_start = self.span();
            if matches!(self.peek(), TokenKind::If) {
                Some(Box::new(self.parse_if(e_start)))
            } else {
                let b = self.parse_block();
                Some(Box::new(Stmt {
                    kind: StmtKind::Block(b),
                    span: e_start.to(self.prev_span()),
                }))
            }
        } else {
            None
        };
        Stmt {
            kind: StmtKind::If { cond, then_block, else_branch },
            span: start.to(self.prev_span()),
        }
    }

    fn parse_for(&mut self, start: Span) -> Stmt {
        self.advance(); // `for`
        let (var, var_span) = self.expect_ident("after `for`");
        self.expect(TokenKind::In, "after the loop variable");
        let first = self.parse_expr_no_struct();
        let iter = if self.eat(TokenKind::DotDot) {
            let end = self.parse_expr_no_struct();
            IterExpr::Range(first, end)
        } else {
            IterExpr::Value(first)
        };
        self.skip_newlines();
        let body = self.parse_block();
        self.skip_terminators();
        Stmt {
            kind: StmtKind::For { var, var_span, iter, body },
            span: start.to(self.prev_span()),
        }
    }

    fn parse_expr_or_assign(&mut self, start: Span) -> Stmt {
        let lhs = self.parse_expr();
        let op = match self.peek() {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::Add),
            TokenKind::MinusEq => Some(AssignOp::Sub),
            TokenKind::StarEq => Some(AssignOp::Mul),
            TokenKind::SlashEq => Some(AssignOp::Div),
            _ => None,
        };
        if let Some(op) = op {
            let op_span = self.span();
            self.advance();
            let value = self.parse_expr();
            let span = start.to(self.prev_span());
            if !lhs.is_lvalue() && !lhs.is_error() {
                self.soft_error(
                    Diagnostic::error(
                        codes::UNEXPECTED_TOKEN,
                        op_span,
                        "the left-hand side of an assignment must be a variable, field, or index",
                    )
                    .with_help("assign to a name, e.g. `x = ...`, `x.field = ...`, or `x[i] = ...`"),
                );
            }
            self.expect_terminator("assignment");
            self.panicking = false;
            return Stmt { kind: StmtKind::Assign { target: lhs, op, value }, span };
        }
        let span = start.to(self.prev_span());
        self.expect_terminator("expression");
        self.panicking = false;
        Stmt { kind: StmtKind::Expr(lhs), span }
    }

    // ---------------- expressions (Pratt) ----------------

    pub fn parse_expr(&mut self) -> Expr {
        self.parse_binary(0, true)
    }
    /// In `if`/`while`/`for` headers a `{` opens the body, not a literal.
    fn parse_expr_no_struct(&mut self) -> Expr {
        self.parse_binary(0, false)
    }

    fn parse_binary(&mut self, min_bp: u8, allow_brace: bool) -> Expr {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            let sp = self.span();
            self.error(codes::UNEXPECTED_TOKEN, sp, "expression nests too deeply");
            return Expr::error(sp);
        }
        let mut lhs = self.parse_unary(allow_brace);
        loop {
            let (op, bp) = match binop_of(self.peek()) {
                Some(x) => x,
                None => break,
            };
            if bp < min_bp {
                break;
            }
            self.advance();
            self.skip_newlines(); // allow `a +\n b`
            // All binary operators are left-associative (FR-PAR-003).
            let rhs = self.parse_binary(bp + 1, allow_brace);
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }
        self.depth -= 1;
        lhs
    }

    fn parse_unary(&mut self, allow_brace: bool) -> Expr {
        let start = self.span();
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let e = self.parse_unary(allow_brace);
                let span = start.to(e.span);
                Expr { kind: ExprKind::Unary(UnOp::Neg, Box::new(e)), span }
            }
            TokenKind::Bang => {
                self.advance();
                let e = self.parse_unary(allow_brace);
                let span = start.to(e.span);
                Expr { kind: ExprKind::Unary(UnOp::Not, Box::new(e)), span }
            }
            _ => {
                let e = self.parse_postfix(allow_brace);
                self.parse_cast(e)
            }
        }
    }

    /// FR-TYP-004: `expr as T`, left-associative.
    fn parse_cast(&mut self, mut e: Expr) -> Expr {
        while matches!(self.peek(), TokenKind::As) {
            self.advance();
            let ty = self.parse_type();
            let span = e.span.to(ty.span);
            e = Expr { kind: ExprKind::Cast(Box::new(e), ty), span };
        }
        e
    }

    fn parse_postfix(&mut self, allow_brace: bool) -> Expr {
        let mut e = self.parse_primary(allow_brace);
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.advance();
                    let (name, name_span) = self.expect_ident("after `.`");
                    let span = e.span.to(name_span);
                    e = Expr {
                        kind: ExprKind::Field { base: Box::new(e), name, name_span },
                        span,
                    };
                }
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_args();
                    self.expect(TokenKind::RParen, "to close the argument list");
                    let span = e.span.to(self.prev_span());
                    e = Expr { kind: ExprKind::Call { callee: Box::new(e), args }, span };
                }
                TokenKind::LBracket => {
                    self.advance();
                    self.skip_newlines();
                    let mut indices = Vec::new();
                    if !self.check(&TokenKind::RBracket) {
                        loop {
                            indices.push(self.parse_expr());
                            self.skip_newlines();
                            if !self.eat(TokenKind::Comma) {
                                break;
                            }
                            self.skip_newlines();
                        }
                    }
                    self.expect(TokenKind::RBracket, "to close the index");
                    let span = e.span.to(self.prev_span());
                    e = Expr { kind: ExprKind::Index { base: Box::new(e), indices }, span };
                }
                TokenKind::Question => {
                    let sp = self.span();
                    self.advance();
                    let span = e.span.to(sp);
                    e = Expr { kind: ExprKind::Try(Box::new(e)), span };
                }
                _ => break,
            }
        }
        e
    }

    /// FR-PAR-011: positional arguments must precede named ones.
    fn parse_args(&mut self) -> Vec<Arg> {
        let mut args = Vec::new();
        let mut seen_named: Option<Span> = None;
        self.skip_newlines();
        if self.check(&TokenKind::RParen) {
            return args;
        }
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::RParen) || self.at_eof() {
                break;
            }
            let a_start = self.span();
            // `name = expr` is a named argument; `name == x` is not.
            let named = matches!(self.peek(), TokenKind::Ident(_))
                && matches!(self.peek_at(1), TokenKind::Eq);
            let (name, name_span) = if named {
                let (n, sp) = self.expect_ident("as an argument name");
                self.advance(); // `=`
                (Some(n), sp)
            } else {
                (None, a_start)
            };
            if name.is_some() {
                seen_named = Some(name_span);
            } else if let Some(prev) = seen_named {
                self.soft_error(
                    Diagnostic::error(
                        codes::POSITIONAL_AFTER_NAMED,
                        a_start,
                        "positional argument follows a named argument",
                    )
                    .with_secondary(prev, "first named argument here")
                    .with_help("put all positional arguments before the named ones"),
                );
            }
            self.skip_newlines();
            let value = self.parse_expr();
            args.push(Arg { name, name_span, value, span: a_start.to(self.prev_span()) });
            self.skip_newlines();
            if !self.eat(TokenKind::Comma) {
                break;
            }
            self.skip_newlines();
            if self.check(&TokenKind::RParen) {
                break; // trailing comma
            }
        }
        self.skip_newlines();
        args
    }

    fn parse_primary(&mut self, allow_brace: bool) -> Expr {
        let start = self.span();
        let kind = match self.peek().clone() {
            TokenKind::Int(v) => {
                self.advance();
                ExprKind::Int(v)
            }
            TokenKind::Float(v) => {
                self.advance();
                ExprKind::Float(v)
            }
            TokenKind::Str(s) => {
                self.advance();
                ExprKind::Str(s)
            }
            TokenKind::True => {
                self.advance();
                ExprKind::Bool(true)
            }
            TokenKind::False => {
                self.advance();
                ExprKind::Bool(false)
            }
            TokenKind::Null => {
                self.advance();
                ExprKind::Null
            }
            TokenKind::Ok => {
                self.advance();
                self.expect(TokenKind::LParen, "after `Ok`");
                let inner = if self.check(&TokenKind::RParen) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()))
                };
                self.expect(TokenKind::RParen, "to close `Ok(...)`");
                ExprKind::Ok(inner)
            }
            TokenKind::Err => {
                self.advance();
                self.expect(TokenKind::LParen, "after `Err`");
                let inner = self.parse_expr();
                self.expect(TokenKind::RParen, "to close `Err(...)`");
                ExprKind::Err(Box::new(inner))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // `Type::function` path (FR-PAR-009 style, used by the stdlib).
                if matches!(self.peek(), TokenKind::ColonColon) {
                    self.advance();
                    let (member, msp) = self.expect_ident("after `::`");
                    let span = start.to(msp);
                    ExprKind::Path { base: name, name: member, span }
                } else {
                    ExprKind::Ident(name)
                }
            }
            TokenKind::Underscore => {
                self.advance();
                ExprKind::Ident("_".to_string())
            }
            TokenKind::LParen => {
                self.advance();
                self.skip_newlines();
                let e = self.parse_expr();
                self.skip_newlines();
                self.expect(TokenKind::RParen, "to close the parenthesized expression");
                return Expr { kind: e.kind, span: start.to(self.prev_span()) };
            }
            TokenKind::LBracket => {
                self.advance();
                self.skip_newlines();
                let mut items = Vec::new();
                if !self.check(&TokenKind::RBracket) {
                    loop {
                        self.skip_newlines();
                        if self.check(&TokenKind::RBracket) {
                            break;
                        }
                        items.push(self.parse_expr());
                        self.skip_newlines();
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.skip_newlines();
                self.expect(TokenKind::RBracket, "to close the array literal");
                ExprKind::Array(items)
            }
            TokenKind::LBrace if allow_brace => {
                self.error_with_help(
                    codes::EXPECTED_EXPRESSION,
                    start,
                    "expected an expression, found `{`",
                    "blocks are not expressions in v0.1.0",
                );
                self.advance();
                ExprKind::Error
            }
            other => {
                let sp = self.span();
                self.error_with_help(
                    codes::EXPECTED_EXPRESSION,
                    sp,
                    format!("expected an expression, found {}", other.describe()),
                    "an expression is a literal, a name, a call, or an operation on those",
                );
                if !matches!(other, TokenKind::Eof | TokenKind::RBrace | TokenKind::RParen) {
                    self.advance();
                }
                ExprKind::Error
            }
        };
        Expr { kind, span: start.to(self.prev_span()) }
    }
}

/// Does this token spell a keyword? Used to give a targeted diagnostic when
/// one appears where a name is required.
fn is_keyword(k: &TokenKind) -> bool {
    use TokenKind as T;
    matches!(
        k,
        T::Let
            | T::Mut
            | T::Fn
            | T::Const
            | T::Return
            | T::If
            | T::Else
            | T::While
            | T::For
            | T::In
            | T::Model
            | T::Train
            | T::Using
            | T::Import
            | T::True
            | T::False
            | T::Null
            | T::As
            | T::NoGrad
            | T::Break
            | T::Continue
            | T::Reserved(_)
    )
}

/// FR-PAR-003 binding powers. Higher binds tighter.
fn binop_of(k: &TokenKind) -> Option<(BinOp, u8)> {
    use TokenKind as T;
    Some(match k {
        T::OrOr => (BinOp::Or, 1),
        T::AndAnd => (BinOp::And, 2),
        T::EqEq => (BinOp::Eq, 3),
        T::BangEq => (BinOp::Ne, 3),
        T::Lt => (BinOp::Lt, 4),
        T::Le => (BinOp::Le, 4),
        T::Gt => (BinOp::Gt, 4),
        T::Ge => (BinOp::Ge, 4),
        T::Plus => (BinOp::Add, 5),
        T::Minus => (BinOp::Sub, 5),
        T::Star => (BinOp::Mul, 6),
        T::Slash => (BinOp::Div, 6),
        T::Percent => (BinOp::Rem, 6),
        T::At => (BinOp::MatMul, 7),
        _ => return None,
    })
}

fn is_known_module(path: &[String]) -> bool {
    let joined = path.join("::");
    matches!(
        joined.as_str(),
        "klions"
            | "klions::ai"
            | "klions::ai::nn"
            | "klions::ai::data"
            | "klions::ai::optim"
            | "klions::ai::optimizers"
            | "klions::ai::loss"
            | "klions::math"
            | "klions::io"
            | "klions::tensor"
    )
}

/// FR-TYP-010: suggest the nearest name within edit distance 2.
pub fn nearest<'a>(word: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for c in candidates {
        let d = edit_distance(word, c);
        if d <= 2 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((c, d));
        }
    }
    best.map(|(c, _)| c)
}

pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}
