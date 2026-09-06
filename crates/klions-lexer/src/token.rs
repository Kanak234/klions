//! Token definitions and the keyword table (FR-LEX-002, FR-LEX-003, FR-LEX-004).

use crate::TriviaKind;
use klions_diagnostics::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // literals
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),

    // FR-LEX-002 keywords
    Let,
    Mut,
    Fn,
    Const,
    Return,
    If,
    Else,
    While,
    For,
    In,
    Model,
    Train,
    Using,
    Import,
    True,
    False,
    Null,
    Ok,
    Err,
    As,
    NoGrad,
    Break,
    Continue,

    /// FR-LEX-003: recognized, meaning deferred.
    Reserved(&'static str),

    // punctuation & operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    At,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Amp,
    Pipe,
    Question,
    Dot,
    DotDot,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Arrow,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Underscore,

    /// FR-LEX-009 virtual statement terminator.
    VirtualNewline,
    Eof,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub leading_trivia: Vec<(Span, TriviaKind)>,
}

impl Token {
    pub fn is(&self, k: &TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(k)
    }
    pub fn ident(&self) -> Option<&str> {
        match &self.kind {
            TokenKind::Ident(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl TokenKind {
    pub fn from_word(w: &str) -> TokenKind {
        use TokenKind::*;
        match w {
            "let" => Let,
            "mut" => Mut,
            "fn" => Fn,
            "const" => Const,
            "return" => Return,
            "if" => If,
            "else" => Else,
            "while" => While,
            "for" => For,
            "in" => In,
            "model" => Model,
            "train" => Train,
            "using" => Using,
            "import" => Import,
            "true" => True,
            "false" => False,
            "null" => Null,
            "Ok" => Ok,
            "Err" => Err,
            "as" => As,
            "no_grad" => NoGrad,
            "break" => Break,
            "continue" => Continue,
            // FR-LEX-003 reserved words
            "struct" => Reserved("struct"),
            "enum" => Reserved("enum"),
            "trait" => Reserved("trait"),
            "impl" => Reserved("impl"),
            "match" => Reserved("match"),
            "pub" => Reserved("pub"),
            "async" => Reserved("async"),
            "await" => Reserved("await"),
            "unsafe" => Reserved("unsafe"),
            "device" => Reserved("device"),
            "spawn" => Reserved("spawn"),
            // A lone `_` is the wildcard token, not an identifier: it reaches
            // here rather than `lex_symbol` because `_` starts identifiers.
            "_" => Underscore,
            other => Ident(other.to_string()),
        }
    }

    /// FR-LEX-009: may this token end a statement?
    pub fn can_end_statement(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Int(_)
                | Float(_)
                | Str(_)
                | Ident(_)
                | True
                | False
                | Null
                | Return
                | Break
                | Continue
                | RParen
                | RBracket
                | RBrace
                | Question
                | Underscore
                | Reserved(_)
        )
    }

    /// Human-readable name used in "expected X, found Y" diagnostics.
    pub fn describe(&self) -> String {
        use TokenKind::*;
        match self {
            Int(v) => format!("integer `{}`", v),
            Float(v) => format!("float `{}`", v),
            Str(_) => "string literal".to_string(),
            Ident(s) => format!("identifier `{}`", s),
            Reserved(s) => format!("reserved word `{}`", s),
            VirtualNewline => "end of line".to_string(),
            Eof => "end of file".to_string(),
            Unknown => "unknown token".to_string(),
            other => format!("`{}`", other.lexeme()),
        }
    }

    pub fn lexeme(&self) -> &'static str {
        use TokenKind::*;
        match self {
            Let => "let",
            Mut => "mut",
            Fn => "fn",
            Const => "const",
            Return => "return",
            If => "if",
            Else => "else",
            While => "while",
            For => "for",
            In => "in",
            Model => "model",
            Train => "train",
            Using => "using",
            Import => "import",
            True => "true",
            False => "false",
            Null => "null",
            Ok => "Ok",
            Err => "Err",
            As => "as",
            NoGrad => "no_grad",
            Break => "break",
            Continue => "continue",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            At => "@",
            Eq => "=",
            PlusEq => "+=",
            MinusEq => "-=",
            StarEq => "*=",
            SlashEq => "/=",
            EqEq => "==",
            BangEq => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            AndAnd => "&&",
            OrOr => "||",
            Bang => "!",
            Amp => "&",
            Pipe => "|",
            Question => "?",
            Dot => ".",
            DotDot => "..",
            Comma => ",",
            Semi => ";",
            Colon => ":",
            ColonColon => "::",
            Arrow => "->",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Underscore => "_",
            _ => "?",
        }
    }

    /// FR-LEX-004 primitive type names.
    pub fn is_primitive_type_name(name: &str) -> bool {
        matches!(
            name,
            "i32" | "i64" | "f32" | "f64" | "bool" | "string" | "void" | "auto" | "tensor"
        )
    }

    pub fn is_terminator(&self) -> bool {
        matches!(self, TokenKind::Semi | TokenKind::VirtualNewline)
    }
}
