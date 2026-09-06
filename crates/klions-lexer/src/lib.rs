//! KLIONS lexical analysis — FR-LEX-001 … FR-LEX-011, EIR-011 … EIR-013.
//!
//! Produces a token stream in which every token carries a byte span, with
//! virtual statement terminators inserted at newlines per FR-LEX-009.

use klions_diagnostics::{codes, Diagnostic, Span};

pub mod token;
pub use token::{Token, TokenKind};

pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    tokens: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
    /// Leading trivia (comments) collected for the next token — FR-PAR-010.
    pending_trivia: Vec<(Span, TriviaKind)>,
    trivia: Vec<(Span, TriviaKind)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TriviaKind {
    LineComment,
    BlockComment,
}

/// EIR-013: skip a UTF-8 byte-order mark.
pub fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{feff}').unwrap_or(s)
}

/// EIR-011: validate UTF-8 and report the first bad byte offset.
pub fn check_utf8(bytes: &[u8]) -> Result<&str, Diagnostic> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(e) => {
            let at = e.valid_up_to();
            Err(Diagnostic::error(
                codes::INVALID_UTF8,
                Span::new(at, at + 1),
                format!("source file is not valid UTF-8; first invalid byte at offset {}", at),
            )
            .with_help("re-save the file with UTF-8 encoding"))
        }
    }
}

pub struct LexResult {
    pub tokens: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn lex(src: &str) -> LexResult {
    let mut lx = Lexer::new(src);
    lx.run();
    LexResult { tokens: lx.tokens, diagnostics: lx.diagnostics }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            tokens: Vec::new(),
            diagnostics: Vec::new(),
            pending_trivia: Vec::new(),
            trivia: Vec::new(),
        }
    }

    fn peek(&self) -> u8 {
        *self.bytes.get(self.pos).unwrap_or(&0)
    }
    fn peek_at(&self, n: usize) -> u8 {
        *self.bytes.get(self.pos + n).unwrap_or(&0)
    }
    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }
    fn bump(&mut self) -> u8 {
        let b = self.peek();
        // Advance by one full UTF-8 scalar so spans never split a character.
        if b < 0x80 {
            self.pos += 1;
        } else {
            let len = utf8_len(b);
            self.pos = (self.pos + len).min(self.bytes.len());
        }
        b
    }

    fn push(&mut self, kind: TokenKind, start: usize) {
        let span = Span::new(start, self.pos);
        let leading = std::mem::take(&mut self.pending_trivia);
        self.tokens.push(Token { kind, span, leading_trivia: leading });
    }

    fn err(&mut self, code: &'static str, span: Span, msg: impl Into<String>) -> usize {
        self.diagnostics.push(Diagnostic::error(code, span, msg));
        self.diagnostics.len() - 1
    }

    pub fn run(&mut self) {
        loop {
            self.skip_trivia_and_newlines();
            if self.at_end() {
                break;
            }
            let start = self.pos;
            let b = self.peek();
            match b {
                b'0'..=b'9' => self.lex_number(start),
                b'"' => self.lex_string(start),
                c if is_ident_start(c) => self.lex_ident(start),
                c if c >= 0x80 => self.lex_non_ascii(start),
                _ => self.lex_symbol(start),
            }
        }
        // Final virtual terminator so the last statement closes.
        if self.can_end_statement() {
            let at = self.bytes.len();
            self.tokens.push(Token {
                kind: TokenKind::VirtualNewline,
                span: Span::new(at, at),
                leading_trivia: Vec::new(),
            });
        }
        let at = self.bytes.len();
        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span::new(at, at),
            leading_trivia: std::mem::take(&mut self.pending_trivia),
        });
    }

    /// Consume whitespace and comments, inserting virtual terminators
    /// at newlines where FR-LEX-009 requires them.
    fn skip_trivia_and_newlines(&mut self) {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\r' if !self.at_end() => {
                    self.pos += 1;
                }
                b'\n' if !self.at_end() => {
                    let nl = self.pos;
                    self.pos += 1;
                    self.maybe_insert_terminator(nl);
                }
                b'/' if self.peek_at(1) == b'/' => {
                    let start = self.pos;
                    while !self.at_end() && self.peek() != b'\n' {
                        self.pos += 1;
                    }
                    let sp = Span::new(start, self.pos);
                    self.pending_trivia.push((sp, TriviaKind::LineComment));
                    self.trivia.push((sp, TriviaKind::LineComment));
                }
                b'/' if self.peek_at(1) == b'*' => {
                    self.lex_block_comment();
                }
                _ => return,
            }
            if self.at_end() {
                return;
            }
        }
    }

    /// FR-LEX-008: nestable block comments.
    fn lex_block_comment(&mut self) {
        let start = self.pos;
        self.pos += 2;
        let mut depth = 1usize;
        while !self.at_end() && depth > 0 {
            if self.peek() == b'/' && self.peek_at(1) == b'*' {
                depth += 1;
                self.pos += 2;
            } else if self.peek() == b'*' && self.peek_at(1) == b'/' {
                depth -= 1;
                self.pos += 2;
            } else {
                self.bump();
            }
        }
        if depth > 0 {
            // FR-LEX-010: point at the opening delimiter, not at EOF.
            self.err(
                codes::UNTERMINATED_BLOCK_COMMENT,
                Span::new(start, start + 2),
                "unterminated block comment",
            );
            self.diagnostics.last_mut().unwrap().help =
                Some("add a closing `*/`; block comments nest, so each `/*` needs its own".into());
        }
        let sp = Span::new(start, self.pos);
        self.pending_trivia.push((sp, TriviaKind::BlockComment));
        self.trivia.push((sp, TriviaKind::BlockComment));
    }

    /// FR-LEX-009: a newline terminates a statement when the previous token
    /// can end one and the next token cannot continue an expression.
    fn maybe_insert_terminator(&mut self, at: usize) {
        if !self.can_end_statement() {
            return;
        }
        // Look ahead past further whitespace/comments to the next real token.
        let save = self.pos;
        let next = self.peek_next_significant();
        self.pos = save;
        if let Some(c) = next {
            if continues_expression(c) {
                return;
            }
        }
        self.tokens.push(Token {
            kind: TokenKind::VirtualNewline,
            span: Span::new(at, at + 1),
            leading_trivia: Vec::new(),
        });
    }

    fn peek_next_significant(&self) -> Option<u8> {
        let mut i = self.pos;
        loop {
            let b = *self.bytes.get(i)?;
            match b {
                b' ' | b'\t' | b'\r' | b'\n' => i += 1,
                b'/' if self.bytes.get(i + 1) == Some(&b'/') => {
                    while i < self.bytes.len() && self.bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if self.bytes.get(i + 1) == Some(&b'*') => {
                    let mut depth = 1;
                    i += 2;
                    while i < self.bytes.len() && depth > 0 {
                        if self.bytes[i] == b'/' && self.bytes.get(i + 1) == Some(&b'*') {
                            depth += 1;
                            i += 2;
                        } else if self.bytes[i] == b'*' && self.bytes.get(i + 1) == Some(&b'/') {
                            depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                }
                _ => return Some(b),
            }
        }
    }

    fn can_end_statement(&self) -> bool {
        match self.tokens.last().map(|t| &t.kind) {
            Some(k) => k.can_end_statement(),
            None => false,
        }
    }

    // ---------- numbers ----------

    fn lex_number(&mut self, start: usize) {
        // FR-LEX-005: 0x / 0b prefixes with `_` separators.
        if self.peek() == b'0' && matches!(self.peek_at(1), b'x' | b'X' | b'b' | b'B') {
            let radix_char = self.peek_at(1);
            let radix = if radix_char == b'x' || radix_char == b'X' { 16 } else { 2 };
            self.pos += 2;
            let digits_start = self.pos;
            let mut digits = String::new();
            while !self.at_end() {
                let c = self.peek();
                if c == b'_' {
                    self.pos += 1;
                } else if (radix == 16 && c.is_ascii_hexdigit())
                    || (radix == 2 && (c == b'0' || c == b'1'))
                {
                    digits.push(c as char);
                    self.pos += 1;
                } else if c.is_ascii_alphanumeric() {
                    // e.g. `0b2` or `0xg` — consume so we report once.
                    digits.push(c as char);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if digits.is_empty() {
                let sp = Span::new(start, self.pos.max(digits_start));
                self.err(codes::MALFORMED_NUMBER, sp, "numeric literal has no digits");
                self.push(TokenKind::Int(0), start);
                return;
            }
            match i64::from_str_radix(&digits, radix) {
                Ok(v) => self.push(TokenKind::Int(v), start),
                Err(_) => {
                    let sp = Span::new(start, self.pos);
                    self.err(
                        codes::MALFORMED_NUMBER,
                        sp,
                        format!(
                            "invalid base-{} integer literal `{}`",
                            radix,
                            &self.src[start..self.pos]
                        ),
                    );
                    self.push(TokenKind::Int(0), start);
                }
            }
            return;
        }

        let mut is_float = false;
        while self.peek().is_ascii_digit() || self.peek() == b'_' {
            self.pos += 1;
        }
        // A `.` starts a fraction only if followed by a digit — `1..5` is a range,
        // and `x.foo` is field access.
        if self.peek() == b'.' && self.peek_at(1).is_ascii_digit() {
            is_float = true;
            self.pos += 1;
            while self.peek().is_ascii_digit() || self.peek() == b'_' {
                self.pos += 1;
            }
        }
        // FR-LEX-006: scientific notation.
        if matches!(self.peek(), b'e' | b'E') {
            let mut ahead = 1;
            if matches!(self.peek_at(1), b'+' | b'-') {
                ahead = 2;
            }
            if self.peek_at(ahead).is_ascii_digit() {
                is_float = true;
                self.pos += ahead;
                while self.peek().is_ascii_digit() || self.peek() == b'_' {
                    self.pos += 1;
                }
            }
        }
        let text: String = self.src[start..self.pos].chars().filter(|c| *c != '_').collect();
        if is_float {
            match text.parse::<f64>() {
                Ok(v) => self.push(TokenKind::Float(v), start),
                Err(_) => {
                    let sp = Span::new(start, self.pos);
                    self.err(codes::MALFORMED_NUMBER, sp, format!("invalid float literal `{}`", text));
                    self.push(TokenKind::Float(0.0), start);
                }
            }
        } else {
            match text.parse::<i64>() {
                Ok(v) => self.push(TokenKind::Int(v), start),
                Err(_) => {
                    let sp = Span::new(start, self.pos);
                    self.err(
                        codes::MALFORMED_NUMBER,
                        sp,
                        format!("integer literal `{}` does not fit in i64", text),
                    );
                    self.push(TokenKind::Int(0), start);
                }
            }
        }
    }

    // ---------- strings ----------

    fn lex_string(&mut self, start: usize) {
        self.pos += 1; // opening quote
        let mut value = String::new();
        loop {
            if self.at_end() {
                // FR-LEX-010: point at the opening quote.
                self.err(
                    codes::UNTERMINATED_STRING,
                    Span::new(start, start + 1),
                    "unterminated string literal",
                );
                self.diagnostics.last_mut().unwrap().help =
                    Some("add a closing `\"`; strings may not span lines".into());
                self.push(TokenKind::Str(value), start);
                return;
            }
            match self.peek() {
                b'"' => {
                    self.pos += 1;
                    self.push(TokenKind::Str(value), start);
                    return;
                }
                b'\n' => {
                    self.err(
                        codes::UNTERMINATED_STRING,
                        Span::new(start, start + 1),
                        "unterminated string literal",
                    );
                    self.diagnostics.last_mut().unwrap().help =
                        Some("use `\\n` for a newline inside a string".into());
                    self.push(TokenKind::Str(value), start);
                    return;
                }
                b'\\' => {
                    let esc_start = self.pos;
                    self.pos += 1;
                    let e = self.peek();
                    self.pos += 1;
                    match e {
                        b'n' => value.push('\n'),
                        b't' => value.push('\t'),
                        b'r' => value.push('\r'),
                        b'\\' => value.push('\\'),
                        b'"' => value.push('"'),
                        b'0' => value.push('\0'),
                        b'u' => self.lex_unicode_escape(esc_start, &mut value),
                        _ => {
                            // FR-LEX-007: unrecognized escape is an error.
                            let sp = Span::new(esc_start, self.pos);
                            self.err(
                                codes::UNKNOWN_ESCAPE,
                                sp,
                                format!("unknown escape sequence `\\{}`", e as char),
                            );
                            self.diagnostics.last_mut().unwrap().help = Some(
                                "valid escapes are \\n \\t \\r \\\\ \\\" \\0 and \\u{XXXX}; write `\\\\` for a literal backslash"
                                    .into(),
                            );
                            value.push(e as char);
                        }
                    }
                }
                _ => {
                    let c_start = self.pos;
                    self.bump();
                    value.push_str(&self.src[c_start..self.pos]);
                }
            }
        }
    }

    fn lex_unicode_escape(&mut self, esc_start: usize, value: &mut String) {
        if self.peek() != b'{' {
            let sp = Span::new(esc_start, self.pos);
            self.err(codes::INVALID_UNICODE_ESCAPE, sp, "expected `{` after `\\u`");
            self.diagnostics.last_mut().unwrap().help = Some("write it as `\\u{1F600}`".into());
            return;
        }
        self.pos += 1;
        let digits_start = self.pos;
        while !self.at_end() && self.peek().is_ascii_hexdigit() {
            self.pos += 1;
        }
        let digits = &self.src[digits_start..self.pos];
        if self.peek() != b'}' {
            let sp = Span::new(esc_start, self.pos);
            self.err(codes::INVALID_UNICODE_ESCAPE, sp, "unterminated `\\u{...}` escape");
            return;
        }
        self.pos += 1;
        match u32::from_str_radix(digits, 16).ok().and_then(char::from_u32) {
            Some(c) => value.push(c),
            None => {
                let sp = Span::new(esc_start, self.pos);
                self.err(
                    codes::INVALID_UNICODE_ESCAPE,
                    sp,
                    format!("`{}` is not a valid Unicode scalar value", digits),
                );
            }
        }
    }

    // ---------- identifiers ----------

    fn lex_ident(&mut self, start: usize) {
        while !self.at_end() && is_ident_continue(self.peek()) {
            self.pos += 1;
        }
        let text = &self.src[start..self.pos];
        let kind = TokenKind::from_word(text);
        if let TokenKind::Reserved(word) = &kind {
            // FR-LEX-003: reserved for a future release.
            let word = *word;
            let sp = Span::new(start, self.pos);
            self.err(
                codes::RESERVED_KEYWORD,
                sp,
                format!("`{}` is reserved for a future release of KLIONS", word),
            );
            let release = reserved_release(word);
            self.diagnostics.last_mut().unwrap().help =
                Some(format!("`{}` is scheduled for v{}; choose another name for now", word, release));
        }
        self.push(kind, start);
    }

    /// FR-LEX-011: non-ASCII identifiers are rejected with an explanation.
    fn lex_non_ascii(&mut self, start: usize) {
        while !self.at_end() && self.peek() >= 0x80 {
            self.bump();
        }
        let sp = Span::new(start, self.pos);
        let text = &self.src[start..self.pos];
        let is_identish = text.chars().all(|c| c.is_alphanumeric() || c == '_');
        if is_identish {
            self.err(
                codes::NON_ASCII_IDENTIFIER,
                sp,
                format!("non-ASCII identifier `{}` is not supported in v0.1.0", text),
            );
            self.diagnostics.last_mut().unwrap().help = Some(
                "identifiers must match [A-Za-z_][A-Za-z0-9_]*; non-ASCII identifiers are planned for a later release"
                    .into(),
            );
        } else {
            self.err(
                codes::UNEXPECTED_CHARACTER,
                sp,
                format!("unexpected character `{}` in source", text),
            );
        }
        self.push(TokenKind::Ident(text.to_string()), start);
    }

    // ---------- symbols ----------

    fn lex_symbol(&mut self, start: usize) {
        use TokenKind::*;
        let b = self.bump();
        let kind = match b {
            b'+' => {
                if self.peek() == b'=' { self.pos += 1; PlusEq } else { Plus }
            }
            b'-' => match self.peek() {
                b'=' => { self.pos += 1; MinusEq }
                b'>' => { self.pos += 1; Arrow }
                _ => Minus,
            },
            b'*' => {
                if self.peek() == b'=' { self.pos += 1; StarEq } else { Star }
            }
            b'/' => {
                if self.peek() == b'=' { self.pos += 1; SlashEq } else { Slash }
            }
            b'%' => Percent,
            b'@' => At,
            b'=' => {
                if self.peek() == b'=' { self.pos += 1; EqEq } else { Eq }
            }
            b'!' => {
                if self.peek() == b'=' { self.pos += 1; BangEq } else { Bang }
            }
            b'<' => {
                if self.peek() == b'=' { self.pos += 1; Le } else { Lt }
            }
            b'>' => {
                if self.peek() == b'=' { self.pos += 1; Ge } else { Gt }
            }
            b'&' => {
                if self.peek() == b'&' { self.pos += 1; AndAnd } else { Amp }
            }
            b'|' => {
                if self.peek() == b'|' { self.pos += 1; OrOr } else { Pipe }
            }
            b'?' => Question,
            b'.' => {
                if self.peek() == b'.' { self.pos += 1; DotDot } else { Dot }
            }
            b',' => Comma,
            b';' => Semi,
            b':' => {
                if self.peek() == b':' { self.pos += 1; ColonColon } else { Colon }
            }
            b'(' => LParen,
            b')' => RParen,
            b'{' => LBrace,
            b'}' => RBrace,
            b'[' => LBracket,
            b']' => RBracket,
            b'_' => Underscore,
            other => {
                let sp = Span::new(start, self.pos);
                self.err(
                    codes::UNEXPECTED_CHARACTER,
                    sp,
                    format!("unexpected character `{}` in source", other as char),
                );
                Unknown
            }
        };
        self.push(kind, start);
    }
}

fn reserved_release(word: &str) -> &'static str {
    match word {
        "struct" | "enum" | "trait" | "impl" | "match" | "pub" => "0.2.0",
        "device" => "0.3.0",
        "async" | "await" | "spawn" => "post-1.0",
        _ => "0.2.0",
    }
}

pub fn is_ident_start(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphabetic()
}
pub fn is_ident_continue(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric()
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// A token starting with one of these bytes continues the previous expression,
/// so no virtual terminator is inserted before it (FR-LEX-009).
fn continues_expression(c: u8) -> bool {
    matches!(
        c,
        b'+' | b'*' | b'/' | b'%' | b'@' | b'.' | b',' | b'?' | b'=' | b'<' | b'>' | b'&' | b'|' | b')' | b']' | b':'
    ) || c == b'-'
}
