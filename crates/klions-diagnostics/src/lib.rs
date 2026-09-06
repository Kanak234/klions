//! KLIONS diagnostics — spans, severities, stable codes, and rendering.
//!
//! Satisfies FR-ERR-002, FR-ERR-003, FR-ERR-004, FR-ERR-005, FR-ERR-011,
//! EIR-007, EIR-008, EIR-009.
//!
//! This crate has no dependency on any other KLIONS crate, so both the
//! front end and the runtime can report through it (DC-004).

pub mod codes;
pub mod render;

use std::fmt;

/// A byte range into a source buffer (FR-LEX-001).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
    pub const fn empty() -> Self {
        Span { start: 0, end: 0 }
    }
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Severity {
    Hint,
    Warning,
    Error,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Hint => "hint",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
    /// LSP DiagnosticSeverity numbering.
    pub fn lsp_code(&self) -> u8 {
        match self {
            Severity::Error => 1,
            Severity::Warning => 2,
            Severity::Hint => 4,
        }
    }
}

/// A structured diagnostic (FR-ERR-002).
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
    pub span: Span,
    /// Additional `note:` lines.
    pub notes: Vec<String>,
    /// A `help:` line proposing a fix.
    pub help: Option<String>,
    /// Secondary labelled spans, rendered after the primary one.
    pub secondary: Vec<(Span, String)>,
}

impl Diagnostic {
    pub fn error(code: &'static str, span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code,
            message: message.into(),
            span,
            notes: Vec::new(),
            help: None,
            secondary: Vec::new(),
        }
    }
    pub fn warning(code: &'static str, span: Span, message: impl Into<String>) -> Self {
        let mut d = Diagnostic::error(code, span, message);
        d.severity = Severity::Warning;
        d
    }
    pub fn hint(code: &'static str, span: Span, message: impl Into<String>) -> Self {
        let mut d = Diagnostic::error(code, span, message);
        d.severity = Severity::Hint;
        d
    }
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
    pub fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.secondary.push((span, label.into()));
        self
    }
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.severity.as_str(), self.code, self.message)
    }
}

/// Collector with de-duplication and the FR-ERR-011 cap.
#[derive(Clone, Debug, Default)]
pub struct DiagnosticBag {
    items: Vec<Diagnostic>,
    suppressed: usize,
    cap: usize,
}

pub const DEFAULT_DIAGNOSTIC_CAP: usize = 50;

impl DiagnosticBag {
    pub fn new() -> Self {
        DiagnosticBag { items: Vec::new(), suppressed: 0, cap: DEFAULT_DIAGNOSTIC_CAP }
    }
    pub fn with_cap(cap: usize) -> Self {
        DiagnosticBag { items: Vec::new(), suppressed: 0, cap }
    }
    pub fn push(&mut self, d: Diagnostic) {
        // Duplicate suppression: same code at the same span is reported once.
        if self.items.iter().any(|x| x.code == d.code && x.span == d.span) {
            return;
        }
        if self.items.len() >= self.cap {
            self.suppressed += 1;
            return;
        }
        self.items.push(d);
    }
    pub fn extend(&mut self, other: impl IntoIterator<Item = Diagnostic>) {
        for d in other {
            self.push(d);
        }
    }
    pub fn items(&self) -> &[Diagnostic] {
        &self.items
    }
    pub fn into_vec(self) -> Vec<Diagnostic> {
        self.items
    }
    pub fn suppressed(&self) -> usize {
        self.suppressed
    }
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_error())
    }
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.is_error()).count()
    }
    pub fn warning_count(&self) -> usize {
        self.items.iter().filter(|d| d.severity == Severity::Warning).count()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    /// FR-ERR-004: stable ordering by position, then code.
    pub fn sort(&mut self) {
        self.items.sort_by(|a, b| {
            a.span
                .start
                .cmp(&b.span.start)
                .then(a.span.end.cmp(&b.span.end))
                .then(a.code.cmp(b.code))
        });
    }
}

/// Maps byte offsets to 1-based line/column, and yields source lines.
pub struct SourceMap<'a> {
    pub name: String,
    pub text: &'a str,
    line_starts: Vec<usize>,
}

impl<'a> SourceMap<'a> {
    pub fn new(name: impl Into<String>, text: &'a str) -> Self {
        let mut line_starts = vec![0usize];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        SourceMap { name: name.into(), text, line_starts }
    }

    /// 1-based (line, column). Columns count UTF-8 characters, not bytes.
    pub fn position(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line];
        let col = self.text[start..offset].chars().count() + 1;
        (line + 1, col)
    }

    /// 0-based line/character, as LSP wants it (UTF-16 code units).
    pub fn lsp_position(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line];
        let character = self.text[start..offset].chars().map(|c| c.len_utf16()).sum();
        (line, character)
    }

    /// Byte offset of a 0-based LSP position.
    pub fn offset_of_lsp(&self, line: usize, character: usize) -> usize {
        let start = match self.line_starts.get(line) {
            Some(s) => *s,
            None => return self.text.len(),
        };
        let rest = &self.text[start..];
        let mut units = 0usize;
        for (i, c) in rest.char_indices() {
            if units >= character || c == '\n' {
                return start + i;
            }
            units += c.len_utf16();
        }
        self.text.len()
    }

    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    pub fn line_text(&self, line_1based: usize) -> &'a str {
        if line_1based == 0 || line_1based > self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[line_1based - 1];
        let end = self
            .line_starts
            .get(line_1based)
            .copied()
            .unwrap_or(self.text.len());
        self.text[start..end].trim_end_matches(['\n', '\r'])
    }

    pub fn line_span(&self, line_1based: usize) -> Span {
        let start = self.line_starts[line_1based - 1];
        let end = self
            .line_starts
            .get(line_1based)
            .copied()
            .unwrap_or(self.text.len());
        Span::new(start, end)
    }
}
