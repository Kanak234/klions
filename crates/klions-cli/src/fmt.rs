//! `klions fmt` — a canonical formatter.
//!
//! Works from the token stream rather than the AST so that comments survive
//! and so that a file with syntax errors can still be tidied. The output is
//! idempotent: formatting twice produces the same bytes as formatting once.

use std::path::PathBuf;
use std::process::ExitCode;

use klions_lexer::{lex, TokenKind};

const INDENT: &str = "    ";

pub fn run(files: &[PathBuf], check_only: bool) -> ExitCode {
    if files.is_empty() {
        eprintln!("klions fmt: no input files");
        return ExitCode::from(2);
    }
    let mut changed = Vec::new();
    let mut failed = false;

    for path in files {
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("klions fmt: {}: {}", path.display(), e);
                failed = true;
                continue;
            }
        };
        let formatted = format_source(&source);
        if formatted == source {
            continue;
        }
        changed.push(path.clone());
        if !check_only {
            if let Err(e) = std::fs::write(path, &formatted) {
                eprintln!("klions fmt: {}: {}", path.display(), e);
                failed = true;
            }
        }
    }

    if check_only && !changed.is_empty() {
        for p in &changed {
            eprintln!("would reformat {}", p.display());
        }
        return ExitCode::from(1);
    }
    if !check_only && !changed.is_empty() {
        for p in &changed {
            eprintln!("formatted {}", p.display());
        }
    }
    if failed {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

pub fn format_source(source: &str) -> String {
    let src = klions_lexer::strip_bom(source);
    let tokens = lex(src).tokens;
    let mut out = String::with_capacity(src.len());
    let mut indent = 0usize;
    let mut at_line_start = true;
    let mut prev: Option<TokenKind> = None;
    let mut blank_pending = false;

    for (i, tok) in tokens.iter().enumerate() {
        // Comments attached to this token come first, on their own lines.
        if !tok.leading_trivia.is_empty() {
            for (span, _) in &tok.leading_trivia {
                let text = src[span.start..span.end].trim_end();
                if !at_line_start {
                    out.push('\n');
                }
                push_indent(&mut out, indent);
                out.push_str(text);
                out.push('\n');
            }
            // Every comment ended with a newline, so the next token starts one.
            at_line_start = true;
        }

        match &tok.kind {
            TokenKind::Eof => break,
            TokenKind::VirtualNewline | TokenKind::Semi => {
                if !at_line_start {
                    out.push('\n');
                    at_line_start = true;
                }
                // Collapse runs of blank lines to at most one.
                if matches!(tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::VirtualNewline)) {
                    blank_pending = true;
                }
                continue;
            }
            TokenKind::RBrace => {
                indent = indent.saturating_sub(1);
                // A closing brace always starts its own line at the outer level.
                if !at_line_start {
                    out.push('\n');
                }
                push_indent(&mut out, indent);
                out.push('}');
                at_line_start = false;
                prev = Some(tok.kind.clone());
                continue;
            }
            _ => {}
        }

        if at_line_start {
            if blank_pending && !out.is_empty() {
                out.push('\n');
            }
            blank_pending = false;
            push_indent(&mut out, indent);
            at_line_start = false;
        } else if needs_space(prev.as_ref(), &tok.kind) {
            out.push(' ');
        }

        out.push_str(&render_token(&tok.kind, src, tok.span));

        if matches!(tok.kind, TokenKind::LBrace) {
            indent += 1;
            out.push('\n');
            at_line_start = true;
        }
        prev = Some(tok.kind.clone());
    }

    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n'); // exactly one trailing newline
    out
}

fn push_indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str(INDENT);
    }
}

fn render_token(k: &TokenKind, src: &str, span: klions_diagnostics::Span) -> String {
    match k {
        TokenKind::Int(_) | TokenKind::Float(_) => {
            // Preserve the author's literal form: 0xFF stays 0xFF, 1e-3 stays 1e-3.
            src[span.start..span.end].to_string()
        }
        TokenKind::Str(_) => src[span.start..span.end].to_string(),
        TokenKind::Ident(s) => s.clone(),
        TokenKind::Reserved(s) => s.to_string(),
        other => other.lexeme().to_string(),
    }
}

/// Spacing rules. Binary operators are surrounded; `(`, `[`, `.`, `,` are not.
fn needs_space(prev: Option<&TokenKind>, next: &TokenKind) -> bool {
    use TokenKind::*;
    let Some(prev) = prev else { return false };

    // Never a space before these.
    if matches!(
        next,
        Comma | RParen | RBracket | Dot | ColonColon | LParen | LBracket | Question | Colon
    ) {
        // `(` hugs a callee or a keyword-free name, but follows `if`/`while` with a space.
        if matches!(next, LParen) {
            return matches!(prev, If | While | For | In | Return | Comma | Eq | ColonColon)
                || is_binary_op(prev);
        }
        if matches!(next, LBracket) {
            return matches!(prev, Comma | Eq | Return) || is_binary_op(prev);
        }
        return false;
    }
    // Never a space after these.
    if matches!(prev, LParen | LBracket | Dot | ColonColon | Bang) {
        return false;
    }
    // `-` as a unary sign hugs its operand.
    if matches!(prev, Minus) && !can_precede_binary(prev) {
        return false;
    }
    if matches!(prev, Comma | Colon) {
        return true;
    }
    true
}

fn is_binary_op(k: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(
        k,
        Plus | Minus | Star | Slash | Percent | At | Eq | EqEq | BangEq | Lt | Le | Gt | Ge
            | AndAnd | OrOr | PlusEq | MinusEq | StarEq | SlashEq | Arrow
    )
}

fn can_precede_binary(_k: &TokenKind) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indentation_is_normalized() {
        let src = "fn main(){\nlet x=1\nprintln(\"{}\",x)\n}";
        let out = format_source(src);
        assert!(out.contains("\n    let x = 1"), "got:\n{}", out);
        assert!(out.ends_with("}\n"));
    }

    #[test]
    fn formatting_is_idempotent() {
        let src = "fn main() {\n    let x = 1 + 2\n    println(\"{}\", x)\n}\n";
        let once = format_source(src);
        let twice = format_source(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn comments_survive() {
        let src = "fn main() {\n// a note\nlet x = 1\n}";
        let out = format_source(src);
        assert!(out.contains("// a note"), "got:\n{}", out);
    }

    #[test]
    fn nested_blocks_indent_progressively() {
        let src = "fn main() {\nif true {\nlet x = 1\n}\n}";
        let out = format_source(src);
        assert!(out.contains("\n        let x = 1"), "got:\n{}", out);
    }

    #[test]
    fn numeric_literal_form_is_preserved() {
        let out = format_source("fn main() { let a = 0xFF\nlet b = 1e-3 }");
        assert!(out.contains("0xFF"), "got:\n{}", out);
        assert!(out.contains("1e-3"), "got:\n{}", out);
    }

    #[test]
    fn blank_line_runs_collapse_to_one() {
        let out = format_source("fn main() {\nlet a = 1\n\n\n\nlet b = 2\n}");
        assert!(!out.contains("\n\n\n"), "got:\n{}", out);
    }
}
