//! `klions repl` — an interactive session.
//!
//! Each entry is wrapped into a whole program and re-run, so the REPL uses the
//! same front end and the same interpreter as `klions run`. Declarations
//! accumulate; the last expression is printed.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use klions_diagnostics::render::{self, ColorMode};
use klions_diagnostics::SourceMap;

pub fn run(seed: u64) -> ExitCode {
    let color = ColorMode::detect(
        klions_platform::stdout_is_tty(),
        klions_platform::no_color_env(),
        false,
    );
    println!("klions {} — interactive session", env!("CARGO_PKG_VERSION"));
    println!("type :help for commands, :quit to leave");

    let stdin = std::io::stdin();
    let mut items: Vec<String> = Vec::new();
    let mut history: Vec<String> = Vec::new();
    let mut buffer = String::new();

    loop {
        let prompt = if buffer.is_empty() { "klions> " } else { "  ... > " };
        print!("{}", prompt);
        let _ = std::io::stdout().flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("klions: {}", e);
                break;
            }
        }
        let trimmed = line.trim_end();

        if buffer.is_empty() {
            match trimmed.trim() {
                ":quit" | ":q" | ":exit" => break,
                ":help" | ":h" => {
                    print_help();
                    continue;
                }
                ":clear" => {
                    items.clear();
                    history.clear();
                    println!("session cleared");
                    continue;
                }
                ":list" => {
                    for (i, h) in history.iter().enumerate() {
                        println!("{:>3}  {}", i + 1, h);
                    }
                    continue;
                }
                "" => continue,
                _ => {}
            }
        }

        buffer.push_str(trimmed);
        buffer.push('\n');

        // Wait for balanced delimiters before evaluating.
        if !is_balanced(&buffer) {
            continue;
        }
        let entry = std::mem::take(&mut buffer);
        let entry = entry.trim().to_string();
        if entry.is_empty() {
            continue;
        }
        history.push(entry.clone());

        // A declaration joins the accumulated program; anything else is a
        // statement to run inside `main`.
        let is_item = entry.starts_with("fn ")
            || entry.starts_with("model ")
            || entry.starts_with("const ")
            || entry.starts_with("import ");

        let program = if is_item {
            let mut candidate = items.clone();
            candidate.push(entry.clone());
            build_program(&candidate, "")
        } else {
            build_program(&items, &entry)
        };

        let analysis = klions_frontend::analyze(&program);
        let sm = SourceMap::new("<repl>", &program);
        let mut had_error = false;
        for d in &analysis.diagnostics {
            if d.severity == klions_diagnostics::Severity::Error {
                had_error = true;
                eprint!("{}", render::render(d, &sm, color));
            }
        }
        if had_error {
            continue;
        }
        if is_item {
            items.push(entry);
            continue;
        }

        let prog = analysis.program;
        let outcome = klions_runtime::on_interpreter_stack(move || {
            let mut interp = klions_runtime::Interpreter::new(prog, seed, PathBuf::from("."));
            interp.run()
        });
        match outcome {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                let d = e.to_diagnostic();
                eprint!("{}", render::render(&d, &sm, color));
            }
            Err(_) => eprintln!("klions: the interpreter thread panicked"),
        }
    }

    println!();
    ExitCode::SUCCESS
}

/// Wrap accumulated declarations and one statement into a whole program.
/// A bare expression is printed; a statement is executed as written.
fn build_program(items: &[String], statement: &str) -> String {
    let mut s = String::new();
    for it in items {
        s.push_str(it);
        s.push_str("\n\n");
    }
    s.push_str("fn main() {\n");
    if !statement.is_empty() {
        if looks_like_expression(statement) {
            s.push_str(&format!("    println(\"{{}}\", {})\n", statement.trim()));
        } else {
            for line in statement.lines() {
                s.push_str("    ");
                s.push_str(line);
                s.push('\n');
            }
        }
    }
    s.push_str("}\n");
    s
}

fn looks_like_expression(s: &str) -> bool {
    let t = s.trim();
    if t.contains('\n') {
        return false;
    }
    let starters = [
        "let ", "if ", "while ", "for ", "return", "train ", "no_grad", "break", "continue",
        "print(", "println(", "assert(",
    ];
    if starters.iter().any(|k| t.starts_with(k)) {
        return false;
    }
    // `x = 1` is an assignment, but `x == 1` is an expression.
    if let Some(pos) = t.find('=') {
        let after = t.as_bytes().get(pos + 1).copied();
        let before = if pos == 0 { None } else { t.as_bytes().get(pos - 1).copied() };
        let comparison = after == Some(b'=')
            || matches!(before, Some(b'=') | Some(b'!') | Some(b'<') | Some(b'>'));
        if !comparison {
            return false;
        }
    }
    true
}

/// Balanced brackets and closed strings — the multi-line continuation rule.
fn is_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '/' if chars.peek() == Some(&'/') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
            }
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0 && !in_string
}

fn print_help() {
    println!(
        r#"REPL commands
    :help    this message
    :list    show the entries in this session
    :clear   forget every declaration
    :quit    leave

Declarations (fn, model, const, import) accumulate across entries.
Anything else runs immediately; a bare expression is printed.
An unclosed brace, bracket, or string continues onto the next line."#
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_detection_handles_nesting() {
        assert!(is_balanced("let x = 1"));
        assert!(!is_balanced("fn f() {"));
        assert!(is_balanced("fn f() { }"));
        assert!(!is_balanced("let s = \"unclosed"));
        assert!(is_balanced("let s = \"a { b\""));
    }

    #[test]
    fn braces_inside_strings_do_not_count() {
        assert!(is_balanced("println(\"{}\", x)"));
    }

    #[test]
    fn expressions_are_distinguished_from_statements() {
        assert!(looks_like_expression("1 + 2"));
        assert!(looks_like_expression("x == 3"));
        assert!(!looks_like_expression("let x = 1"));
        assert!(!looks_like_expression("x = 1"));
        assert!(!looks_like_expression("println(\"hi\")"));
    }

    #[test]
    fn a_bare_expression_is_wrapped_in_a_print() {
        let p = build_program(&[], "1 + 2");
        assert!(p.contains("println(\"{}\", 1 + 2)"), "got:\n{}", p);
    }

    #[test]
    fn declarations_are_hoisted_above_main() {
        let p = build_program(&["fn f() -> i64 { return 1 }".to_string()], "f()");
        assert!(p.find("fn f()").unwrap() < p.find("fn main()").unwrap());
    }
}
