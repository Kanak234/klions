//! The KLIONS front end as a single reusable unit — DC-004, FR-LSP-001.
//!
//! The CLI and the language server both call [`analyze`]. There is no second
//! parser and no second type checker, so a diagnostic in the editor is the
//! same diagnostic the compiler produces, by construction.
//!
//! This crate deliberately does **not** depend on the runtime: analysis must
//! never require the ability to execute.

use std::collections::HashMap;

use klions_diagnostics::{Diagnostic, DiagnosticBag, Severity, SourceMap};

pub use klions_ast as ast;
pub use klions_diagnostics as diagnostics;
pub use klions_lexer as lexer;
pub use klions_parser as parser;
pub use klions_types as types;

pub struct Analysis {
    pub program: ast::Program,
    pub diagnostics: Vec<Diagnostic>,
    pub models: HashMap<String, types::ModelInfo>,
    pub functions: HashMap<String, types::FnInfo>,
    pub tokens: Vec<lexer::Token>,
}

impl Analysis {
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.severity == Severity::Error)
    }
    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }
    pub fn error_count(&self) -> usize {
        self.errors().count()
    }
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }
}

/// Lex, parse, and type-check. Always returns an `Analysis`: a program with
/// errors still yields a partial AST so the editor can offer completions.
pub fn analyze(source: &str) -> Analysis {
    let src = lexer::strip_bom(source);
    let mut bag = DiagnosticBag::new();

    let lexed = lexer::lex(src);
    for d in lexed.diagnostics {
        bag.push(d);
    }

    let parsed = parser::parse(lexed.tokens.clone());
    for d in parsed.diagnostics {
        bag.push(d);
    }

    // FR-TYP-016: skip type checking only if parsing produced no usable tree.
    let checked = types::check(&parsed.program);
    for d in checked.diagnostics {
        bag.push(d);
    }

    bag.sort();
    Analysis {
        program: parsed.program,
        diagnostics: bag.into_vec(),
        models: checked.models,
        functions: checked.functions,
        tokens: lexed.tokens,
    }
}

/// Convenience for the CLI: analyze and render every diagnostic.
pub fn analyze_and_render(
    name: &str,
    source: &str,
    color: diagnostics::render::ColorMode,
) -> (Analysis, String) {
    let a = analyze(source);
    let sm = SourceMap::new(name, source);
    let mut out = String::new();
    for d in &a.diagnostics {
        out.push_str(&diagnostics::render::render(d, &sm, color));
        out.push('\n');
    }
    (a, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_valid_program_analyzes_cleanly() {
        let a = analyze("fn main() { println(\"hello\") }");
        assert!(!a.has_errors(), "{:?}", a.diagnostics);
        assert!(a.functions.contains_key("main"));
    }

    #[test]
    fn models_are_visible_to_callers() {
        let a = analyze(
            "model Net {
                 h = Dense(inputs=4, outputs=8, activation=\"relu\")
                 o = Dense(inputs=8, outputs=1)
             }
             fn main() { }",
        );
        assert!(!a.has_errors(), "{:?}", a.diagnostics);
        let net = &a.models["Net"];
        assert_eq!(net.input_width, Some(4));
        assert_eq!(net.output_width, Some(1));
    }

    #[test]
    fn diagnostics_arrive_sorted_by_position() {
        let a = analyze(
            "fn main() {
                 let _a = first_undefined
                 let _b = second_undefined
             }",
        );
        let spans: Vec<usize> = a.diagnostics.iter().map(|d| d.span.start).collect();
        let mut sorted = spans.clone();
        sorted.sort();
        assert_eq!(spans, sorted);
    }

    #[test]
    fn a_syntax_error_still_yields_a_partial_tree() {
        let a = analyze("fn main() { let x = }\nfn other() { }");
        assert!(a.has_errors());
        // `other` survived recovery, so the editor can still work with it.
        assert!(a.functions.contains_key("other"));
    }

    #[test]
    fn rendering_produces_a_source_excerpt() {
        let src = "fn main() {\n    let _ = missing_name\n}";
        let (_, text) = analyze_and_render(
            "test.kl",
            src,
            diagnostics::render::ColorMode::Never,
        );
        assert!(text.contains("test.kl:2:"));
        assert!(text.contains("missing_name"));
        assert!(text.contains("^"));
    }
}
