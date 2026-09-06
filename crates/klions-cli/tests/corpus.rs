//! Integration tests over the `tests/` corpora.
//!
//! Two suites:
//!   * `tests/programs/*.kl` with a matching `.expected` — run it, compare stdout.
//!   * `tests/diagnostics/*.kl` with a matching `.codes` — check it, and assert
//!     that every listed diagnostic code was produced.
//!
//! Both walk the directory, so adding a case is adding a file.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/klions-cli; the corpora live two levels up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn cases(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", dir.display(), e))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == ext).unwrap_or(false))
        .collect();
    out.sort();
    out
}

/// Run a source file through the front end and the interpreter, capturing output.
fn run_program(path: &Path) -> Result<String, String> {
    let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let analysis = klions_frontend::analyze(&source);
    let errors: Vec<String> = analysis
        .errors()
        .map(|d| format!("[{}] {}", d.code, d.message))
        .collect();
    if !errors.is_empty() {
        return Err(format!("analysis failed: {:?}", errors));
    }

    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let buf = Buf::new();
    let sink = buf.clone();
    let program = analysis.program;

    let outcome = klions_runtime::on_interpreter_stack(move || {
        let mut interp = klions_runtime::Interpreter::new(program, 42, dir);
        interp.out = Box::new(sink);
        interp.run()
    })
    .map_err(|_| "the interpreter thread panicked".to_string())?;

    outcome.map_err(|e| format!("[{}] {}", e.code, e.message))?;
    Ok(buf.contents())
}

#[derive(Clone)]
struct Buf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl Buf {
    fn new() -> Buf {
        Buf(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
    }
    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
    }
}

impl std::io::Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn golden_programs_produce_their_expected_output() {
    let dir = repo_root().join("tests/programs");
    let files = cases(&dir, "kl");
    assert!(!files.is_empty(), "no program cases found in {}", dir.display());

    let mut failures = Vec::new();
    for path in &files {
        let expected_path = path.with_extension("expected");
        let Ok(expected) = std::fs::read_to_string(&expected_path) else {
            failures.push(format!("{}: missing .expected file", path.display()));
            continue;
        };
        match run_program(path) {
            Ok(actual) => {
                if actual.trim_end() != expected.trim_end() {
                    failures.push(format!(
                        "{}:\n  expected:\n{}\n  actual:\n{}",
                        path.display(),
                        indent(expected.trim_end()),
                        indent(actual.trim_end())
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: {}", path.display(), e)),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

fn indent(s: &str) -> String {
    s.lines().map(|l| format!("    {}", l)).collect::<Vec<_>>().join("\n")
}

/// FR-ERR-003: each case must produce the diagnostic codes it names.
#[test]
fn diagnostic_corpus_produces_the_expected_codes() {
    let dir = repo_root().join("tests/diagnostics");
    let files = cases(&dir, "kl");
    assert!(
        files.len() >= 20,
        "the corpus should hold at least 20 cases, found {}",
        files.len()
    );

    let mut failures = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        let want: Vec<String> = std::fs::read_to_string(path.with_extension("codes"))
            .unwrap_or_default()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        if want.is_empty() {
            failures.push(format!("{}: no .codes file", path.display()));
            continue;
        }

        let analysis = klions_frontend::analyze(&source);
        let got: Vec<&str> = analysis.diagnostics.iter().map(|d| d.code).collect();

        for code in &want {
            if !got.contains(&code.as_str()) {
                failures.push(format!(
                    "{}: expected {} but got {:?}",
                    path.file_name().unwrap().to_string_lossy(),
                    code,
                    analysis
                        .diagnostics
                        .iter()
                        .map(|d| format!("{}: {}", d.code, d.message))
                        .collect::<Vec<_>>()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// FR-ERR-002: every error carries a span pointing into the source, and
/// diagnostics that can suggest a fix do so.
#[test]
fn diagnostics_have_usable_spans_and_guidance() {
    let dir = repo_root().join("tests/diagnostics");
    let mut without_help = Vec::new();

    for path in cases(&dir, "kl") {
        let source = std::fs::read_to_string(&path).unwrap();
        let analysis = klions_frontend::analyze(&source);
        for d in analysis.errors() {
            assert!(
                d.span.end <= source.len(),
                "{}: span {:?} runs past the end of a {}-byte file",
                path.display(),
                d.span,
                source.len()
            );
            assert!(
                !d.message.is_empty(),
                "{}: {} has an empty message",
                path.display(),
                d.code
            );
            if d.help.is_none() && d.notes.is_empty() {
                without_help.push(format!(
                    "{}: {} — {}",
                    path.file_name().unwrap().to_string_lossy(),
                    d.code,
                    d.message
                ));
            }
        }
    }
    // Most diagnostics should offer a next step; a few purely structural ones
    // legitimately cannot.
    assert!(
        without_help.len() <= 4,
        "too many diagnostics without help or notes:\n{}",
        without_help.join("\n")
    );
}

/// The example programs shipped with the toolchain must analyze cleanly.
#[test]
fn shipped_examples_analyze_without_errors() {
    let dir = repo_root().join("examples");
    for path in cases(&dir, "kl") {
        let source = std::fs::read_to_string(&path).unwrap();
        let analysis = klions_frontend::analyze(&source);
        let errors: Vec<String> = analysis
            .errors()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect();
        assert!(
            errors.is_empty(),
            "{} has errors: {:?}",
            path.display(),
            errors
        );
    }
}

/// FR-AI-014: a fixed seed reproduces a run exactly.
#[test]
fn identical_seeds_reproduce_identical_runs() {
    let dir = repo_root().join("tests/programs");
    let path = dir.join("seeded.kl");
    std::fs::write(
        &path,
        "fn main() {\n    let a = randn(4, 4)\n    println(\"{:.6}\", a.sum())\n    let b = rand(3, 3)\n    println(\"{:.6}\", b.mean())\n}\n",
    )
    .unwrap();

    let first = run_program(&path).unwrap();
    let second = run_program(&path).unwrap();
    assert_eq!(first, second, "the same seed produced different output");
    assert!(!first.trim().is_empty());
    std::fs::remove_file(&path).ok();
}
