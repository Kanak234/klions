//! The `klions` command-line driver — EIR-001 … EIR-010.
//!
//! Exit codes (EIR-010):
//! ```text
//!   0    success
//!   1    the program ran and returned a non-zero status
//!   2    compilation failed (any error diagnostic)
//!   101  the program failed at run time
//!   130  interrupted by SIGINT
//! ```

mod fmt;
mod repl;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use klions_diagnostics::render::{self, ColorMode};
use klions_diagnostics::{Severity, SourceMap};

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Options {
    command: Command,
    files: Vec<PathBuf>,
    json: bool,
    seed: u64,
    color: Option<bool>,
    time: bool,
    quiet: bool,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Command {
    Run,
    Check,
    Fmt,
    FmtCheck,
    Repl,
    Version,
    Help,
    Explain,
    Health,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("klions: {}", msg);
            eprintln!("try `klions --help`");
            return ExitCode::from(2);
        }
    };

    match opts.command {
        Command::Version => {
            println!("klions {}", VERSION);
            println!("target: {}", std::env::consts::ARCH);
            ExitCode::SUCCESS
        }
        Command::Help => {
            print_help();
            ExitCode::SUCCESS
        }
        Command::Health => health_check(),
        Command::Explain => {
            let code = opts.files.first().map(|p| p.to_string_lossy().to_string());
            explain(code.as_deref())
        }
        Command::Repl => repl::run(opts.seed),
        Command::Check => check_files(&opts),
        Command::Fmt | Command::FmtCheck => fmt::run(&opts.files, opts.command == Command::FmtCheck),
        Command::Run => run_file(&opts),
    }
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        command: Command::Help,
        files: Vec::new(),
        json: false,
        seed: 42, // FR-RT-010 default
        color: None,
        time: false,
        quiet: false,
    };
    if args.is_empty() {
        return Ok(o);
    }

    let mut i = 0;
    o.command = match args[0].as_str() {
        "run" => {
            i = 1;
            Command::Run
        }
        "check" => {
            i = 1;
            Command::Check
        }
        "fmt" => {
            i = 1;
            Command::Fmt
        }
        "repl" => {
            i = 1;
            Command::Repl
        }
        "explain" => {
            i = 1;
            Command::Explain
        }
        "health" => return Ok(Options { command: Command::Health, ..o }),
        "version" | "--version" | "-V" => return Ok(Options { command: Command::Version, ..o }),
        "help" | "--help" | "-h" => return Ok(Options { command: Command::Help, ..o }),
        other if other.starts_with('-') => {
            return Err(format!("unknown option `{}`", other));
        }
        // `klions program.kl` is shorthand for `klions run program.kl`.
        _ => Command::Run,
    };

    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--json" => o.json = true,
            "--no-color" => o.color = Some(false),
            "--color" => o.color = Some(true),
            "--time" => o.time = true,
            "--quiet" | "-q" => o.quiet = true,
            "--check" if o.command == Command::Fmt => o.command = Command::FmtCheck,
            "--help" | "-h" => {
                o.command = Command::Help;
                return Ok(o);
            }
            "--version" | "-V" => {
                o.command = Command::Version;
                return Ok(o);
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("`--seed` needs a value")?;
                o.seed = v
                    .parse()
                    .map_err(|_| format!("`--seed` expects a non-negative integer, got `{}`", v))?;
            }
            s if s.starts_with("--seed=") => {
                o.seed = s[7..]
                    .parse()
                    .map_err(|_| format!("`--seed` expects a non-negative integer, got `{}`", &s[7..]))?;
            }
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option `{}`", s));
            }
            path => o.files.push(PathBuf::from(path)),
        }
        i += 1;
    }
    Ok(o)
}

fn color_mode(opts: &Options) -> ColorMode {
    match opts.color {
        Some(true) => ColorMode::Always,
        Some(false) => ColorMode::Never,
        None => ColorMode::detect(
            klions_platform::stderr_is_tty(),
            klions_platform::no_color_env(),
            false,
        ),
    }
}

fn read_source(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("no such file: {}", path.display()),
        _ => format!("{}: {}", path.display(), e),
    })?;
    // EIR-011: report invalid UTF-8 with the byte offset.
    match klions_lexer::check_utf8(&bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(d) => Err(format!("{}: {}", path.display(), d.message)),
    }
}

/// Report every diagnostic. Returns true when any error was present.
fn report(path: &Path, source: &str, opts: &Options, diagnostics: &[klions_diagnostics::Diagnostic]) -> bool {
    let sm = SourceMap::new(&path.display().to_string(), source);
    let color = color_mode(opts);
    let mut errors = 0;
    let mut warnings = 0;

    for d in diagnostics {
        match d.severity {
            Severity::Error => errors += 1,
            Severity::Warning => warnings += 1,
            Severity::Hint => {}
        }
        if opts.json {
            // EIR-008: one JSON object per line, on stdout for piping.
            println!("{}", render::render_json(d, &sm));
        } else {
            eprint!("{}", render::render(d, &sm, color));
            eprintln!();
        }
    }
    if !opts.json && (errors > 0 || warnings > 0) {
        eprint!("{}", render::summary(errors, warnings, 0, color));
    }
    errors > 0
}

fn check_files(opts: &Options) -> ExitCode {
    if opts.files.is_empty() {
        eprintln!("klions check: no input files");
        return ExitCode::from(2);
    }
    let started = klions_platform::now_ms();
    let mut failed = false;
    let mut checked = 0usize;

    for path in &opts.files {
        let source = match read_source(path) {
            Ok(s) => s,
            Err(m) => {
                eprintln!("klions: {}", m);
                failed = true;
                continue;
            }
        };
        let analysis = klions_frontend::analyze(&source);
        if report(path, &source, opts, &analysis.diagnostics) {
            failed = true;
        }
        checked += 1;
    }

    if !failed && !opts.quiet && !opts.json {
        eprintln!(
            "checked {} file{} — no errors",
            checked,
            if checked == 1 { "" } else { "s" }
        );
    }
    if opts.time {
        eprintln!("check took {} ms", klions_platform::now_ms() - started);
    }
    if failed {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

fn run_file(opts: &Options) -> ExitCode {
    let Some(path) = opts.files.first().cloned() else {
        eprintln!("klions run: no input file");
        eprintln!("usage: klions run <file.kl>");
        return ExitCode::from(2);
    };

    let source = match read_source(&path) {
        Ok(s) => s,
        Err(m) => {
            eprintln!("klions: {}", m);
            return ExitCode::from(2);
        }
    };

    let analysis = klions_frontend::analyze(&source);
    let had_errors = report(&path, &source, opts, &analysis.diagnostics);
    if had_errors {
        // EIR-010: compilation failure is exit 2, distinct from a run-time fault.
        return ExitCode::from(2);
    }

    let started = klions_platform::now_ms();
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let root = find_project_root(&dir);
    let seed = opts.seed;
    let program = analysis.program;

    // FR-RT-005 depends on a stack large enough that the frame limit, not the
    // operating system, is what stops runaway recursion.
    let outcome = klions_runtime::on_interpreter_stack(move || {
        let mut interp = klions_runtime::Interpreter::new(program, seed, dir);
        interp.project_root = root;
        interp.run()
    });

    let elapsed = klions_platform::now_ms() - started;
    if opts.time {
        eprintln!("ran in {} ms", elapsed);
    }

    match outcome {
        Err(_) => {
            eprintln!("klions: internal error — the interpreter thread panicked");
            eprintln!("this is a bug; please report it with the program that triggered it");
            ExitCode::from(101)
        }
        Ok(Ok(code)) => {
            if code == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(code.clamp(0, 255) as u8)
            }
        }
        Ok(Err(e)) => {
            if e.interrupted {
                eprintln!("klions: interrupted");
                return ExitCode::from(130);
            }
            let sm = SourceMap::new(&path.display().to_string(), &source);
            let d = e.to_diagnostic();
            if opts.json {
                println!("{}", render::render_json(&d, &sm));
            } else {
                eprint!("{}", render::render(&d, &sm, color_mode(opts)));
            }
            ExitCode::from(101)
        }
    }
}

/// The project root is the nearest ancestor holding `klions.toml`, else the
/// source directory. Used by NFR-S-006 to bound dataset paths.
fn find_project_root(from: &Path) -> Option<PathBuf> {
    let mut cur = from.canonicalize().ok()?;
    loop {
        if cur.join("klions.toml").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

/// `klions explain K1001` — FR-ERR-012.
fn explain(code: Option<&str>) -> ExitCode {
    let Some(code) = code else {
        eprintln!("usage: klions explain <code>    e.g. klions explain K1001");
        return ExitCode::from(2);
    };
    let code = code.to_uppercase();
    match explanation(&code) {
        Some(text) => {
            println!("{}\n", code);
            println!("{}", text);
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("klions: no explanation is registered for `{}`", code);
            eprintln!("codes look like K1001; see docs/diagnostics/ for the full index");
            ExitCode::from(2)
        }
    }
}

fn explanation(code: &str) -> Option<&'static str> {
    use klions_diagnostics::codes as c;
    Some(match code {
        _ if code == c::SHAPE_MISMATCH => {
            "Two tensors in an element-wise operation had shapes that cannot be broadcast.\n\n\
             Shapes are compared from the right. A dimension of 1 is stretched to match; any\n\
             other disagreement is an error. So [64, 128] and [128] combine fine, but\n\
             [64, 128] and [64, 10] do not: axis 1 is 128 on one side and 10 on the other.\n\n\
             Fix it by reshaping one operand, or by checking whether an earlier layer produced\n\
             the width you expected."
        }
        _ if code == c::MATMUL_SHAPE => {
            "Matrix multiplication requires the inner dimensions to agree: [m, k] @ [k, n]\n\
             gives [m, n]. If the left operand is [64, 784], the right must have 784 rows.\n\n\
             A frequent cause is a transposed weight matrix. `.t()` swaps the last two axes."
        }
        _ if code == c::LAYER_CHAIN_MISMATCH => {
            "Layers in a model run in declaration order, so each layer's `inputs` must equal\n\
             the previous layer's `outputs`.\n\n\
             This is checked when the model is declared, before any data is loaded, so the\n\
             error appears immediately rather than on the first forward pass."
        }
        _ if code == c::NO_IMPLICIT_CONVERSION => {
            "KLIONS does not convert between numeric types on its own. An `i32` and an `f32`\n\
             cannot be added directly; write `x as f32` to make the conversion visible.\n\n\
             Numeric literals are exempt: they take their type from context, so\n\
             `let x: i32 = 1` is exact, not a conversion."
        }
        _ if code == c::NON_FINITE_GRADIENT => {
            "A loss or gradient became NaN or infinity, so training stopped. Continuing would\n\
             corrupt every parameter.\n\n\
             The usual causes are a learning rate that is too high, or unscaled inputs. Try\n\
             `dataset.normalize()` and a smaller learning rate."
        }
        _ if code == c::DEFERRED_FEATURE => {
            "This feature is recognized but not implemented in this release. The message names\n\
             the release that will carry it.\n\n\
             KLIONS reports these rather than ignoring them: silently accepting an option that\n\
             does nothing would misrepresent what actually ran."
        }
        _ if code == c::STACK_OVERFLOW => {
            "The interpreter's call-stack limit was reached, which almost always means a\n\
             recursive function is missing its base case.\n\n\
             The limit is reported as an error rather than being allowed to crash the process."
        }
        _ if code == c::MODEL_INPUT_MISMATCH => {
            "The model's first layer expects a different number of input features than the\n\
             dataset provides.\n\n\
             For flattened 28x28 images that is 784. Check both the layer's `inputs` and any\n\
             reshaping applied to the dataset."
        }
        _ => return None,
    })
}

fn print_help() {
    println!(
        r#"klions {VERSION} — the KLIONS toolchain

USAGE
    klions <command> [options] [files]
    klions <file.kl>                 shorthand for `klions run`

COMMANDS
    run <file.kl>       compile and execute a program
    check <files...>    analyze without executing
    fmt <files...>      format source in place (--check to verify only)
    repl                start an interactive session
    explain <code>      describe a diagnostic code, e.g. K1001
    health              run compiler self-diagnostic health check
    version             print the version
    help                print this message

OPTIONS
    --seed <n>          seed the global generator (default 42)
    --json              emit diagnostics as newline-delimited JSON
    --time              report elapsed time
    --color/--no-color  force colour on or off
    -q, --quiet         suppress the success summary

EXIT CODES
    0    success
    1    the program returned a non-zero status
    2    compilation failed
    101  the program failed at run time
    130  interrupted

EXAMPLES
    klions run examples/xor.kl
    klions run examples/mnist.kl --seed 7 --time
    klions check src/*.kl --json
    klions explain K1001
    klions health
"#
    );
}

fn health_check() -> ExitCode {
    let source = "fn main() { println(\"healthy\") }";
    let analysis = klions_frontend::analyze(source);
    if analysis.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        eprintln!("klions: health check failed: frontend analysis failed");
        return ExitCode::from(1);
    }
    println!("{{\"status\":\"healthy\",\"engine\":\"klions\",\"version\":\"{}\"}}", VERSION);
    ExitCode::SUCCESS
}

