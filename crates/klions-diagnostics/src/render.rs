//! Rendering diagnostics for humans (FR-ERR-002) and machines (EIR-008).

use crate::{codes, Diagnostic, Severity, SourceMap};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    Always,
    Never,
}

impl ColorMode {
    /// EIR-009: colour only on a TTY, never when NO_COLOR is set.
    pub fn detect(is_tty: bool, no_color_env: bool, forced_off: bool) -> Self {
        if forced_off || no_color_env || !is_tty {
            ColorMode::Never
        } else {
            ColorMode::Always
        }
    }
}

struct Palette {
    on: bool,
}

impl Palette {
    fn severity(&self, s: Severity) -> &'static str {
        if !self.on {
            return "";
        }
        match s {
            Severity::Error => "\x1b[1;31m",
            Severity::Warning => "\x1b[1;33m",
            Severity::Hint => "\x1b[1;36m",
        }
    }
    fn bold(&self) -> &'static str {
        if self.on { "\x1b[1m" } else { "" }
    }
    fn blue(&self) -> &'static str {
        if self.on { "\x1b[1;34m" } else { "" }
    }
    fn green(&self) -> &'static str {
        if self.on { "\x1b[1;32m" } else { "" }
    }
    fn dim(&self) -> &'static str {
        if self.on { "\x1b[2m" } else { "" }
    }
    fn reset(&self) -> &'static str {
        if self.on { "\x1b[0m" } else { "" }
    }
}

/// Render one diagnostic in the rustc-style block form:
///
/// ```text
/// error[K1001]: shape mismatch in `+`
///   --> mnist.kl:12:19
///    |
/// 12 |     let z = a + b
///    |             ^^^^^ left is [64, 128], right is [64, 10]
///    |
///    = note: axis 1 disagrees: 128 vs 10
///    = help: reshape one operand, or use a broadcast-compatible shape
/// ```
pub fn render(d: &Diagnostic, sm: &SourceMap<'_>, color: ColorMode) -> String {
    let p = Palette { on: color == ColorMode::Always };
    let mut out = String::new();

    let (line, col) = sm.position(d.span.start);
    let sev = d.severity.as_str();

    out.push_str(&format!(
        "{}{}[{}]{}: {}{}{}\n",
        p.severity(d.severity),
        sev,
        d.code,
        p.reset(),
        p.bold(),
        d.message,
        p.reset()
    ));

    let gutter_w = line.to_string().len().max(2);
    let pad = " ".repeat(gutter_w);

    out.push_str(&format!(
        "{}{}-->{} {}:{}:{}\n",
        pad,
        p.blue(),
        p.reset(),
        sm.name,
        line,
        col
    ));

    let bar = format!("{}{} |{}", p.blue(), pad, p.reset());
    out.push_str(&bar);
    out.push('\n');

    let src_line = sm.line_text(line);
    out.push_str(&format!(
        "{}{:>w$} |{} {}\n",
        p.blue(),
        line,
        p.reset(),
        src_line.replace('\t', "    "),
        w = gutter_w
    ));

    // Underline. Columns are character counts; expand tabs to 4 to match above.
    let line_start = sm.line_span(line).start;
    let prefix = &sm.text[line_start..d.span.start.min(sm.text.len())];
    let lead: usize = prefix.chars().map(|c| if c == '\t' { 4 } else { 1 }).sum();
    let body_end = d.span.end.min(sm.line_span(line).end).max(d.span.start);
    let body = &sm.text[d.span.start.min(sm.text.len())..body_end.min(sm.text.len())];
    let width: usize = body
        .chars()
        .filter(|c| *c != '\n' && *c != '\r')
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum::<usize>()
        .max(1);

    let caret = "^".repeat(width);
    let label = d
        .secondary
        .first()
        .map(|(_, l)| format!(" {}", l))
        .unwrap_or_default();
    out.push_str(&format!(
        "{}{} |{} {}{}{}{}{}\n",
        p.blue(),
        pad,
        p.reset(),
        " ".repeat(lead),
        p.severity(d.severity),
        caret,
        label,
        p.reset()
    ));

    if !d.notes.is_empty() || d.help.is_some() {
        out.push_str(&bar);
        out.push('\n');
    }
    for n in &d.notes {
        out.push_str(&format!(
            "{}{} ={} {}note{}: {}\n",
            p.blue(),
            pad,
            p.reset(),
            p.dim(),
            p.reset(),
            n
        ));
    }
    if let Some(h) = &d.help {
        out.push_str(&format!(
            "{}{} ={} {}help{}: {}\n",
            p.blue(),
            pad,
            p.reset(),
            p.green(),
            p.reset(),
            h
        ));
    }
    out
}

/// EIR-008: one newline-delimited JSON object per diagnostic.
pub fn render_json(d: &Diagnostic, sm: &SourceMap<'_>) -> String {
    let (line, col) = sm.position(d.span.start);
    let notes: Vec<String> = d
        .notes
        .iter()
        .chain(d.help.iter())
        .map(|s| json_string(s))
        .collect();
    format!(
        "{{\"severity\":\"{}\",\"code\":\"{}\",\"class\":\"{}\",\"message\":{},\"file\":{},\"line\":{},\"column\":{},\"span\":[{},{}],\"notes\":[{}]}}",
        d.severity.as_str(),
        d.code,
        codes::class_of(d.code),
        json_string(&d.message),
        json_string(&sm.name),
        line,
        col,
        d.span.start,
        d.span.end,
        notes.join(",")
    )
}

pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Trailing summary line, e.g. `error: 3 errors, 1 warning emitted`.
pub fn summary(errors: usize, warnings: usize, suppressed: usize, color: ColorMode) -> String {
    let p = Palette { on: color == ColorMode::Always };
    let mut parts = Vec::new();
    if errors > 0 {
        parts.push(format!("{} error{}", errors, plural(errors)));
    }
    if warnings > 0 {
        parts.push(format!("{} warning{}", warnings, plural(warnings)));
    }
    if parts.is_empty() {
        return String::new();
    }
    let sev = if errors > 0 { Severity::Error } else { Severity::Warning };
    let mut s = format!(
        "{}{}{}: {} emitted\n",
        p.severity(sev),
        sev.as_str(),
        p.reset(),
        parts.join(", ")
    );
    if suppressed > 0 {
        s.push_str(&format!(
            "{}note{}: {} further diagnostic{} suppressed\n",
            p.dim(),
            p.reset(),
            suppressed,
            plural(suppressed)
        ));
    }
    s
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
