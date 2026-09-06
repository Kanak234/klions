//! `klions-lsp` — a Language Server Protocol 3.17 implementation.
//!
//! FR-LSP-001: the server calls `klions_frontend::analyze`, the same function
//! the compiler calls. There is no second parser and no second type checker,
//! so an editor squiggle and a `klions check` error are the same computation.

mod json;

use std::collections::HashMap;
use std::io::{BufRead, Write};

use json::Json;
use klions_ast::{Item, Program};
use klions_diagnostics::{Severity, SourceMap, Span};
use klions_frontend::Analysis;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Server {
    documents: HashMap<String, String>,
    shutdown_requested: bool,
}

fn main() {
    let mut server = Server { documents: HashMap::new(), shutdown_requested: false };
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();

    loop {
        let message = match read_message(&mut reader) {
            Ok(Some(m)) => m,
            Ok(None) => break, // clean EOF
            Err(e) => {
                eprintln!("klions-lsp: {}", e);
                break;
            }
        };
        let request = match json::parse(&message) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("klions-lsp: malformed request: {}", e);
                continue;
            }
        };
        if server.handle(&request) {
            break;
        }
    }
}

/// LSP framing: `Content-Length: N\r\n\r\n` then N bytes.
fn read_message(reader: &mut impl BufRead) -> Result<Option<String>, String> {
    let mut length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("read failed: {}", e))?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(v) = trimmed
            .strip_prefix("Content-Length:")
            .or_else(|| trimmed.strip_prefix("content-length:"))
        {
            length = v.trim().parse().ok();
        }
    }
    let len = length.ok_or("message had no Content-Length header")?;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("short read: {}", e))?;
    String::from_utf8(buf)
        .map(Some)
        .map_err(|e| format!("body was not UTF-8: {}", e))
}

fn send(payload: &Json) {
    let body = payload.to_string();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = write!(out, "Content-Length: {}\r\n\r\n{}", body.len(), body);
    let _ = out.flush();
}

impl Server {
    /// Returns true when the server should exit.
    fn handle(&mut self, req: &Json) -> bool {
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = req.get("id").cloned();

        match method {
            "initialize" => {
                self.respond(id, capabilities());
            }
            "initialized" => {}
            "shutdown" => {
                self.shutdown_requested = true;
                self.respond(id, Json::Null);
            }
            "exit" => return true,
            "textDocument/didOpen" => {
                if let (Some(uri), Some(text)) = (
                    req.path("params.textDocument.uri").and_then(|v| v.as_str()),
                    req.path("params.textDocument.text").and_then(|v| v.as_str()),
                ) {
                    self.documents.insert(uri.to_string(), text.to_string());
                    self.publish(uri);
                }
            }
            "textDocument/didChange" => {
                if let Some(uri) = req.path("params.textDocument.uri").and_then(|v| v.as_str()) {
                    // Full-sync only, as advertised in the capabilities.
                    if let Some(changes) =
                        req.path("params.contentChanges").and_then(|v| v.as_array())
                    {
                        if let Some(text) =
                            changes.last().and_then(|c| c.get("text")).and_then(|t| t.as_str())
                        {
                            self.documents.insert(uri.to_string(), text.to_string());
                            self.publish(uri);
                        }
                    }
                }
            }
            "textDocument/didSave" => {
                if let Some(uri) = req.path("params.textDocument.uri").and_then(|v| v.as_str()) {
                    self.publish(uri);
                }
            }
            "textDocument/didClose" => {
                if let Some(uri) = req.path("params.textDocument.uri").and_then(|v| v.as_str()) {
                    self.documents.remove(uri);
                    // Clear the squiggles for a closed file.
                    send(&notification(
                        "textDocument/publishDiagnostics",
                        json_obj! {
                            "uri" => Json::str(uri),
                            "diagnostics" => Json::Arr(vec![]),
                        },
                    ));
                }
            }
            "textDocument/hover" => {
                let r = self.hover(req);
                self.respond(id, r);
            }
            "textDocument/completion" => {
                let r = self.completion(req);
                self.respond(id, r);
            }
            "textDocument/documentSymbol" => {
                let r = self.document_symbols(req);
                self.respond(id, r);
            }
            "textDocument/definition" => {
                let r = self.definition(req);
                self.respond(id, r);
            }
            "textDocument/formatting" => {
                self.respond(id, Json::Null);
            }
            _ => {
                // Unknown request: answer with null so the client is not left waiting.
                if id.is_some() {
                    self.respond(id, Json::Null);
                }
            }
        }
        false
    }

    fn respond(&self, id: Option<Json>, result: Json) {
        let Some(id) = id else { return };
        send(&json_obj! {
            "jsonrpc" => Json::str("2.0"),
            "id" => id,
            "result" => result,
        });
    }

    fn document(&self, req: &Json) -> Option<(String, String)> {
        let uri = req.path("params.textDocument.uri")?.as_str()?.to_string();
        let text = self.documents.get(&uri)?.clone();
        Some((uri, text))
    }

    /// FR-LSP-002: diagnostics come from the compiler's own analysis.
    fn publish(&self, uri: &str) {
        let Some(text) = self.documents.get(uri) else { return };
        let analysis = klions_frontend::analyze(text);
        let sm = SourceMap::new(uri, text);

        let items: Vec<Json> = analysis
            .diagnostics
            .iter()
            .map(|d| {
                json_obj! {
                    "range" => range_of(&sm, d.span),
                    "severity" => Json::int(match d.severity {
                        Severity::Error => 1,
                        Severity::Warning => 2,
                        Severity::Hint => 4,
                    }),
                    "code" => Json::str(d.code),
                    "source" => Json::str("klions"),
                    "message" => Json::str(full_message(d)),
                }
            })
            .collect();

        send(&notification(
            "textDocument/publishDiagnostics",
            json_obj! {
                "uri" => Json::str(uri),
                "diagnostics" => Json::Arr(items),
            },
        ));
    }

    /// FR-LSP-003: hover shows the type, and for tensors, the shape.
    fn hover(&self, req: &Json) -> Json {
        let Some((uri, text)) = self.document(req) else {
            return Json::Null;
        };
        let Some(offset) = position_offset(req, &uri, &text) else {
            return Json::Null;
        };
        let analysis = klions_frontend::analyze(&text);
        let Some((word, span)) = word_at(&text, offset) else {
            return Json::Null;
        };

        let doc = describe(&analysis, &word).unwrap_or_else(|| {
            builtin_doc(&word).unwrap_or_else(|| format!("`{}`", word))
        });
        let sm = SourceMap::new(&uri, &text);
        json_obj! {
            "contents" => json_obj! {
                "kind" => Json::str("markdown"),
                "value" => Json::str(doc),
            },
            "range" => range_of(&sm, span),
        }
    }

    /// FR-LSP-004: completion for keywords, layers, options, and local names.
    fn completion(&self, req: &Json) -> Json {
        let Some((_uri, text)) = self.document(req) else {
            return Json::Arr(vec![]);
        };
        let analysis = klions_frontend::analyze(&text);
        let mut items: Vec<Json> = Vec::new();

        for kw in KEYWORDS {
            items.push(entry(kw, 14, "keyword"));
        }
        for l in klions_ast::LAYER_NAMES {
            items.push(entry(l, 7, "layer"));
        }
        for (name, release) in klions_ast::DEFERRED_LAYERS {
            // Offered, but marked and sorted last so it never shadows a usable
            // layer. Hiding it entirely would leave the user guessing why
            // `Conv2D` is rejected (FR-ERR-010).
            let mut e = entry(name, 7, &format!("layer — scheduled for {}", release));
            e.set("sortText", Json::str(format!("zz{}", name)));
            items.push(e);
        }
        for k in klions_ast::TRAIN_KEYS {
            items.push(entry(k, 5, "training option"));
        }
        for b in klions_types::builtins::BUILTIN_NAMES {
            items.push(entry(b, 3, "builtin function"));
        }
        for a in klions_types::builtins::ACTIVATIONS {
            items.push(entry(a, 12, "activation"));
        }
        for l in klions_types::builtins::LOSSES {
            items.push(entry(l, 12, "loss function"));
        }
        for o in klions_types::builtins::OPTIMIZERS {
            items.push(entry(o, 3, "optimizer"));
        }
        for (name, info) in &analysis.functions {
            let sig = format!(
                "fn {}({}) -> {}",
                name,
                info.params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t.display()))
                    .collect::<Vec<_>>()
                    .join(", "),
                info.ret.display()
            );
            items.push(entry(name, 3, &sig));
        }
        for name in analysis.models.keys() {
            items.push(entry(name, 7, "model"));
        }
        Json::Arr(items)
    }

    /// FR-LSP-005: an outline of the file's items.
    fn document_symbols(&self, req: &Json) -> Json {
        let Some((uri, text)) = self.document(req) else {
            return Json::Arr(vec![]);
        };
        let analysis = klions_frontend::analyze(&text);
        let sm = SourceMap::new(&uri, &text);
        Json::Arr(symbols(&analysis.program, &sm))
    }

    /// FR-LSP-006: jump to a top-level definition.
    fn definition(&self, req: &Json) -> Json {
        let Some((uri, text)) = self.document(req) else {
            return Json::Null;
        };
        let Some(offset) = position_offset(req, &uri, &text) else {
            return Json::Null;
        };
        let Some((word, _)) = word_at(&text, offset) else {
            return Json::Null;
        };
        let analysis = klions_frontend::analyze(&text);
        let sm = SourceMap::new(&uri, &text);

        let span = analysis
            .functions
            .get(&word)
            .map(|f| f.span)
            .or_else(|| analysis.models.get(&word).map(|m| m.span));

        match span {
            Some(s) => json_obj! {
                "uri" => Json::str(uri),
                "range" => range_of(&sm, s),
            },
            None => Json::Null,
        }
    }
}

const KEYWORDS: &[&str] = &[
    "let", "mut", "fn", "const", "return", "if", "else", "while", "for", "in", "model", "train",
    "using", "import", "true", "false", "null", "as", "no_grad", "break", "continue", "tensor",
];

/// One completion item.
fn entry(label: &str, kind: i64, detail: &str) -> Json {
    json_obj! {
        "label" => Json::str(label),
        "kind" => Json::int(kind),
        "detail" => Json::str(detail),
    }
}

fn notification(method: &str, params: Json) -> Json {
    json_obj! {
        "jsonrpc" => Json::str("2.0"),
        "method" => Json::str(method),
        "params" => params,
    }
}

fn capabilities() -> Json {
    json_obj! {
        "capabilities" => json_obj! {
            // 1 = full document sync.
            "textDocumentSync" => Json::int(1),
            "hoverProvider" => Json::Bool(true),
            "definitionProvider" => Json::Bool(true),
            "documentSymbolProvider" => Json::Bool(true),
            "completionProvider" => json_obj! {
                "triggerCharacters" => Json::Arr(vec![Json::str("."), Json::str(":")]),
            },
        },
        "serverInfo" => json_obj! {
            "name" => Json::str("klions-lsp"),
            "version" => Json::str(VERSION),
        },
    }
}

fn full_message(d: &klions_diagnostics::Diagnostic) -> String {
    let mut m = d.message.clone();
    for n in &d.notes {
        m.push_str(&format!("\n\nnote: {}", n));
    }
    if let Some(h) = &d.help {
        m.push_str(&format!("\n\nhelp: {}", h));
    }
    m
}

/// LSP positions are UTF-16 code units on zero-based lines.
fn range_of(sm: &SourceMap<'_>, span: Span) -> Json {
    let (sl, sc) = sm.lsp_position(span.start);
    let (el, ec) = sm.lsp_position(span.end.max(span.start));
    json_obj! {
        "start" => json_obj! { "line" => Json::int(sl as i64), "character" => Json::int(sc as i64) },
        "end"   => json_obj! { "line" => Json::int(el as i64), "character" => Json::int(ec as i64) },
    }
}

fn position_offset(req: &Json, uri: &str, text: &str) -> Option<usize> {
    let line = req.path("params.position.line")?.as_usize()?;
    let ch = req.path("params.position.character")?.as_usize()?;
    let sm = SourceMap::new(uri, text);
    Some(sm.offset_of_lsp(line, ch))
}

/// The identifier surrounding a byte offset.
fn word_at(text: &str, offset: usize) -> Option<(String, Span)> {
    let b = text.as_bytes();
    if b.is_empty() {
        return None;
    }
    let mut start = offset.min(b.len().saturating_sub(1));
    if !is_word_byte(b[start]) && start > 0 {
        start -= 1;
    }
    if !is_word_byte(*b.get(start)?) {
        return None;
    }
    while start > 0 && is_word_byte(b[start - 1]) {
        start -= 1;
    }
    let mut end = start;
    while end < b.len() && is_word_byte(b[end]) {
        end += 1;
    }
    Some((text[start..end].to_string(), Span::new(start, end)))
}

fn is_word_byte(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric()
}

/// Hover text for a name defined in this file.
fn describe(analysis: &Analysis, name: &str) -> Option<String> {
    if let Some(f) = analysis.functions.get(name) {
        let params = f
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t.display()))
            .collect::<Vec<_>>()
            .join(", ");
        return Some(format!(
            "```klions\nfn {}({}) -> {}\n```",
            name,
            params,
            f.ret.display()
        ));
    }
    if let Some(m) = analysis.models.get(name) {
        let mut s = format!("```klions\nmodel {}\n```\n\n", name);
        // FR-LSP-003: shapes are the point of hovering on a model.
        match (m.input_width, m.output_width) {
            (Some(i), Some(o)) => {
                s.push_str(&format!("Accepts `[batch, {}]`, produces `[batch, {}]`.", i, o))
            }
            (Some(i), None) => s.push_str(&format!("Accepts `[batch, {}]`.", i)),
            _ => s.push_str("Layer widths are not statically known."),
        }
        return Some(s);
    }
    None
}

fn builtin_doc(name: &str) -> Option<String> {
    let text = match name {
        "tensor" => "`tensor<f32, D...>` — a dense array. Dimensions are part of the type, so shape errors are caught at compile time. Write `_` for a dimension known only at run time.",
        "train" => "`train <model> using <dataset> { ... }` — runs the training loop. `epochs` is required; there is no default worth guessing.",
        "no_grad" => "`no_grad { ... }` — suspends gradient tracking. Use it for inference and for metrics.",
        "model" => "`model Name { layer = Dense(...) }` — declares a layer chain. Layers run in declaration order, and widths are checked against each other at compile time.",
        "Dense" => "`Dense(inputs, outputs, activation)` — a fully connected layer. Weights use Glorot uniform initialization; biases start at zero.",
        "Dropout" => "`Dropout(p)` — zeroes a fraction `p` of activations while training, and does nothing during inference.",
        "predict" => "`model.predict(x)` — a forward pass with gradient tracking off.",
        "evaluate" => "`model.evaluate(dataset)` — returns loss and accuracy over the whole dataset.",
        "cross_entropy" => "Consumes raw logits and applies log-sum-exp internally. Do **not** put a `Softmax` layer before it.",
        "reshape" => "`t.reshape(dims...)` — one dimension may be `-1` and is inferred from the element count.",
        "let" => "`let name = value` binds an immutable name. Add `mut` to allow assignment.",
        "as" => "`expr as T` — an explicit numeric conversion. KLIONS never converts silently.",
        _ => return None,
    };
    Some(text.to_string())
}

fn symbols(program: &Program, sm: &SourceMap<'_>) -> Vec<Json> {
    program
        .items
        .iter()
        .map(|item| {
            let kind = match item {
                Item::Function(_) => 12, // Function
                Item::Model(_) => 5,     // Class
                Item::Const(_) => 14,    // Constant
            };
            let children = match item {
                Item::Model(m) => m
                    .layers
                    .iter()
                    .map(|l| {
                        json_obj! {
                            "name" => Json::str(&l.name),
                            "detail" => Json::str(&l.layer.kind),
                            "kind" => Json::int(8), // Field
                            "range" => range_of(sm, l.span),
                            "selectionRange" => range_of(sm, l.name_span),
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            json_obj! {
                "name" => Json::str(item.name()),
                "detail" => Json::str(item.kind_str()),
                "kind" => Json::int(kind),
                "range" => range_of(sm, item.span()),
                "selectionRange" => range_of(sm, item.name_span()),
                "children" => Json::Arr(children),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server_with(text: &str) -> Server {
        let mut s = Server { documents: HashMap::new(), shutdown_requested: false };
        s.documents.insert("file:///t.kl".to_string(), text.to_string());
        s
    }

    fn request(method: &str, params: Json) -> Json {
        json_obj! {
            "jsonrpc" => Json::str("2.0"),
            "id" => Json::int(1),
            "method" => Json::str(method),
            "params" => params,
        }
    }

    fn doc_params() -> Json {
        json_obj! {
            "textDocument" => json_obj! { "uri" => Json::str("file:///t.kl") },
        }
    }

    fn at(line: usize, ch: usize) -> Json {
        json_obj! {
            "textDocument" => json_obj! { "uri" => Json::str("file:///t.kl") },
            "position" => json_obj! {
                "line" => Json::int(line as i64),
                "character" => Json::int(ch as i64),
            },
        }
    }

    #[test]
    fn framing_round_trips() {
        let body = r#"{"jsonrpc":"2.0","method":"exit"}"#;
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut cursor = std::io::Cursor::new(framed.into_bytes());
        let got = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(got, body);
    }

    #[test]
    fn word_lookup_finds_the_identifier_under_the_cursor() {
        let text = "let hidden = 1";
        let (w, _) = word_at(text, 6).unwrap();
        assert_eq!(w, "hidden");
        // Cursor at the very end of the word still resolves it.
        let (w, _) = word_at(text, 10).unwrap();
        assert_eq!(w, "hidden");
    }

    #[test]
    fn hover_reports_a_function_signature() {
        let s = server_with("fn add(a: i64, b: i64) -> i64 { return a + b }\nfn main() { }");
        let r = s.hover(&request("textDocument/hover", at(0, 4)));
        let text = r.path("contents.value").unwrap().as_str().unwrap();
        assert!(text.contains("fn add(a: i64, b: i64) -> i64"), "got {}", text);
    }

    /// FR-LSP-003: hovering a model reports the shapes it accepts and produces.
    #[test]
    fn hover_on_a_model_reports_its_shapes() {
        let s = server_with(
            "model Net {\n    h = Dense(inputs=784, outputs=128)\n    o = Dense(inputs=128, outputs=10)\n}\nfn main() { }",
        );
        let r = s.hover(&request("textDocument/hover", at(0, 7)));
        let text = r.path("contents.value").unwrap().as_str().unwrap();
        assert!(text.contains("[batch, 784]") && text.contains("[batch, 10]"), "got {}", text);
    }

    #[test]
    fn completion_offers_layers_and_local_functions() {
        let s = server_with("fn helper() { }\nfn main() { }");
        let r = s.completion(&request("textDocument/completion", doc_params()));
        let labels: Vec<&str> = r
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|i| i.get("label").and_then(|l| l.as_str()))
            .collect();
        assert!(labels.contains(&"Dense"));
        assert!(labels.contains(&"helper"));
        assert!(labels.contains(&"epochs"));
    }

    /// FR-ERR-010: deferred layers are offered but sorted last.
    #[test]
    fn deferred_layers_are_marked_and_sorted_last() {
        let s = server_with("fn main() { }");
        let r = s.completion(&request("textDocument/completion", doc_params()));
        let conv = r
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i.get("label").and_then(|l| l.as_str()) == Some("Conv2D"))
            .expect("Conv2D should be offered");
        assert!(conv.get("detail").unwrap().as_str().unwrap().contains("0.2.0"));
        assert!(conv.get("sortText").unwrap().as_str().unwrap().starts_with("zz"));
    }

    #[test]
    fn document_symbols_nest_layers_under_their_model() {
        let s = server_with(
            "model Net {\n    h = Dense(inputs=4, outputs=2)\n}\nfn main() { }",
        );
        let r = s.document_symbols(&request("textDocument/documentSymbol", doc_params()));
        let items = r.as_array().unwrap();
        assert_eq!(items.len(), 2);
        let net = &items[0];
        assert_eq!(net.get("name").unwrap().as_str(), Some("Net"));
        assert_eq!(net.get("children").unwrap().as_array().unwrap().len(), 1);
    }

    #[test]
    fn definition_points_at_the_declaration() {
        let s = server_with("fn helper() { }\nfn main() { helper() }");
        let r = s.definition(&request("textDocument/definition", at(1, 13)));
        assert_eq!(r.get("uri").unwrap().as_str(), Some("file:///t.kl"));
        assert_eq!(r.path("range.start.line").unwrap().as_i64(), Some(0));
    }

    #[test]
    fn initialize_advertises_the_implemented_capabilities() {
        let c = capabilities();
        assert_eq!(c.path("capabilities.hoverProvider"), Some(&Json::Bool(true)));
        assert_eq!(c.path("capabilities.textDocumentSync").unwrap().as_i64(), Some(1));
        assert!(c.path("capabilities.completionProvider").is_some());
    }

    /// FR-LSP-002: the squiggle carries the same code the compiler emits.
    #[test]
    fn diagnostics_carry_the_compiler_code() {
        let text = "fn main() {\n    let a: tensor<f32, 2, 3> = zeros(2, 3)\n    let b: tensor<f32, 2, 4> = zeros(2, 4)\n    let _c = a + b\n}";
        let analysis = klions_frontend::analyze(text);
        assert!(analysis
            .diagnostics
            .iter()
            .any(|d| d.code == klions_diagnostics::codes::SHAPE_MISMATCH));
    }
}
