//! A minimal JSON reader and writer.
//!
//! The language server needs JSON and nothing else, and DC-003 asks the whole
//! toolchain to build with one command and no external crates. A few hundred
//! lines here buys that.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn obj() -> Json {
        Json::Obj(BTreeMap::new())
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    /// Follow a dotted path, e.g. `params.textDocument.uri`.
    pub fn path(&self, path: &str) -> Option<&Json> {
        let mut cur = self;
        for seg in path.split('.') {
            cur = cur.get(seg)?;
        }
        Some(cur)
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) => Some(*n as i64),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().map(|v| v.max(0) as usize)
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn set(&mut self, key: &str, v: Json) -> &mut Json {
        if let Json::Obj(m) = self {
            m.insert(key.to_string(), v);
        }
        self
    }

    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }
    #[allow(dead_code)]
    pub fn num(n: impl Into<f64>) -> Json {
        Json::Num(n.into())
    }
    /// Integers are the common case in LSP payloads.
    pub fn int(n: i64) -> Json {
        Json::Num(n as f64)
    }

    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 9e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{}", n));
                }
            }
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(s: &str, out: &mut String) {
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
}

/// Build an object from key/value pairs.
#[macro_export]
macro_rules! json_obj {
    ($($k:expr => $v:expr),* $(,)?) => {{
        let mut o = $crate::json::Json::obj();
        $( o.set($k, $v); )*
        o
    }};
}

pub fn parse(text: &str) -> Result<Json, String> {
    let b: Vec<char> = text.chars().collect();
    let mut p = Parser { b, i: 0 };
    p.skip_ws();
    let v = p.value()?;
    Ok(v)
}

struct Parser {
    b: Vec<char>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.b.get(self.i).copied()
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n') | Some('\r')) {
            self.i += 1;
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected `{}` at offset {}", c, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('t') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some('f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some('n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected {:?} at offset {}", other, self.i)),
        }
    }

    fn literal(&mut self, want: &str) -> Result<(), String> {
        for c in want.chars() {
            if self.peek() != Some(c) {
                return Err(format!("expected `{}` at offset {}", want, self.i));
            }
            self.i += 1;
        }
        Ok(())
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect('{')?;
        let mut m = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.skip_ws();
            let k = self.string()?;
            self.skip_ws();
            self.expect(':')?;
            let v = self.value()?;
            m.insert(k, v);
            self.skip_ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                _ => return Err(format!("expected `,` or `}}` at offset {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected `,` or `]` at offset {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".to_string()),
                Some('"') => {
                    self.i += 1;
                    return Ok(s);
                }
                Some('\\') => {
                    self.i += 1;
                    let c = self.peek().ok_or("unterminated escape")?;
                    self.i += 1;
                    match c {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        'b' => s.push('\u{8}'),
                        'f' => s.push('\u{c}'),
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'u' => {
                            let hex: String = (0..4).filter_map(|k| self.b.get(self.i + k)).collect();
                            self.i += 4;
                            let cp = u32::from_str_radix(&hex, 16)
                                .map_err(|_| format!("bad \\u escape `{}`", hex))?;
                            // Surrogate pairs, as emitted by some clients.
                            if (0xD800..0xDC00).contains(&cp) && self.peek() == Some('\\') {
                                self.i += 2; // consume `\u`
                                let lo: String =
                                    (0..4).filter_map(|k| self.b.get(self.i + k)).collect();
                                self.i += 4;
                                let lo = u32::from_str_radix(&lo, 16).unwrap_or(0);
                                let combined =
                                    0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                s.push(char::from_u32(combined).unwrap_or('\u{fffd}'));
                            } else {
                                s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                        }
                        other => s.push(other),
                    }
                }
                Some(c) => {
                    s.push(c);
                    self.i += 1;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.peek() == Some('.') {
            self.i += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.i += 1;
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let text: String = self.b[start..self.i].iter().collect();
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("invalid number `{}`", text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_an_object() {
        let src = r#"{"a":1,"b":"two","c":[1,2,3],"d":{"e":true},"f":null}"#;
        let v = parse(src).unwrap();
        assert_eq!(v.get("a").unwrap().as_i64(), Some(1));
        assert_eq!(v.get("b").unwrap().as_str(), Some("two"));
        assert_eq!(v.get("c").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(v.path("d.e"), Some(&Json::Bool(true)));
        assert_eq!(parse(&v.to_string()).unwrap(), v);
    }

    #[test]
    fn handles_escapes_both_ways() {
        let v = parse(r#""line\nbreak \"quoted\" \u00e9""#).unwrap();
        assert_eq!(v.as_str(), Some("line\nbreak \"quoted\" é"));
        let out = v.to_string();
        assert!(out.contains("\\n"));
        assert_eq!(parse(&out).unwrap(), v);
    }

    #[test]
    fn parses_negative_and_exponent_numbers() {
        assert_eq!(parse("-4.5e2").unwrap(), Json::Num(-450.0));
    }

    #[test]
    fn dotted_paths_reach_nested_values() {
        let v = parse(r#"{"params":{"textDocument":{"uri":"file:///a.kl"}}}"#).unwrap();
        assert_eq!(
            v.path("params.textDocument.uri").and_then(|x| x.as_str()),
            Some("file:///a.kl")
        );
    }

    #[test]
    fn integers_serialize_without_a_decimal_point() {
        assert_eq!(Json::int(3).to_string(), "3");
    }

    #[test]
    fn empty_containers_survive() {
        assert_eq!(parse("{}").unwrap(), Json::obj());
        assert_eq!(parse("[]").unwrap(), Json::Arr(vec![]));
    }
}
