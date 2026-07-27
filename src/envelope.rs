//! The messages envelope — the structured alternative to a plain-string prompt.
//!
//! This module exists because the envelope was previously a bare string literal
//! spelled independently in every consumer: the engine that parses it and the
//! language that builds it. That is the exact duplication this crate was created
//! to end, and the marker key is the part most likely to drift, because a typo in
//! it does not fail loudly — an unrecognized envelope is silently treated as a
//! plain prompt, and the model is handed raw JSON.
//!
//! Producers should call [`build`], consumers [`parse`]. Neither should spell
//! [`MESSAGES_ENVELOPE_KEY`] itself.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The marker key identifying a prompt as a messages envelope.
///
/// Renamed from `_jade_protocol` in protocol version 2; there is no compatibility
/// path, and a client still sending the old key gets its JSON treated as prompt text.
pub const MESSAGES_ENVELOPE_KEY: &str = "_ovata_infer_protocol";

/// The marker key's only recognized value.
pub const MESSAGES_ENVELOPE_VALUE: &str = "messages";

/// One conversation turn.
///
/// `role` is free-form on the wire (`user`, `assistant`, `system`, `tool`, …).
/// This crate does not interpret it: what a role *means* to a model is a template
/// concern owned by the consumer, not part of the wire contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A parsed messages envelope.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Envelope {
    /// The system prompt; empty when the envelope omitted it.
    pub system: String,
    /// The conversation turns, in order.
    pub messages: Vec<Message>,
}

/// Parse a prompt string as a messages envelope.
///
/// Returns `None` when the string is not one — a plain single-turn prompt that the
/// caller should wrap as a lone `user` turn. Malformed *turns* are tolerated rather
/// than rejected (a turn missing `role` defaults to `user`, one missing `content` to
/// empty) so a single bad turn cannot fail an otherwise valid request; a missing or
/// non-array `messages` field is still `None`.
pub fn parse(prompt: &str) -> Option<Envelope> {
    // fast reject: an envelope is always a JSON object, so anything not starting
    // with '{' skips the parse entirely.
    if !prompt.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(prompt).ok()?;
    // it is JSON — but is it ours? Any other JSON falls through to a plain prompt.
    if v.get(MESSAGES_ENVELOPE_KEY).and_then(|p| p.as_str()) != Some(MESSAGES_ENVELOPE_VALUE) {
        return None;
    }
    let system = v.get("system").and_then(|s| s.as_str()).unwrap_or("").to_owned();
    let raw = v.get("messages").and_then(|m| m.as_array())?;

    let messages = raw
        .iter()
        .map(|m| Message {
            role: m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_owned(),
            content: m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_owned(),
        })
        .collect();

    Some(Envelope { system, messages })
}

/// Build a messages envelope. The `system` field is omitted when empty.
pub fn build(system: &str, messages: &[Message]) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert(MESSAGES_ENVELOPE_KEY.to_owned(), Value::String(MESSAGES_ENVELOPE_VALUE.to_owned()));
    if !system.is_empty() {
        obj.insert("system".to_owned(), Value::String(system.to_owned()));
    }
    obj.insert("messages".to_owned(), serde_json::to_value(messages).unwrap_or(Value::Null));
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message { role: role.to_owned(), content: content.to_owned() }
    }

    /// The whole point of the module: what `build` writes, `parse` reads back.
    /// If these drift, the language and the engine disagree about the envelope.
    #[test]
    fn build_and_parse_round_trip() {
        let msgs = vec![msg("user", "hi"), msg("assistant", "hello"), msg("tool", "26°C")];
        let json = build("be terse", &msgs).to_string();

        let env = parse(&json).expect("built envelope must parse");
        assert_eq!(env.system, "be terse");
        assert_eq!(env.messages, msgs);
    }

    #[test]
    fn build_omits_empty_system() {
        let json = build("", &[msg("user", "hi")]);
        assert!(json.get("system").is_none());
        assert_eq!(parse(&json.to_string()).unwrap().system, "");
    }

    #[test]
    fn plain_prompts_are_not_envelopes() {
        assert!(parse("Hello, what's the weather?").is_none());
        assert!(parse("").is_none());
        // valid JSON, but not ours
        assert!(parse(r#"{"role":"user","content":"hi"}"#).is_none());
        // not valid JSON at all, despite the leading brace
        assert!(parse("{not json").is_none());
    }

    /// A hard break: version 1 clients get no envelope handling, by design.
    #[test]
    fn the_old_jade_marker_is_not_accepted() {
        let old = r#"{"_jade_protocol":"messages","messages":[{"role":"user","content":"hi"}]}"#;
        assert!(parse(old).is_none());
    }

    #[test]
    fn wrong_marker_value_is_rejected() {
        let json = format!(r#"{{"{MESSAGES_ENVELOPE_KEY}":"turns","messages":[]}}"#);
        assert!(parse(&json).is_none());
    }

    #[test]
    fn missing_messages_array_is_none() {
        let json = format!(r#"{{"{MESSAGES_ENVELOPE_KEY}":"messages","system":"s"}}"#);
        assert!(parse(&json).is_none());
        // present but not an array
        let json = format!(r#"{{"{MESSAGES_ENVELOPE_KEY}":"messages","messages":"nope"}}"#);
        assert!(parse(&json).is_none());
    }

    /// One malformed turn must not fail the request — it degrades to a default.
    #[test]
    fn malformed_turns_degrade_rather_than_reject() {
        let json = format!(
            r#"{{"{MESSAGES_ENVELOPE_KEY}":"messages","messages":[
                {{"content":"no role"}}, {{"role":"assistant"}}, {{"role":42,"content":7}}
            ]}}"#
        );
        let env = parse(&json).unwrap();
        assert_eq!(env.messages, vec![msg("user", "no role"), msg("assistant", ""), msg("user", "")]);
    }

    #[test]
    fn empty_messages_array_parses() {
        let env = parse(&build("s", &[]).to_string()).unwrap();
        assert!(env.messages.is_empty());
    }

    /// Turn order is conversation history; reordering changes meaning.
    #[test]
    fn turn_order_is_preserved() {
        let msgs: Vec<Message> = (0..5).map(|i| msg("user", &i.to_string())).collect();
        assert_eq!(parse(&build("", &msgs).to_string()).unwrap().messages, msgs);
    }
}
