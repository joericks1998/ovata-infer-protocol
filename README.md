# ovata-infer-protocol

The wire protocol for local LLM inference: the request/response types and frame
codec spoken between a caller and an inference provider.

## Why it lives on its own

Two codebases speak this protocol — the engine that implements it, and the Jade
language that calls it. Each had written it out separately, and the copies had
drifted: a short `DONE` payload meant "zero tokens" to one and "malformed" to
the other, invalid UTF-8 passed through one and was rejected by the other, and
the ceilings bounding a runaway generation existed on only one side. Those are
behaviour differences between running a Jade program and compiling it.

It sits in its own repository rather than inside either consumer because of
repository visibility: the language repo is public, the engine repo is not, and
a public repo cannot depend on a private one without breaking `cargo build` for
anyone who clones it. Depending on a public third repo works in both directions.

- **Inference engine** — consumes this as a git submodule and a workspace member.
- **Jade language** — consumes it as a git dependency, pinned to a tag.

## Keep it small

Neither consumer can see the other, so every dependency added here is one that
both must build. `serde` and `serde_json` are the whole list, and that is
deliberate — nothing from either consumer's tree belongs in it.

## Versioning

Consumers pin a **tag**, never a branch, so a protocol change cannot silently
alter what a client sends. `PROTOCOL_VERSION` in `lib.rs` is separate from the
crate version: bump it only when the wire format itself changes. Additive,
default-tolerant changes — a new `#[serde(default)]` request field, a new frame
type older clients can ignore — do not require a bump.

## Layout

| File | Contents |
|---|---|
| `request.rs` | `InferenceRequest` — caller → provider, JSON (length-prefixed only for a stream transport) |
| `response.rs` | `Frame` — provider → caller, `[type][len][payload]`, plus the `tag` constants |
| `health.rs` | `Health` — the `health_only` report |
