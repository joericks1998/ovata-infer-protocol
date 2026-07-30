# ovata-infer-protocol

The contract between a caller and an inference provider: the request and
response shapes, in both languages that need them.

There are two halves, and they are used at different layers.

- **`src/` (Rust)** — the request/response types and frame codec for a *stream*
  transport, where the two sides are separate processes.
- **`jade/` (Jade)** — `InferRequest` and the response frames, for a provider
  package the caller loads in-process and calls directly. No bytes are serialized
  on this path; each value crosses the native ABI whole, carrying its type name
  so the receiver can tell it from anything else shaped like it.

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
- **Jade language** — consumes it as a git submodule, for `jade/infer.jde`. It
  dropped the Rust crate dependency in v1.1.30, when the daemon socket was
  removed and inference became a direct call into a provider package.
- **Provider packages** — register `jade/` as a `[lib]` in their `jade.toml` and
  `use ovata::infer`, so the shapes they read and return are these definitions
  rather than copies of them.

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
| `jade/infer.jde` | `InferRequest`, and the `Token`/`Done`/`Error`/`Meta`/`Json` frames — both directions of the in-process call, as Jade structs across the native ABI |

## Keeping the Jade half honest

The Jade compiler cannot import a `.jde` into its own Rust and C sources, so it
carries a hand-written copy of these names — twice over, since its two engines
build the request in Rust and in C. A tripwire test in that repo
(`src/llm/tests.rs`) parses `jade/infer.jde` with the compiler's own parser and
fails the build on any difference, in either direction, for the request fields
and for every frame name. Rename something here and the language will not compile
until it follows.
