//! ovata-infer-protocol — the wire protocol for local LLM inference.
//!
//! ## What this crate is for
//!
//! Two codebases speak this protocol: the inference engine that implements it, and
//! the Jade language that calls it. It was written out separately in each —
//! four times over, once the language's compiled runtime and its VM are counted
//! apart — and the copies drifted, in ways that made `jade run` and `jade build`
//! behave differently against the same engine.
//!
//! This crate exists so there is one definition. It lives in its own repository
//! rather than inside either consumer because of visibility: the language repo
//! is public and the engine repo is not, and a public repo cannot depend on a
//! private one without breaking `cargo build` for everyone who clones it. The
//! engine takes this as a submodule; the language takes it as a git dependency.
//!
//! **Keep the dependency list to serde.** Neither consumer can see the other,
//! so anything added here is something both must build.
//!
//! ## Transport
//!
//! There is none. This protocol is a direct link: the caller loads a provider library
//! and calls it.
//!
//! It used to describe a Unix domain socket, with the engine running as a daemon behind
//! it. That was the wrong shape. Inference here is almost entirely GPU-bound work, so a
//! socket added an IPC hop per request, a second process to install and supervise, and a
//! path that every consumer resolved slightly differently. The engine already
//! implemented [`Provider`], which is all the C ABI needs, so the daemon was removable
//! without touching the engine. What is left is the ordinary shape for reaching a GPU:
//! the caller links a library, the library talks to the driver.
//!
//! ```text
//!   a host that is not the Jade compiler
//!     dlopen  → libdovata.so                 (once)
//!     ovata_provider_new(config)  → handle    (loads the model; null on failure)
//!     for each request:
//!       ovata_provider_infer(handle, request_json, callback, ctx)
//!         callback(ctx, frame_bytes, len)     ← once per frame, until Done or Error
//!     ovata_provider_free(handle)
//! ```
//!
//! [`export_provider!`] emits those four symbols for any `impl Provider`, so a provider
//! is a `cdylib` and nothing more. See [`provider`] for the ABI itself.
//!
//! ## Which entry point a Jade program uses — and it is not the one above
//!
//! The Jade compiler does **not** call `ovata_provider_*`. It loads a provider through
//! its own native-package machinery: `dlopen`, then `jade_pkg_init` to collect
//! name-to-function-pointer bindings, then a binding called `infer` taking an
//! [`InferRequest`](../jade/infer.jde) struct and returning an array of frames. The
//! same loader serves `jade run` and a compiled binary.
//!
//! This paragraph exists because the diagram above used to name "a compiled Jade
//! program" as its caller, and had done since the daemon was removed. It was not true.
//! The compiler dropped its dependency on this crate's Rust half in jade v1.1.30 and
//! kept only `jade/infer.jde`, so the two sides agreed on every *message* and disagreed
//! on the *entry point* — with this file asserting an answer one of them had stopped
//! honouring. Both repos vendor this text and believed it. `dovata` was built to the
//! C ABI, the compiler looked for `jade_pkg_init`, and every `?p` failed at load.
//!
//! **The compiler's loader is the authority.** A provider that wants to serve a Jade
//! program exports `jade_pkg_init`, whatever language it is written in. The remote
//! providers get there by being Jade `--lib`s; a Rust provider like the engine exports
//! the same entry point directly, over the same `Provider` impl.
//!
//! `ovata_provider_*` keeps its meaning for every host that is not the compiler — it is
//! a plain C ABI over the same `Provider` trait, and both entry points can live in one
//! library. What changed is only the claim about who uses which.
//!
//! ## Encodings
//!
//! The two encodings below are the payload formats, not a transport. Across the ABI the
//! request is passed as bare JSON, because the FFI argument already carries its length,
//! and each response frame arrives as one callback invocation.
//!
//! [`InferenceRequest::encode`] and [`InferenceRequest::decode`] add and strip a 4-byte
//! length prefix. Nothing on the ABI path uses them; they are kept for a caller whose
//! transport is a byte stream and needs to find message boundaries itself. Frames keep
//! their own 3-byte header either way, so one decoder serves both cases.
//!
//! ## Request encoding
//!
//! ```text
//! [4 bytes LE: json_len][json_len bytes: JSON InferenceRequest]
//! ```
//!
//! ## Response frame encoding
//!
//! ```text
//! [1 byte: type][2 bytes LE: payload_len][payload_len bytes: payload]
//!
//! 0x01  TOKEN   UTF-8 token text
//! 0x02  DONE    8-byte LE u64 total_tokens_used
//! 0x03  ERROR   UTF-8 error message
//! 0x04  META    UTF-8 provider name (sent first)
//! 0x05  JSON    UTF-8 structured payload (e.g. the health report)
//! ```
//!
//! ## Request prompt format — the two-layer message contract
//!
//! `InferenceRequest.prompt` is interpreted one of two ways. The parser and builder
//! both live in [`envelope`] — this crate is the authority, and neither the engine
//! nor the language should spell the marker key itself:
//!
//! * **Plain string** — wrapped as a single `user` message.
//! * **A "messages" envelope** — a JSON object with `"_ovata_infer_protocol": "messages"`,
//!   rendered through the model's own GGUF chat template:
//!
//!   ```json
//!   {
//!     "_ovata_infer_protocol": "messages",
//!     "system": "optional system prompt",
//!     "messages": [
//!       {"role": "user",      "content": "What's the weather in SF?"},
//!       {"role": "assistant", "content": "<tool>{\"tool_name\":\"get_weather\",\"city\":\"SF\"}</tool>"},
//!       {"role": "tool",      "content": "26°C and sunny"}
//!     ]
//!   }
//!   ```
//!
//!   A `tool`-role message is **load-bearing**, and the role must be carried through as
//!   `tool` (NOT remapped to `user`). This crate passes roles through verbatim and
//!   assigns them no meaning; what a consumer does with one is a template concern. For
//!   the record, the engine wraps tool content in `<tool_response>…</tool_response>`
//!   before rendering, because Qwen3's embedded template only associates a tool result
//!   with the preceding tool call when it sees `<|im_start|>tool`. That wrapper is
//!   model-specific and deliberately lives there, not here — same reasoning as the
//!   anchor delimiters below.
//!
//! ## Anchored spans (model-agnostic)
//!
//! A span is requested by setting `anchor` (its opening delimiter), `stop_anchor` (its
//! closing delimiter), and a `grammar` constraining the body. With `keep_anchors = true`,
//! the engine makes the closing boundary observable in-band so a caller can delimit the
//! span by pure string parsing (see [`InferenceRequest::keep_anchors`]).
//!
//! The delimiter *strings* are deliberately **not** defined here: they are model-specific
//! (e.g. `<tool>…</tool>` for one model, a different convention for another) and belong in
//! a per-model profile owned by the language layer (`jade-model-profile`), not baked into
//! the wire protocol. The protocol only knows "there is an anchored span"; it never knows
//! the span means "tool call" or which tokens spell it.

pub mod envelope;
pub mod health;
pub mod provider;
pub mod request;
pub mod response;

pub use envelope::{Envelope, Message, MESSAGES_ENVELOPE_KEY};
pub use health::Health;
pub use provider::{FrameSink, Provider, PROVIDER_ABI_VERSION};
pub use request::{InferenceRequest, RequestDecodeError};
pub use response::{Frame, FrameError};

/// Protocol revision the provider and its callers agree on. Reported in
/// [`Health::protocol_version`]. Independent of any implementation's crate semver;
/// bump only when the wire format changes. Additive, default-tolerant changes
/// (new `#[serde(default)]` request fields, new frame types old clients can ignore)
/// do not require a bump.
/// Version 2 renamed the messages-envelope marker key from `_jade_protocol` to
/// `_ovata_infer_protocol`, with no compatibility path — a breaking change, hence
/// the bump.
/// Version 3 repurposed the `Meta` frame: it no longer reports the serving *model*
/// name but the *provider* name (an opaque string; the protocol enumerates no
/// providers — see [`provider`]). The wire *format* of `Meta` is unchanged (still a
/// UTF-8 string), but its meaning is not, so a v2 client that displayed it as a
/// model name is incompatible.
pub const PROTOCOL_VERSION: u32 = 3;
