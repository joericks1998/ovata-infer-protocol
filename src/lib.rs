//! ovata-infer-protocol — the wire protocol for local LLM inference.
//!
//! ## What this crate is for
//!
//! Two codebases speak this protocol: the inference daemon that serves it, and
//! the Jade language that calls it. It was written out separately in each —
//! four times over, once the language's compiled runtime and its VM are counted
//! apart — and the copies drifted, in ways that made `jade run` and `jade build`
//! behave differently against the same daemon.
//!
//! This crate exists so there is one definition. It lives in its own repository
//! rather than inside either consumer because of visibility: the language repo
//! is public and the daemon repo is not, and a public repo cannot depend on a
//! private one without breaking `cargo build` for everyone who clones it. The
//! daemon takes this as a submodule; the language takes it as a git dependency.
//!
//! **Keep the dependency list to serde.** Neither consumer can see the other,
//! so anything added here is something both must build.
//!
//! ## Transport
//!
//! Communication flows through a Unix domain socket at `/run/jade/llm.sock`.
//! The client (typically the AOT-compiled program's C runtime) opens one
//! connection at first use and reuses it for the program's lifetime, sending
//! multiple requests sequentially. jade-tree loops on each connection,
//! processing one request at a time, until the client closes (EOF).
//!
//! ```text
//!   jade (compiler/tool)
//!     connect → /run/jade/llm.sock     (once per program)
//!     for each ?p in the program:
//!       write() → InferenceRequest    (length-prefixed JSON)
//!       read()  ← Frame stream         (binary frames until DONE or ERROR)
//!     close
//!   jade-tree (inference daemon)
//!     accept() ← connection
//!     loop:
//!       read()  ← InferenceRequest    (EOF → close connection)
//!       write() → Frame stream
//! ```
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
//! 0x04  META    UTF-8 model name (sent first)
//! 0x05  JSON    UTF-8 structured payload (e.g. the health report)
//! ```
//!
//! ## Request prompt format — the two-layer message contract
//!
//! `InferenceRequest.prompt` is interpreted one of two ways. The parser and builder
//! both live in [`envelope`] — this crate is the authority, and neither the daemon
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
//!   the record, the daemon wraps tool content in `<tool_response>…</tool_response>`
//!   before rendering, because Qwen3's embedded template only associates a tool result
//!   with the preceding tool call when it sees `<|im_start|>tool`. That wrapper is
//!   model-specific and deliberately lives there, not here — same reasoning as the
//!   anchor delimiters below.
//!
//! ## Anchored spans (model-agnostic)
//!
//! A span is requested by setting `anchor` (its opening delimiter), `stop_anchor` (its
//! closing delimiter), and a `grammar` constraining the body. With `keep_anchors = true`,
//! the daemon makes the closing boundary observable in-band so a client can delimit the
//! span by pure string parsing (see [`InferenceRequest::keep_anchors`]).
//!
//! The delimiter *strings* are deliberately **not** defined here: they are model-specific
//! (e.g. `<tool>…</tool>` for one model, a different convention for another) and belong in
//! a per-model profile owned by the language layer (`jade-model-profile`), not baked into
//! the wire protocol. The protocol only knows "there is an anchored span"; it never knows
//! the span means "tool call" or which tokens spell it.

pub mod envelope;
pub mod health;
pub mod request;
pub mod response;

pub use envelope::{Envelope, Message, MESSAGES_ENVELOPE_KEY};
pub use health::Health;
pub use request::{InferenceRequest, RequestDecodeError};
pub use response::{Frame, FrameError};

/// Wire-protocol revision the daemon and clients agree on. Reported in
/// [`Health::protocol_version`]. Independent of the daemon's release/crate semver;
/// bump only when the wire format changes. Additive, default-tolerant changes
/// (new `#[serde(default)]` request fields, new frame types old clients can ignore)
/// do not require a bump.
/// Version 2 renamed the messages-envelope marker key from `_jade_protocol` to
/// `_ovata_infer_protocol`, with no compatibility path — a breaking change, hence
/// the bump.
pub const PROTOCOL_VERSION: u32 = 2;
