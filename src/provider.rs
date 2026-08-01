//! The provider surface — the model-agnostic interface a backend implements.
//!
//! This crate names no specific provider. It defines the [`Provider`] trait (the
//! interface) and the [`export_provider!`] macro that wraps any `impl Provider`
//! into a C-ABI dynamic library (`cdylib` — a `.so`) the caller loads at runtime.
//! Anthropic, OpenAI, and the local llama backend are each their own package
//! implementing this trait; adding a provider ships a new package and never
//! touches this crate.
//!
//! ## This ABI does not reach a Jade program
//!
//! The Jade compiler loads a provider through its own native-package machinery —
//! `jade_pkg_init`, then a binding called `infer` — not through the symbols below.
//! It dropped its dependency on this crate's Rust half in jade v1.1.30. Implementing
//! [`Provider`] and adding [`export_provider!`] therefore produces a library that
//! every *other* host can load and that the compiler cannot, which is exactly the
//! trap `dovata` fell into. See the crate-level docs for the full account.
//!
//! Serving a Jade program means exporting `jade_pkg_init` as well. Both entry points
//! can sit in one library over the same `Provider` impl, so this is an addition
//! rather than a replacement.
//!
//! ## Why the boundary passes bytes, not types
//!
//! The dynamic boundary carries the protocol's own wire bytes — request JSON in,
//! encoded [`Frame`] bytes out — never rich Rust types. That keeps the exported
//! symbol set tiny and, crucially, stable across protocol changes: a new frame
//! type or request field does not change the ABI (only [`PROVIDER_ABI_VERSION`]
//! tracks the ABI itself). A provider could therefore be authored in any language
//! that speaks C, not only Rust.
//!
//! ## Exported symbols (the ABI)
//!
//! [`export_provider!`] emits these `extern "C"` functions:
//!
//! ```text
//! ovata_provider_abi_version()      -> u32
//! ovata_provider_protocol_version() -> u32
//! ovata_provider_name_ptr()         -> *const u8   // UTF-8, static
//! ovata_provider_name_len()         -> usize
//! ovata_provider_new(cfg_ptr, cfg_len)                 -> *mut c_void  // null = failed
//! ovata_provider_infer(handle, req_ptr, req_len, cb, cb_ctx) -> i32    // 0 = ok
//! ovata_provider_free(handle)
//! ```
//!
//! ## The two version numbers, and why both are read before loading
//!
//! [`PROVIDER_ABI_VERSION`] answers "can I call this library at all" — the symbol set
//! and their signatures. [`crate::PROTOCOL_VERSION`] answers "will we understand each
//! other" — the request and frame payloads those symbols carry. They move
//! independently, which is the point: adding a request field bumps neither, adding a
//! frame type bumps only the protocol, and changing a C signature bumps only the ABI.
//!
//! Both are exported so a caller can decide at `dlopen` time. The protocol version was
//! previously only discoverable by making a health request and reading the reply, which
//! meant a mismatch surfaced as strange behaviour on a real request rather than as a
//! refusal to load. A caller that reads both up front can say exactly what is wrong
//! before anything is attempted.
//!
//! `ovata_provider_protocol_version` is additive, so a library built before it existed
//! simply will not export it. Treat the symbol as missing rather than fatal, and fall
//! back to the health report.

use core::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::request::InferenceRequest;
use crate::response::Frame;

/// Version of the provider C-ABI. A caller refuses to load a `.so` whose
/// `ovata_provider_abi_version()` differs. Bump on any change to the exported
/// symbol set or their signatures — NOT on wire-protocol changes, which the
/// byte-oriented boundary already absorbs (see [`crate::PROTOCOL_VERSION`]).
pub const PROVIDER_ABI_VERSION: u32 = 1;

/// Return codes from the FFI shims.
pub const OK: i32 = 0;
/// A required pointer argument was null.
pub const ERR_NULL_ARG: i32 = -1;
/// The request bytes did not decode as an `InferenceRequest`.
pub const ERR_BAD_REQUEST: i32 = -2;
/// The provider panicked (unwinding was caught at the boundary — crossing it is UB).
pub const ERR_PANIC: i32 = -3;

/// Where a provider streams its response frames.
///
/// The host supplies the sink; the provider calls [`emit`](FrameSink::emit) once
/// per frame, in order (`Token*` → `Done`, or `Error`). The host has already sent
/// the leading `Meta` frame carrying the provider name, so a provider must NOT
/// emit `Meta` itself.
pub trait FrameSink {
    fn emit(&mut self, frame: Frame);

    /// Emit `text` as one or more `Token` frames, each guaranteed to fit the wire
    /// format's per-frame length. A single frame's payload is length-prefixed with
    /// a `u16`, so it must be under 64 KiB — [`Frame::encode`] would otherwise
    /// silently truncate it. This splits `text` on UTF-8 char boundaries into
    /// safely-sized `Token` frames. Non-streaming providers (which have the whole
    /// completion in hand) should emit through this rather than one giant `Token`.
    fn emit_text(&mut self, text: &str) {
        // Conservative cap, well under u16::MAX (65_535), leaving header headroom.
        const MAX: usize = 60_000;
        let mut buf = String::new();
        for ch in text.chars() {
            if buf.len() + ch.len_utf8() > MAX {
                self.emit(Frame::Token(std::mem::take(&mut buf)));
            }
            buf.push(ch);
        }
        if !buf.is_empty() {
            self.emit(Frame::Token(buf));
        }
    }
}

/// A model-agnostic inference backend.
///
/// Implement this, add [`export_provider!`]`(YourType)` at your crate root, and
/// build as a `cdylib` to produce a loadable provider `.so`.
pub trait Provider: Send + Sync + 'static {
    /// Stable, human-readable provider name, surfaced verbatim in the `Meta`
    /// frame (e.g. `"anthropic"`, `"openai"`, `"dovata"`).
    const NAME: &'static str;

    /// Build the provider from opaque config bytes the host passes through
    /// (typically JSON from the caller's own config). `Err` aborts loading; the
    /// message is for the provider's own logging — only success/failure crosses
    /// the ABI (a failed load surfaces as a null handle).
    fn from_config(config: &[u8]) -> Result<Self, String>
    where
        Self: Sized;

    /// Serve one request, streaming frames to `sink`: `Token`s then `Done` on
    /// success, or a single `Error` on failure. Do not emit `Meta`.
    fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink);
}

/// The C callback the host passes to receive each encoded frame's bytes.
/// `ctx` is the host's opaque pointer, echoed back unchanged. The bytes are only
/// valid for the duration of the call; the host must copy anything it keeps.
pub type FrameCallback = extern "C" fn(ctx: *mut c_void, frame_ptr: *const u8, frame_len: usize);

// A FrameSink that forwards each frame's wire bytes to a C callback.
struct CallbackSink {
    cb: FrameCallback,
    ctx: *mut c_void,
}

impl FrameSink for CallbackSink {
    fn emit(&mut self, frame: Frame) {
        let bytes = frame.encode();
        (self.cb)(self.ctx, bytes.as_ptr(), bytes.len());
    }
}

// ---- FFI glue invoked by the export_provider! macro. Not called directly. ----
//
// Each shim catches unwinding: a panic must never cross the C boundary (that is
// UB). Panics degrade to a null handle / error code instead.

/// # Safety
/// `config_ptr` must be null or valid for reads of `config_len` bytes.
#[doc(hidden)]
pub unsafe fn __ffi_new<P: Provider>(config_ptr: *const u8, config_len: usize) -> *mut c_void {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let config: &[u8] = if config_ptr.is_null() || config_len == 0 {
            &[]
        } else {
            // SAFETY: caller contract — valid for config_len bytes.
            unsafe { std::slice::from_raw_parts(config_ptr, config_len) }
        };
        P::from_config(config)
    }));
    match outcome {
        Ok(Ok(provider)) => Box::into_raw(Box::new(provider)) as *mut c_void,
        // construction error or panic → null handle (host treats as load failure)
        _ => std::ptr::null_mut(),
    }
}

/// # Safety
/// `handle` must be a live pointer from `__ffi_new::<P>`; `req_ptr` valid for `req_len` bytes.
#[doc(hidden)]
pub unsafe fn __ffi_infer<P: Provider>(
    handle: *mut c_void,
    req_ptr: *const u8,
    req_len: usize,
    cb: FrameCallback,
    cb_ctx: *mut c_void,
) -> i32 {
    if handle.is_null() || req_ptr.is_null() {
        return ERR_NULL_ARG;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: handle came from Box::into_raw(Box::<P>) and is still live; P: Sync
        // makes shared access sound.
        let provider = unsafe { &*(handle as *const P) };
        // SAFETY: caller contract — valid for req_len bytes.
        let req_bytes = unsafe { std::slice::from_raw_parts(req_ptr, req_len) };
        let req: InferenceRequest = match serde_json::from_slice(req_bytes) {
            Ok(r) => r,
            Err(_) => return ERR_BAD_REQUEST,
        };
        let mut sink = CallbackSink { cb, ctx: cb_ctx };
        provider.infer(&req, &mut sink);
        OK
    }));
    outcome.unwrap_or(ERR_PANIC)
}

/// # Safety
/// `handle` must come from `__ffi_new::<P>` and be freed at most once.
#[doc(hidden)]
pub unsafe fn __ffi_free<P: Provider>(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: reconstitute the Box we leaked in __ffi_new and drop it. Catch a
    // panicking Drop so it cannot cross the boundary.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(handle as *mut P) });
    }));
}

/// Export an `impl Provider` as a loadable provider `.so`.
///
/// Invoke once at the crate root of a `cdylib` crate:
/// ```ignore
/// ovata_infer_protocol::export_provider!(MyProvider);
/// ```
#[macro_export]
macro_rules! export_provider {
    ($ty:ty) => {
        #[no_mangle]
        pub extern "C" fn ovata_provider_abi_version() -> u32 {
            $crate::provider::PROVIDER_ABI_VERSION
        }
        #[no_mangle]
        pub extern "C" fn ovata_provider_protocol_version() -> u32 {
            $crate::PROTOCOL_VERSION
        }
        #[no_mangle]
        pub extern "C" fn ovata_provider_name_ptr() -> *const u8 {
            <$ty as $crate::provider::Provider>::NAME.as_ptr()
        }
        #[no_mangle]
        pub extern "C" fn ovata_provider_name_len() -> usize {
            <$ty as $crate::provider::Provider>::NAME.len()
        }
        /// # Safety: `config_ptr` null or valid for `config_len` bytes.
        #[no_mangle]
        pub unsafe extern "C" fn ovata_provider_new(
            config_ptr: *const u8,
            config_len: usize,
        ) -> *mut ::core::ffi::c_void {
            unsafe { $crate::provider::__ffi_new::<$ty>(config_ptr, config_len) }
        }
        /// # Safety: `handle` from ovata_provider_new; `req_ptr` valid for `req_len`.
        #[no_mangle]
        pub unsafe extern "C" fn ovata_provider_infer(
            handle: *mut ::core::ffi::c_void,
            req_ptr: *const u8,
            req_len: usize,
            cb: $crate::provider::FrameCallback,
            cb_ctx: *mut ::core::ffi::c_void,
        ) -> i32 {
            unsafe {
                $crate::provider::__ffi_infer::<$ty>(handle, req_ptr, req_len, cb, cb_ctx)
            }
        }
        /// # Safety: `handle` from ovata_provider_new, freed at most once.
        #[no_mangle]
        pub unsafe extern "C" fn ovata_provider_free(handle: *mut ::core::ffi::c_void) {
            unsafe { $crate::provider::__ffi_free::<$ty>(handle) }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial provider that echoes the prompt as one token then completes.
    struct Echo;
    impl Provider for Echo {
        const NAME: &'static str = "echo";
        fn from_config(_config: &[u8]) -> Result<Self, String> {
            Ok(Echo)
        }
        fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) {
            sink.emit(Frame::Token(req.prompt.clone()));
            sink.emit(Frame::Done { tokens_used: 1 });
        }
    }

    // Export it — generates the ovata_provider_* symbols in the test binary.
    crate::export_provider!(Echo);

    // Host-side callback: push each frame's bytes into a Vec passed via ctx.
    extern "C" fn collect(ctx: *mut c_void, ptr: *const u8, len: usize) {
        // SAFETY: ctx is the &mut Vec we pass below; ptr/len valid for this call.
        let sink = unsafe { &mut *(ctx as *mut Vec<Vec<u8>>) };
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        sink.push(bytes);
    }

    /// Drive the whole FFI path in-process: new → infer → free, decoding the
    /// frames that came back through the C callback.
    #[test]
    fn ffi_roundtrip_through_exported_symbols() {
        assert_eq!(ovata_provider_abi_version(), PROVIDER_ABI_VERSION);
        assert_eq!(ovata_provider_protocol_version(), crate::PROTOCOL_VERSION);

        // SAFETY: name symbols return a static &str's ptr/len.
        let name =
            unsafe { std::slice::from_raw_parts(ovata_provider_name_ptr(), ovata_provider_name_len()) };
        assert_eq!(name, b"echo");

        // SAFETY: null config is allowed (Echo ignores it).
        let handle = unsafe { ovata_provider_new(std::ptr::null(), 0) };
        assert!(!handle.is_null());

        let req = InferenceRequest { prompt: "hi".into(), ..Default::default() };
        let body = req.encode_body().unwrap();
        let mut frames: Vec<Vec<u8>> = Vec::new();
        // SAFETY: live handle, body valid for its len, collect matches FrameCallback.
        let rc = unsafe {
            ovata_provider_infer(
                handle,
                body.as_ptr(),
                body.len(),
                collect,
                &mut frames as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, OK);
        assert_eq!(frames.len(), 2);

        let (f0, _) = Frame::decode(&frames[0]).unwrap();
        assert!(matches!(f0, Frame::Token(ref t) if t == "hi"));
        let (f1, _) = Frame::decode(&frames[1]).unwrap();
        assert!(matches!(f1, Frame::Done { tokens_used: 1 }));

        // SAFETY: freeing the handle exactly once.
        unsafe { ovata_provider_free(handle) };
    }

    // A minimal in-memory sink for exercising default trait methods.
    struct VecSink(Vec<Frame>);
    impl FrameSink for VecSink {
        fn emit(&mut self, frame: Frame) {
            self.0.push(frame);
        }
    }

    #[test]
    fn emit_text_chunks_under_the_frame_limit_and_reassembles() {
        // A payload larger than one frame can hold, with a multi-byte char at the
        // boundary to prove we split on char boundaries, not bytes.
        let text = "é".repeat(50_000); // 100_000 bytes
        let mut sink = VecSink(Vec::new());
        sink.emit_text(&text);

        assert!(sink.0.len() >= 2, "should have split into multiple frames");
        let mut reassembled = String::new();
        for f in &sink.0 {
            match f {
                Frame::Token(t) => {
                    assert!(t.len() <= 60_000, "each token frame stays under the cap");
                    reassembled.push_str(t);
                }
                other => panic!("expected only Token frames, got {other:?}"),
            }
        }
        assert_eq!(reassembled, text);
    }

    #[test]
    fn emit_text_empty_produces_no_frames() {
        let mut sink = VecSink(Vec::new());
        sink.emit_text("");
        assert!(sink.0.is_empty());
    }

    #[test]
    fn null_handle_infer_is_rejected() {
        let cb: FrameCallback = collect;
        let mut sink: Vec<Vec<u8>> = Vec::new();
        // SAFETY: deliberately passing a null handle — must return ERR_NULL_ARG, not deref.
        let rc = unsafe {
            ovata_provider_infer(
                std::ptr::null_mut(),
                b"{}".as_ptr(),
                2,
                cb,
                &mut sink as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, ERR_NULL_ARG);
    }
}
