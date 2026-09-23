//! Shared paged-KV attention state.

mod backend;
mod base;

pub use backend::{AttentionBackend, AttentionBackendKind, create_attention_backend};
pub use base::BaseAttention;
