//! Paged KV-cache ownership and allocation.
//!
//! Page bookkeeping lives in Rust. The backing K/V tensors remain Python
//! `torch.Tensor` objects until the model and tensor runtime are migrated.

mod allocator;
mod error;
mod pool;
mod radix;

pub use allocator::{
    Device, KVCacheAllocationConfig, KVCacheAllocator, KVCacheModelConfig, KVCacheServerConfig,
};
pub use error::{KVCacheError, Result};
pub use pool::{BaseCacheHandle, CacheManager, KVCacheLayout, KVCachePool};
pub use radix::{RadixCacheManager, RadixNode};
