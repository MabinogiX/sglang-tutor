use pyo3::{Bound, Python, types::PyAny};

use super::{KVCacheError, KVCacheLayout, KVCachePool, Result};

/// The CPU fallback budget used by mini-sglang when accelerator memory metrics
/// are unavailable.
pub const CPU_KV_CACHE_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Npu,
}

impl Device {
    fn torch_name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Npu => "npu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KVCacheServerConfig {
    pub page_size: usize,
    pub max_running_req: usize,
    pub max_seq_len: usize,
    pub memory_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCacheModelConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KVCacheAllocationConfig {
    pub server: KVCacheServerConfig,
    pub model: KVCacheModelConfig,
}

/// Computes the pool size and asks Python/Torch to allocate the backing tensor.
pub struct KVCacheAllocator {
    config: KVCacheAllocationConfig,
}

impl KVCacheAllocator {
    pub fn new(config: KVCacheAllocationConfig) -> Result<Self> {
        let server = config.server;
        let model = config.model;
        if server.page_size == 0
            || server.max_running_req == 0
            || server.max_seq_len == 0
            || model.num_layers == 0
            || model.num_kv_heads == 0
            || model.head_dim == 0
        {
            return Err(KVCacheError::InvalidArgument(
                "KV-cache allocation dimensions must be greater than zero".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&server.memory_ratio) {
            return Err(KVCacheError::InvalidArgument(
                "memory_ratio must be in the range 0.0..=1.0".to_owned(),
            ));
        }
        Ok(Self { config })
    }

    pub fn available_memory(&self, device: Device) -> Result<usize> {
        match device {
            Device::Cpu => Ok(CPU_KV_CACHE_BYTES),
            Device::Cuda | Device::Npu => Err(KVCacheError::NotImplemented(
                "CUDA/NPU 空闲显存查询（依赖尚未迁移的 Python device 工具）",
            )),
        }
    }

    /// Number of pages that fit the memory budget, capped by scheduler demand.
    pub fn num_pages(
        &self,
        available: usize,
        num_kv_heads_per_rank: usize,
        dtype_itemsize: usize,
    ) -> Result<usize> {
        if num_kv_heads_per_rank == 0 || dtype_itemsize == 0 {
            return Err(KVCacheError::InvalidArgument(
                "num_kv_heads_per_rank and dtype_itemsize must be greater than zero".to_owned(),
            ));
        }

        let server = self.config.server;
        let model = self.config.model;
        let bytes_per_page = 2usize
            .checked_mul(model.num_layers)
            .and_then(|bytes| bytes.checked_mul(server.page_size))
            .and_then(|bytes| bytes.checked_mul(num_kv_heads_per_rank))
            .and_then(|bytes| bytes.checked_mul(model.head_dim))
            .and_then(|bytes| bytes.checked_mul(dtype_itemsize))
            .ok_or_else(|| KVCacheError::InvalidArgument("bytes per page overflow".to_owned()))?;
        let pages_that_fit = usize::max(1, available / bytes_per_page);
        let pages_per_request = server.max_seq_len / server.page_size + 1;
        let max_pages_needed = server.max_running_req.saturating_mul(pages_per_request);

        Ok(pages_that_fit.min(max_pages_needed))
    }

    /// Allocate a Torch-backed pool. `dtype` must be a Python `torch.dtype`.
    pub fn allocate(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        dtype_itemsize: usize,
        device: Device,
        tp_size: usize,
    ) -> Result<KVCachePool> {
        if tp_size == 0 {
            return Err(KVCacheError::InvalidArgument(
                "tp_size must be greater than zero".to_owned(),
            ));
        }

        let heads_per_rank = usize::max(1, self.config.model.num_kv_heads / tp_size);
        let available = self.available_memory(device)?;
        let num_pages = self.num_pages(available, heads_per_rank, dtype_itemsize)?;
        let layout = KVCacheLayout::new(
            self.config.model.num_layers,
            num_pages,
            self.config.server.page_size,
            heads_per_rank,
            self.config.model.head_dim,
        )?;
        KVCachePool::new(py, layout, dtype, device.torch_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocator() -> KVCacheAllocator {
        KVCacheAllocator::new(KVCacheAllocationConfig {
            server: KVCacheServerConfig {
                page_size: 16,
                max_running_req: 8,
                max_seq_len: 128,
                memory_ratio: 0.9,
            },
            model: KVCacheModelConfig {
                num_layers: 2,
                num_kv_heads: 8,
                head_dim: 64,
            },
        })
        .unwrap()
    }

    #[test]
    fn caps_pages_by_scheduler_demand() {
        // 1 GiB can fit more pages than 8 requests * (128 / 16 + 1) pages.
        assert_eq!(allocator().num_pages(1024 * 1024 * 1024, 8, 4).unwrap(), 72);
    }

    #[test]
    fn preserves_a_minimum_of_one_page() {
        assert_eq!(allocator().num_pages(1, 8, 4).unwrap(), 1);
    }

    #[test]
    fn accelerator_memory_lookup_is_explicitly_deferred() {
        assert!(matches!(
            allocator().available_memory(Device::Cuda),
            Err(KVCacheError::NotImplemented(_))
        ));
    }
}
