use pyo3::{
    Bound, Py, Python,
    types::{PyAny, PyAnyMethods, PyDict, PyDictMethods, PyModule},
};

use super::{KVCacheError, Result};

/// A request's complete KV page table.
///
/// When prefix caching is introduced, the leading `num_shared` page IDs are
/// borrowed from that cache. The remaining IDs belong to this request.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BaseCacheHandle {
    pub page_ids: Vec<usize>,
    pub cached_len: usize,
    pub num_shared: usize,
}

impl BaseCacheHandle {
    pub fn num_pages(&self) -> usize {
        self.page_ids.len()
    }
}

/// Interface that future naive/radix cache managers must implement.
///
/// The concrete cache managers are intentionally not migrated in this step.
pub trait CacheManager {
    fn match_prefix(&self, input_ids: &[i64]) -> Result<(usize, Vec<usize>)>;
    fn insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
    fn evict(&mut self, num_pages: usize) -> Result<Vec<usize>>;
    fn remove(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
    fn rollback_insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
}

/// Shape metadata for the backing `(K, V)` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCacheLayout {
    pub num_layers: usize,
    pub num_pages: usize,
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KVCacheLayout {
    pub fn new(
        num_layers: usize,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let layout = Self {
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
        };
        if [num_layers, num_pages, page_size, num_kv_heads, head_dim].contains(&0) {
            return Err(KVCacheError::InvalidArgument(
                "all KV-cache dimensions must be greater than zero".to_owned(),
            ));
        }
        Ok(layout)
    }

    fn tensor_shape(self) -> (usize, usize, usize, usize, usize, usize) {
        (
            2,
            self.num_layers,
            self.num_pages,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

/// Owns page allocation state and a Python-owned Torch buffer.
pub struct KVCachePool {
    pub layout: KVCacheLayout,
    buffer: Option<Py<PyAny>>,
    free_pages: Vec<usize>,
}

impl KVCachePool {
    /// Allocate the backing tensor with `torch.empty`.
    pub fn new(
        py: Python<'_>,
        layout: KVCacheLayout,
        dtype: &Bound<'_, PyAny>,
        device: &str,
    ) -> Result<Self> {
        let torch = PyModule::import(py, "torch")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", dtype)?;
        kwargs.set_item("device", device)?;
        let buffer = torch.call_method("empty", (layout.tensor_shape(),), Some(&kwargs))?;

        Ok(Self {
            layout,
            buffer: Some(buffer.unbind()),
            free_pages: (0..layout.num_pages).collect(),
        })
    }

    /// Create only the page allocator. It is useful for deterministic unit
    /// tests; requesting tensors from it returns an explicit 未实现 error.
    pub fn without_tensor(layout: KVCacheLayout) -> Self {
        Self {
            layout,
            buffer: None,
            free_pages: (0..layout.num_pages).collect(),
        }
    }

    /// Allocate pages in the same LIFO order as mini-sglang's Python pool.
    pub fn alloc(&mut self, num_pages: usize) -> Result<BaseCacheHandle> {
        let available = self.free_pages.len();
        if available < num_pages {
            return Err(KVCacheError::OutOfMemory {
                requested: num_pages,
                available,
            });
        }

        let page_ids = (0..num_pages)
            .map(|_| self.free_pages.pop().expect("checked free page count"))
            .collect();
        Ok(BaseCacheHandle {
            page_ids,
            ..Default::default()
        })
    }

    /// Return a handle's pages. Clearing the handle makes repeated frees safe.
    pub fn free(&mut self, handle: &mut BaseCacheHandle) {
        self.free_pages.append(&mut handle.page_ids);
    }

    /// Return pages owned by a future cache manager, such as a radix tree.
    pub fn free_pages_by_id(&mut self, page_ids: impl IntoIterator<Item = usize>) {
        self.free_pages.extend(page_ids);
    }

    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Return `(k_cache, v_cache)`, each shaped
    /// `(num_layers, num_pages, page_size, num_kv_heads, head_dim)`.
    pub fn get_all_kv_cache(&self, py: Python<'_>) -> Result<(Py<PyAny>, Py<PyAny>)> {
        let buffer = self.buffer.as_ref().ok_or(KVCacheError::NotImplemented(
            "无 Python/Torch buffer 的 KVCachePool 不能提供 K/V tensor",
        ))?;
        let buffer = buffer.bind(py);
        Ok((buffer.get_item(0)?.unbind(), buffer.get_item(1)?.unbind()))
    }

    /// Binding Rust-owned pools to Python attention modules awaits migration of
    /// the model layer and remains deliberately unsupported for this step.
    pub fn bind_layers(&self, _py: Python<'_>, _model: &Bound<'_, PyAny>) -> Result<()> {
        Err(KVCacheError::NotImplemented(
            "Python attention layer 的 KV-cache 绑定",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(num_pages: usize) -> KVCachePool {
        KVCachePool::without_tensor(KVCacheLayout::new(2, num_pages, 4, 4, 32).unwrap())
    }

    #[test]
    fn alloc_and_free_match_the_python_pool_contract() {
        let mut pool = pool(10);
        let mut first = pool.alloc(3).unwrap();
        let mut second = pool.alloc(2).unwrap();

        assert_eq!(first.page_ids, vec![9, 8, 7]);
        assert_eq!(pool.free_count(), 5);
        pool.free(&mut first);
        assert_eq!(pool.free_count(), 8);
        pool.free(&mut second);
        assert_eq!(pool.free_count(), 10);
        pool.free(&mut second);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn allocation_fails_without_mutating_the_pool() {
        let mut pool = pool(2);
        let error = pool.alloc(3).unwrap_err();

        assert!(matches!(
            error,
            KVCacheError::OutOfMemory {
                requested: 3,
                available: 2
            }
        ));
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn layout_rejects_zero_dimensions() {
        assert!(KVCacheLayout::new(0, 1, 1, 1, 1).is_err());
    }

    #[test]
    #[ignore = "requires Python/Torch from mini-sglang's virtual environment"]
    fn torch_backed_pool_exposes_k_and_v_slices() {
        Python::attach(|py| -> Result<()> {
            let torch = PyModule::import(py, "torch")?;
            let dtype = torch.getattr("float32")?;
            let layout = KVCacheLayout::new(2, 3, 4, 5, 6)?;
            let pool = KVCachePool::new(py, layout, &dtype, "cpu")?;
            let (k_cache, v_cache) = pool.get_all_kv_cache(py)?;

            let k_shape: Vec<usize> = k_cache.bind(py).getattr("shape")?.extract()?;
            let v_shape: Vec<usize> = v_cache.bind(py).getattr("shape")?.extract()?;
            assert_eq!(k_shape, vec![2, 3, 4, 5, 6]);
            assert_eq!(v_shape, vec![2, 3, 4, 5, 6]);
            Ok(())
        })
        .unwrap();
    }
}
