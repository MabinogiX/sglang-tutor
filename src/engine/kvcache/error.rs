use std::fmt;

use pyo3::PyErr;

pub type Result<T> = std::result::Result<T, KVCacheError>;

#[derive(Debug)]
pub enum KVCacheError {
    InvalidArgument(String),
    OutOfMemory { requested: usize, available: usize },
    NotImplemented(&'static str),
    Python(PyErr),
}

impl fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid KV-cache argument: {message}"),
            Self::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "KV cache out of memory: requested {requested} pages, only {available} free"
            ),
            Self::NotImplemented(feature) => write!(f, "未实现: {feature}"),
            Self::Python(error) => write!(f, "Python/Torch error: {error}"),
        }
    }
}

impl std::error::Error for KVCacheError {}

impl From<PyErr> for KVCacheError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}
