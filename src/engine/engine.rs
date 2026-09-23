//! Engine orchestration for the Rust migration.
//!
//! Model assembly, scheduling, graph execution, and sampling have not yet
//! moved out of Python.  This module nevertheless owns the work that already
//! has Rust implementations: validating model input, normalizing configuration,
//! and allocating/releasing the paged KV cache.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use tch::{Device, Kind, Tensor};

use super::kvcache::{
    KVCacheAllocationConfig, KVCacheAllocator, KVCacheError, KVCacheModelConfig, KVCachePool,
    KVCacheServerConfig,
};
use super::sampling::{Sampler, SamplingError, SamplingParams};
use super::{Batch, ModelRunner, ModelRunnerError};

/// Server-side settings consumed by [`Engine`].
///
/// This is the Rust counterpart of mini-sglang's `ServerArgs`; fields that
/// only affect HTTP serving remain outside the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerArgs {
    pub model_path: PathBuf,
    pub tp_size: usize,
    pub memory_ratio: f64,
    pub max_running_req: usize,
    pub max_seq_len: usize,
    pub page_size: usize,
    pub dtype: String,
    pub attention_backend: String,
    pub trust_remote_code: bool,
}

impl ServerArgs {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tp_size: 1,
            memory_ratio: 0.9,
            max_running_req: 256,
            max_seq_len: 8192,
            page_size: 16,
            dtype: "auto".to_owned(),
            attention_backend: "fa".to_owned(),
            trust_remote_code: false,
        }
    }
}

/// Model architecture values required to size the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArgs {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
}

/// Engine construction and lifecycle failures.
#[derive(Debug)]
pub enum EngineError {
    InvalidArgument(String),
    ModelPathDoesNotExist(PathBuf),
    ModelPathIsNotDirectory(PathBuf),
    MissingModelConfig(PathBuf),
    KVCache(KVCacheError),
    ModelRunner(ModelRunnerError),
    Sampling(SamplingError),
    ModelRunnerNotAttached,
    Released,
    NotImplemented(&'static str),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid engine argument: {message}"),
            Self::ModelPathDoesNotExist(path) => write!(
                f,
                "模型路径不存在: {}。请传入包含 config.json 的目录。",
                path.display()
            ),
            Self::ModelPathIsNotDirectory(path) => {
                write!(f, "模型路径不是目录: {}", path.display())
            }
            Self::MissingModelConfig(path) => write!(
                f,
                "模型目录没有 config.json: {}。这不是 Hugging Face 模型目录。",
                path.display()
            ),
            Self::KVCache(error) => write!(f, "KV cache 初始化失败: {error}"),
            Self::ModelRunner(error) => write!(f, "模型前向失败: {error}"),
            Self::Sampling(error) => write!(f, "采样失败: {error}"),
            Self::ModelRunnerNotAttached => write!(f, "Engine 尚未绑定 ModelRunner"),
            Self::Released => write!(f, "Engine 已清理，不能再使用"),
            Self::NotImplemented(feature) => write!(f, "未实现: {feature}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<KVCacheError> for EngineError {
    fn from(error: KVCacheError) -> Self {
        Self::KVCache(error)
    }
}

impl From<SamplingError> for EngineError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<ModelRunnerError> for EngineError {
    fn from(error: ModelRunnerError) -> Self {
        Self::ModelRunner(error)
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Owns Rust-side engine state during the gradual migration.
pub struct Engine {
    server_args: ServerArgs,
    model_args: ModelArgs,
    tp_rank: usize,
    device: Device,
    kind: Kind,
    kv_cache_pool: Option<KVCachePool>,
    model_runner: Option<ModelRunner>,
    sampler: Sampler,
}

impl Engine {
    /// Builds a CPU, float32 engine. Use [`Self::with_runtime`] to select a
    /// different libtorch device or element type.
    pub fn new(server_args: ServerArgs, model_args: ModelArgs, tp_rank: usize) -> Result<Self> {
        Self::with_runtime(server_args, model_args, tp_rank, Kind::Float, Device::Cpu)
    }

    /// Validates configuration and allocates the libtorch-backed KV cache.
    pub fn with_runtime(
        mut server_args: ServerArgs,
        model_args: ModelArgs,
        tp_rank: usize,
        kind: Kind,
        device: Device,
    ) -> Result<Self> {
        validate_model_path(&server_args.model_path)?;
        validate_parallelism(server_args.tp_size, tp_rank)?;
        if server_args.tp_size > 1 {
            return Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"));
        }
        clamp_max_seq_len(&mut server_args, model_args);

        let allocator = KVCacheAllocator::new(KVCacheAllocationConfig {
            server: KVCacheServerConfig {
                page_size: server_args.page_size,
                max_running_req: server_args.max_running_req,
                max_seq_len: server_args.max_seq_len,
                memory_ratio: server_args.memory_ratio,
            },
            model: KVCacheModelConfig {
                num_layers: model_args.num_layers,
                num_kv_heads: model_args.num_kv_heads,
                head_dim: model_args.head_dim,
            },
        })?;
        let kv_cache_pool = allocator.allocate(kind, device, server_args.tp_size)?;

        Ok(Self {
            server_args,
            model_args,
            tp_rank,
            device,
            kind,
            kv_cache_pool: Some(kv_cache_pool),
            model_runner: None,
            sampler: Sampler,
        })
    }

    pub fn server_args(&self) -> &ServerArgs {
        &self.server_args
    }

    pub fn model_args(&self) -> ModelArgs {
        self.model_args
    }

    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn kv_cache_pool(&self) -> Result<&KVCachePool> {
        self.kv_cache_pool.as_ref().ok_or(EngineError::Released)
    }

    pub fn kv_cache_pool_mut(&mut self) -> Result<&mut KVCachePool> {
        self.kv_cache_pool.as_mut().ok_or(EngineError::Released)
    }

    /// Binds the migrated eager execution path after the Rust model is built.
    pub fn attach_model_runner(&mut self, model_runner: ModelRunner) -> Result<()> {
        self.ensure_live()?;
        if model_runner.device() != self.device {
            return Err(EngineError::InvalidArgument(format!(
                "ModelRunner device ({:?}) must match Engine device ({:?})",
                model_runner.device(),
                self.device
            )));
        }
        self.model_runner = Some(model_runner);
        Ok(())
    }

    pub fn model_runner(&self) -> Result<&ModelRunner> {
        self.ensure_live()?;
        self.model_runner
            .as_ref()
            .ok_or(EngineError::ModelRunnerNotAttached)
    }

    /// Releases Rust-owned accelerator memory. Calling it repeatedly is safe.
    pub fn cleanup(&mut self) {
        if let Some(model_runner) = &mut self.model_runner {
            model_runner.clear_graphs();
        }
        self.model_runner.take();
        self.kv_cache_pool.take();
    }

    /// Executes a scheduler-prepared batch through the attached ModelRunner.
    pub fn forward(&self, batch: &Batch) -> Result<Tensor> {
        self.ensure_live()?;
        Ok(self
            .model_runner
            .as_ref()
            .ok_or(EngineError::ModelRunnerNotAttached)?
            .forward(batch)?)
    }

    /// Samples one token per logits row using request-aligned parameters.
    pub fn sample(&self, logits: &Tensor, params: &[SamplingParams]) -> Result<Vec<i64>> {
        self.ensure_live()?;
        Ok(self.sampler.sample_batch(logits, params)?)
    }

    fn ensure_live(&self) -> Result<()> {
        if self.kv_cache_pool.is_none() {
            return Err(EngineError::Released);
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Clamps scheduler and cache sizing to the trained context window.
pub fn clamp_max_seq_len(server_args: &mut ServerArgs, model_args: ModelArgs) {
    if model_args.max_position_embeddings > 0
        && server_args.max_seq_len > model_args.max_position_embeddings
    {
        server_args.max_seq_len = model_args.max_position_embeddings;
    }
}

/// Fails early when `model_path` cannot be a local Hugging Face model directory.
pub fn validate_model_path(model_path: &Path) -> Result<()> {
    if !model_path.exists() {
        return Err(EngineError::ModelPathDoesNotExist(model_path.to_path_buf()));
    }
    if !model_path.is_dir() {
        return Err(EngineError::ModelPathIsNotDirectory(
            model_path.to_path_buf(),
        ));
    }
    if !model_path.join("config.json").is_file() {
        return Err(EngineError::MissingModelConfig(model_path.to_path_buf()));
    }
    Ok(())
}

fn validate_parallelism(tp_size: usize, tp_rank: usize) -> Result<()> {
    if tp_size == 0 {
        return Err(EngineError::InvalidArgument(
            "tp_size must be greater than zero".to_owned(),
        ));
    }
    if tp_rank >= tp_size {
        return Err(EngineError::InvalidArgument(format!(
            "tp_rank ({tp_rank}) must be smaller than tp_size ({tp_size})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::super::{AttentionMetadata, ModelExecutor};
    use super::*;

    fn model_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after UNIX epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sglang-rust-engine-{nonce}"));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("config.json"), "{}").unwrap();
        path
    }

    fn model_args() -> ModelArgs {
        ModelArgs {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_position_embeddings: 4,
        }
    }

    #[test]
    fn initializes_kv_cache_and_clamps_sequence_length() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 8;
        args.page_size = 2;

        let engine = Engine::new(args, model_args(), 0).unwrap();
        assert_eq!(engine.server_args().max_seq_len, 4);
        assert_eq!(engine.kv_cache_pool().unwrap().layout.num_pages, 3);
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn cleanup_releases_the_pool_idempotently() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, model_args(), 0).unwrap();

        engine.cleanup();
        engine.cleanup();
        assert!(matches!(engine.kv_cache_pool(), Err(EngineError::Released)));
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn validates_model_directory_and_parallelism() {
        assert!(matches!(
            validate_model_path(Path::new("/definitely/not/a/model")),
            Err(EngineError::ModelPathDoesNotExist(_))
        ));
        assert!(matches!(
            validate_parallelism(2, 2),
            Err(EngineError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tensor_parallelism_is_explicitly_deferred() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.tp_size = 2;

        assert!(matches!(
            Engine::new(args, model_args(), 0),
            Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"))
        ));
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn delegates_sampling_to_the_rust_sampler() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let engine = Engine::new(args, model_args(), 0).unwrap();
        let logits = Tensor::from_slice(&[0.1f32, 2.0, 0.2]).view([1, 3]);

        assert_eq!(
            engine
                .sample(&logits, &[SamplingParams::default()])
                .unwrap(),
            vec![1]
        );
        fs::remove_dir_all(model_dir).unwrap();
    }

    struct EchoModel;

    impl ModelExecutor for EchoModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _attention_metadata: Option<&AttentionMetadata>,
            _logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            Ok(input_ids.shallow_clone())
        }
    }

    #[test]
    fn delegates_forward_to_an_attached_model_runner() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, model_args(), 0).unwrap();
        engine
            .attach_model_runner(ModelRunner::new(Box::new(EchoModel), Device::Cpu))
            .unwrap();
        let batch = Batch::prefill(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[0i64, 1]),
            None,
            Tensor::from_slice(&[1i64]),
        );

        assert_eq!(
            Vec::<i64>::try_from(&engine.forward(&batch).unwrap()).unwrap(),
            vec![10, 11]
        );
        fs::remove_dir_all(model_dir).unwrap();
    }
}
