//! Eager model execution over scheduler-prepared batches.
//!
//! `GraphRunner` is deliberately absent from this migration step. Decode uses
//! the same eager path as prefill until CUDA/NPU graph capture is ported.

use std::fmt;

use tch::{Device, TchError, Tensor, no_grad};

/// Identifies the scheduler phase that produced a [`Batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
}

/// Per-batch attention tensors passed through the model stack.
///
/// The fields mirror mini-sglang's `AttentionMetadata`. Attention kernels are
/// migrated separately, so this is intentionally a transport type.
#[derive(Debug, Default)]
pub struct AttentionMetadata {
    pub write_loc: Option<Tensor>,
    pub cu_seqlens_q: Option<Tensor>,
    pub prefix_lens: Option<Tensor>,
    pub block_table: Option<Tensor>,
    pub req_to_token: Option<Tensor>,
    pub cache_seqlens: Option<Tensor>,
    pub max_seqlen: Option<usize>,
}

/// A scheduler-prepared model invocation.
///
/// Scheduler and `BatchContext` migration will construct these tensors. The
/// runner intentionally does not create request/page-table metadata itself.
#[derive(Debug)]
pub struct Batch {
    pub phase: BatchPhase,
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub attention_metadata: Option<AttentionMetadata>,
    /// Last uncached-token positions for prefill LM-head gathering.
    pub logits_indices: Option<Tensor>,
}

impl Batch {
    pub fn prefill(
        input_ids: Tensor,
        positions: Tensor,
        attention_metadata: Option<AttentionMetadata>,
        logits_indices: Tensor,
    ) -> Self {
        Self {
            phase: BatchPhase::Prefill,
            input_ids,
            positions,
            attention_metadata,
            logits_indices: Some(logits_indices),
        }
    }

    pub fn decode(
        input_ids: Tensor,
        positions: Tensor,
        attention_metadata: Option<AttentionMetadata>,
    ) -> Self {
        Self {
            phase: BatchPhase::Decode,
            input_ids,
            positions,
            attention_metadata,
            logits_indices: None,
        }
    }
}

/// Boundary implemented by the future Rust model architecture.
pub trait ModelExecutor {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attention_metadata: Option<&AttentionMetadata>,
        logits_indices: Option<&Tensor>,
    ) -> Result<Tensor>;
}

#[derive(Debug)]
pub enum ModelRunnerError {
    InputPositionLengthMismatch { input_ids: usize, positions: usize },
    TensorOnWrongDevice { expected: Device, actual: Device },
    MissingPrefillLogitsIndices,
    Model(String),
    Torch(TchError),
}

impl fmt::Display for ModelRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputPositionLengthMismatch {
                input_ids,
                positions,
            } => write!(
                f,
                "input_ids 数量 ({input_ids}) 必须与 positions 数量 ({positions}) 相同"
            ),
            Self::TensorOnWrongDevice { expected, actual } => {
                write!(f, "Batch tensor 位于 {actual:?}，期望位于 {expected:?}")
            }
            Self::MissingPrefillLogitsIndices => {
                write!(f, "prefill Batch 必须提供 logits_indices")
            }
            Self::Model(message) => write!(f, "模型执行失败: {message}"),
            Self::Torch(error) => write!(f, "libtorch error: {error}"),
        }
    }
}

impl std::error::Error for ModelRunnerError {}

impl From<TchError> for ModelRunnerError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}

pub type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Owns a model executor and runs its eager forward path.
pub struct ModelRunner {
    model: Box<dyn ModelExecutor>,
    device: Device,
}

impl ModelRunner {
    pub fn new(model: Box<dyn ModelExecutor>, device: Device) -> Self {
        Self { model, device }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The single model-forward entry point retained for future GraphRunner use.
    pub fn run_model(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attention_metadata: Option<&AttentionMetadata>,
        logits_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.validate_tensors(input_ids, positions)?;
        no_grad(|| {
            self.model
                .forward(input_ids, positions, attention_metadata, logits_indices)
        })
    }

    /// Executes prefill or decode eagerly. Decode graph replay is intentionally
    /// skipped until `GraphRunner` is migrated.
    pub fn forward(&self, batch: &Batch) -> Result<Tensor> {
        let logits_indices = match batch.phase {
            BatchPhase::Prefill => Some(
                batch
                    .logits_indices
                    .as_ref()
                    .ok_or(ModelRunnerError::MissingPrefillLogitsIndices)?,
            ),
            BatchPhase::Decode => None,
        };
        self.run_model(
            &batch.input_ids,
            &batch.positions,
            batch.attention_metadata.as_ref(),
            logits_indices,
        )
    }

    /// Compatibility hook for Engine cleanup. There are no graphs to clear yet.
    pub fn clear_graphs(&mut self) {}

    fn validate_tensors(&self, input_ids: &Tensor, positions: &Tensor) -> Result<()> {
        if input_ids.numel() != positions.numel() {
            return Err(ModelRunnerError::InputPositionLengthMismatch {
                input_ids: input_ids.numel(),
                positions: positions.numel(),
            });
        }
        for tensor in [input_ids, positions] {
            if tensor.device() != self.device {
                return Err(ModelRunnerError::TensorOnWrongDevice {
                    expected: self.device,
                    actual: tensor.device(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    struct IndicesModel;

    impl ModelExecutor for IndicesModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _attention_metadata: Option<&AttentionMetadata>,
            logits_indices: Option<&Tensor>,
        ) -> Result<Tensor> {
            Ok(logits_indices
                .map(Tensor::shallow_clone)
                .unwrap_or_else(|| {
                    Tensor::zeros([input_ids.numel() as i64], (Kind::Int64, Device::Cpu))
                }))
        }
    }

    #[test]
    fn prefill_passes_logits_indices_to_the_model() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::prefill(
            Tensor::from_slice(&[10i64, 11, 12]),
            Tensor::from_slice(&[0i64, 1, 2]),
            None,
            Tensor::from_slice(&[2i64]),
        );

        assert_eq!(
            Vec::<i64>::try_from(&runner.forward(&batch).unwrap()).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn decode_does_not_pass_prefill_indices() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::decode(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[3i64, 4]),
            None,
        );

        assert_eq!(
            Vec::<i64>::try_from(&runner.forward(&batch).unwrap()).unwrap(),
            vec![0, 0]
        );
    }

    #[test]
    fn rejects_mismatched_input_and_position_lengths() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::decode(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[3i64]),
            None,
        );

        assert!(matches!(
            runner.forward(&batch),
            Err(ModelRunnerError::InputPositionLengthMismatch { .. })
        ));
    }
}
