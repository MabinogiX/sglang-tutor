//! Token sampling strategies backed by libtorch.

use std::fmt;

use tch::{Kind, TchError, Tensor};

/// Per-request generation settings, mirroring mini-sglang's `SamplingParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub ignore_eos: bool,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
            max_tokens: 1024,
        }
    }
}

impl SamplingParams {
    /// Applies the same safe defaults as mini-sglang's HTTP-facing config.
    pub fn normalized(self) -> Self {
        Self {
            max_tokens: self.max_tokens.max(1),
            top_p: if self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p) {
                self.top_p
            } else {
                1.0
            },
            ..self
        }
    }

    fn has_same_distribution(self, other: Self) -> bool {
        self.temperature.to_bits() == other.temperature.to_bits()
            && self.top_k == other.top_k
            && self.top_p.to_bits() == other.top_p.to_bits()
    }
}

#[derive(Debug)]
pub enum SamplingError {
    InvalidLogitsShape(Vec<i64>),
    ParamsLengthMismatch { logits_rows: usize, params: usize },
    Torch(TchError),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogitsShape(shape) => {
                write!(
                    f,
                    "logits 必须是二维 (num_reqs, vocab_size)，实际为 {shape:?}"
                )
            }
            Self::ParamsLengthMismatch {
                logits_rows,
                params,
            } => write!(
                f,
                "SamplingParams 数量 ({params}) 必须与 logits 行数 ({logits_rows}) 相同"
            ),
            Self::Torch(error) => write!(f, "libtorch sampling error: {error}"),
        }
    }
}

impl std::error::Error for SamplingError {}

impl From<TchError> for SamplingError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}

pub type Result<T> = std::result::Result<T, SamplingError>;

/// Samples one next-token ID per logits row.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sampler;

impl Sampler {
    /// Samples rows that all share one set of sampling parameters.
    pub fn sample(&self, logits: &Tensor, params: SamplingParams) -> Result<Tensor> {
        validate_logits(logits)?;
        let params = params.normalized();
        if params.temperature <= 0.0 {
            return Ok(logits.argmax(-1, false));
        }

        let mut filtered = logits / params.temperature;
        let vocab_size = filtered.size()[1];
        if params.top_k > 0 {
            let top_k = params.top_k.min(vocab_size);
            let (top_k_values, _) = filtered.topk(top_k, -1, true, true);
            let threshold = top_k_values.select(-1, top_k - 1).unsqueeze(-1);
            filtered = filtered.masked_fill(&filtered.lt_tensor(&threshold), f64::NEG_INFINITY);
        }
        if params.top_p < 1.0 {
            filtered = apply_top_p(&filtered, params.top_p);
        }

        Ok(filtered
            .softmax(-1, Kind::Float)
            .multinomial(1, false)
            .squeeze_dim(-1))
    }

    /// Samples a heterogeneous batch while preserving its original row order.
    /// Requests with equal temperature/top-k/top-p settings share one libtorch
    /// sampling call, matching mini-sglang's batching behavior.
    pub fn sample_batch(
        &self,
        logits: &Tensor,
        params_list: &[SamplingParams],
    ) -> Result<Vec<i64>> {
        validate_logits(logits)?;
        let num_rows = usize::try_from(logits.size()[0]).expect("validated non-negative shape");
        if num_rows != params_list.len() {
            return Err(SamplingError::ParamsLengthMismatch {
                logits_rows: num_rows,
                params: params_list.len(),
            });
        }
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        if params_list.iter().all(|params| params.temperature <= 0.0) {
            return Vec::<i64>::try_from(&logits.argmax(-1, false)).map_err(Into::into);
        }

        let mut groups: Vec<(SamplingParams, Vec<i64>)> = Vec::new();
        for (row, params) in params_list.iter().copied().enumerate() {
            let params = params.normalized();
            if let Some((_, rows)) = groups
                .iter_mut()
                .find(|(group_params, _)| group_params.has_same_distribution(params))
            {
                rows.push(row as i64);
            } else {
                groups.push((params, vec![row as i64]));
            }
        }

        let mut output = Tensor::zeros([num_rows as i64], (Kind::Int64, logits.device()));
        for (params, rows) in groups {
            let row_indices = Tensor::from_slice(&rows).to_device(logits.device());
            let sampled = self.sample(&logits.index_select(0, &row_indices), params)?;
            output = output.index_copy(0, &row_indices, &sampled);
        }
        Vec::<i64>::try_from(&output).map_err(Into::into)
    }
}

fn validate_logits(logits: &Tensor) -> Result<()> {
    let shape = logits.size();
    if shape.len() != 2 || shape[1] <= 0 {
        return Err(SamplingError::InvalidLogitsShape(shape));
    }
    Ok(())
}

fn apply_top_p(logits: &Tensor, top_p: f64) -> Tensor {
    let (sorted_logits, sorted_indices) = logits.sort(-1, true);
    let cumulative_probs = sorted_logits
        .softmax(-1, Kind::Float)
        .cumsum(-1, Kind::Float);
    let remove = cumulative_probs.gt(top_p);

    // Keep the first token whose cumulative probability crosses the threshold.
    let mut first_column_shape = remove.size();
    let last_dim = first_column_shape.len() - 1;
    first_column_shape[last_dim] = 1;
    let first_column = Tensor::zeros(first_column_shape, (Kind::Bool, logits.device()));
    let shifted = Tensor::cat(
        &[first_column, remove.slice(-1, 0, logits.size()[1] - 1, 1)],
        -1,
    );
    let removal_mask = Tensor::zeros_like(&shifted).scatter(-1, &sorted_indices, &shifted);
    logits.masked_fill(&removal_mask, f64::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Tensor {
        Tensor::from_slice(&[0.1f32, 2.0, 0.2, 3.0, 1.0, 0.5]).view([2, 3])
    }

    #[test]
    fn samples_greedily_for_non_positive_temperature() {
        let sampled = Sampler
            .sample_batch(&logits(), &[SamplingParams::default(); 2])
            .unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn top_k_one_makes_sampling_deterministic() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let sampled = Sampler.sample_batch(&logits(), &[params; 2]).unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn top_p_zero_keeps_the_highest_logit() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.0,
            ..Default::default()
        };
        let sampled = Sampler.sample_batch(&logits(), &[params; 2]).unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn rejects_mismatched_batch_metadata() {
        assert!(matches!(
            Sampler.sample_batch(&logits(), &[SamplingParams::default()]),
            Err(SamplingError::ParamsLengthMismatch { .. })
        ));
    }
}
