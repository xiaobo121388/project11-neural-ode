//! Versioned model checkpoints and physics-independent neural inference.

use crate::{
    data::Normalizer,
    nn::{Activation, Network},
    physics::{Physics, State},
};
use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Provenance for the selected training checkpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingInfo {
    pub seed: u64,
    pub epochs: usize,
    pub best_epoch: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub validation_mse: f64,
    pub elapsed_seconds: f64,
}

/// Portable f64 residual model. Inference never calls the pendulum derivative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub schema_version: u32,
    pub kind: String,
    pub physics: Physics,
    pub input_normalization: Normalizer,
    pub delta_normalization: Normalizer,
    pub network: Network,
    pub training: TrainingInfo,
}
impl Model {
    /// Structural and numerical validation performed when loading any checkpoint.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.kind == "residual-state-transition",
            "Unsupported model format"
        );
        self.physics.validate()?;
        self.input_normalization.validate()?;
        self.delta_normalization.validate()?;
        self.network.validate()?;
        ensure!(
            self.network.layers[0].weights.nrows() == 2,
            "Model must accept two state coordinates"
        );
        let output = self
            .network
            .layers
            .last()
            .expect("validated nonempty network");
        ensure!(
            output.weights.ncols() == 2 && output.activation == Activation::Linear,
            "Invalid output layer"
        );
        Ok(())
    }
    /// Deserialize and validate; errors are never replaced with an RK4 prediction.
    pub fn from_json(json: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(json).context("Invalid model JSON")?;
        model.validate().context("Incompatible model checkpoint")?;
        Ok(model)
    }
    /// Load a bounded checkpoint from disk.
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            std::fs::metadata(path)
                .with_context(|| format!("Missing model: {}", path.display()))?
                .len()
                <= 16_000_000,
            "Checkpoint exceeds 16 MB limit"
        );
        Self::from_json(&std::fs::read_to_string(path)?)
    }
    /// Validate and save through a temporary sibling file before replacement.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(temporary, path)?;
        Ok(())
    }
    /// Predict the next physical state using learned increments only.
    pub fn next(&self, state: State) -> Result<State> {
        ensure!(
            state.iter().all(|v| v.is_finite()),
            "Non-finite simulation state"
        );
        let x = self.input_normalization.encode(state);
        let prediction = self
            .network
            .forward(&Array2::from_shape_vec((1, 2), x.to_vec())?)?;
        let delta = self
            .delta_normalization
            .decode([prediction[[0, 0]], prediction[[0, 1]]]);
        let next = [state[0] + delta[0], state[1] + delta[1]];
        ensure!(
            next.iter().all(|v| v.is_finite()),
            "Neural rollout produced a non-finite state"
        );
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> Model {
        Model {
            schema_version: 1,
            kind: "residual-state-transition".into(),
            physics: Physics::default(),
            input_normalization: Normalizer {
                mean: [0.0; 2],
                std: [1.0; 2],
            },
            delta_normalization: Normalizer {
                mean: [0.0; 2],
                std: [0.02; 2],
            },
            network: Network::new(&[2, 3, 2], Activation::Tanh, 42).unwrap(),
            training: TrainingInfo::default(),
        }
    }
    #[test]
    fn checkpoint_round_trip_and_corruption() {
        let m = model();
        let text = serde_json::to_string(&m).unwrap();
        let loaded = Model::from_json(&text).unwrap();
        for (a, b) in loaded
            .next([0.2, 0.1])
            .unwrap()
            .iter()
            .zip(m.next([0.2, 0.1]).unwrap())
        {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(
            Model::from_json(&text.replace("\"schema_version\":1", "\"schema_version\":99"))
                .is_err()
        );
        let mut bad = m;
        bad.delta_normalization.std[0] = 0.0;
        assert!(bad.validate().is_err());
        assert!(loaded.next([f64::INFINITY, 0.0]).is_err());
    }
}
