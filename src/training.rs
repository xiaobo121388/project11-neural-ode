//! Reproducible mini-batch training and validation-only checkpoint selection.

use crate::{
    data::{Dataset, Normalizer, Split},
    model::{Model, TrainingInfo},
    nn::{Activation, Network},
};
use anyhow::{Result, ensure};
use ndarray::Axis;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};

/// Training controls shared by the two model sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 200,
            batch_size: 128,
            learning_rate: 0.01,
            seed: 42,
        }
    }
}
/// End-of-epoch losses, both computed after the last update of the epoch.
#[derive(Debug, Serialize)]
pub struct History {
    pub epoch: usize,
    pub train_mse: f64,
    pub validation_mse: f64,
    pub elapsed_seconds: f64,
}

/// Train one architecture. Test samples are never loaded into this function.
pub fn train(
    dataset: &Dataset,
    sizes: &[usize],
    hidden: Activation,
    config: &TrainConfig,
    model_path: &Path,
    history_path: &Path,
) -> Result<Model> {
    ensure!(
        config.epochs > 0
            && config.batch_size > 0
            && config.learning_rate.is_finite()
            && config.learning_rate > 0.0,
        "Invalid training configuration"
    );
    let (raw_x, raw_y) = dataset.matrices(Split::Train)?;
    let (val_x, val_y) = dataset.matrices(Split::Validation)?;
    let input_normalization = Normalizer::fit(&raw_x)?;
    let delta_normalization = Normalizer::fit(&raw_y)?;
    let x = input_normalization.matrix(&raw_x)?;
    let y = delta_normalization.matrix(&raw_y)?;
    let vx = input_normalization.matrix(&val_x)?;
    let vy = delta_normalization.matrix(&val_y)?;
    let mut network = Network::new(sizes, hidden, config.seed)?;
    let mut best_network = network.clone();
    let mut best_loss = network.loss(&vx, &vy)?;
    let mut best_epoch = 0;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed + 1);
    let mut order: Vec<usize> = (0..x.nrows()).collect();
    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut history = csv::Writer::from_path(history_path)?;
    let start = Instant::now();
    history.serialize(History {
        epoch: 0,
        train_mse: network.loss(&x, &y)?,
        validation_mse: best_loss,
        elapsed_seconds: 0.0,
    })?;
    for epoch in 1..=config.epochs {
        order.shuffle(&mut rng);
        for batch in order.chunks(config.batch_size) {
            network.train_batch(
                &x.select(Axis(0), batch),
                &y.select(Axis(0), batch),
                config.learning_rate,
            )?;
        }
        let train_mse = network.loss(&x, &y)?;
        let validation_mse = network.loss(&vx, &vy)?;
        if validation_mse < best_loss {
            best_loss = validation_mse;
            best_epoch = epoch;
            best_network = network.clone();
        }
        history.serialize(History {
            epoch,
            train_mse,
            validation_mse,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        })?;
        if epoch == 1 || epoch % 20 == 0 || epoch == config.epochs {
            println!(
                "epoch {epoch:3}/{} train={train_mse:.6e} validation={validation_mse:.6e}",
                config.epochs
            );
        }
    }
    history.flush()?;
    let model = Model {
        schema_version: 1,
        kind: "residual-state-transition".into(),
        physics: dataset.config.physics,
        input_normalization,
        delta_normalization,
        network: best_network,
        training: TrainingInfo {
            seed: config.seed,
            epochs: config.epochs,
            best_epoch,
            batch_size: config.batch_size,
            learning_rate: config.learning_rate,
            validation_mse: best_loss,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        },
    };
    model.save(model_path)?;
    println!(
        "selected epoch {best_epoch}, {} parameters, {:.3}s",
        model.network.parameter_count(),
        model.training.elapsed_seconds
    );
    Ok(model)
}
