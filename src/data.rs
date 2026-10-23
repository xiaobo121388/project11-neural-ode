//! Seeded trajectory generation and train-only standardization.

use crate::physics::{Physics, State};
use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Split membership belongs to an entire trajectory, never to individual rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

/// One supervised transition in physical units.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transition {
    pub trajectory: usize,
    pub split: Split,
    pub t: f64,
    pub theta: f64,
    pub omega: f64,
    pub next_theta: f64,
    pub next_omega: f64,
}
impl Transition {
    /// Current physical state.
    pub fn state(&self) -> State {
        [self.theta, self.omega]
    }
    /// Reference next state.
    pub fn next(&self) -> State {
        [self.next_theta, self.next_omega]
    }
    /// Target increment, not the absolute next state.
    pub fn delta(&self) -> State {
        [self.next_theta - self.theta, self.next_omega - self.omega]
    }
}

/// Fully specified experiment generation recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataConfig {
    pub seed: u64,
    pub physics: Physics,
    pub trajectories: usize,
    pub steps: usize,
    pub train_trajectories: usize,
    pub validation_trajectories: usize,
    pub theta_range: State,
    pub omega_range: State,
}
impl Default for DataConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            physics: Physics::default(),
            trajectories: 80,
            steps: 500,
            train_trajectories: 56,
            validation_trajectories: 12,
            theta_range: [-1.5, 1.5],
            omega_range: [-2.0, 2.0],
        }
    }
}
impl DataConfig {
    /// Validate before allocating or integrating trajectories.
    pub fn validate(&self) -> Result<()> {
        self.physics.validate()?;
        ensure!(
            self.trajectories
                > self
                    .train_trajectories
                    .saturating_add(self.validation_trajectories)
                && self.train_trajectories > 0
                && self.validation_trajectories > 0,
            "Need three nonempty trajectory splits"
        );
        ensure!(
            self.steps > 0 && self.steps <= 100000 && self.trajectories <= 10000,
            "Dataset size out of bounds"
        );
        ensure!(
            self.trajectories.saturating_mul(self.steps) <= 2_000_000,
            "Dataset exceeds two million transitions"
        );
        for range in [self.theta_range, self.omega_range] {
            ensure!(
                range.iter().all(|x| x.is_finite()) && range[0] < range[1],
                "Invalid initial-state range"
            );
        }
        Ok(())
    }
    /// Deterministic disjoint split assignment by trajectory identifier.
    pub fn split(&self, id: usize) -> Split {
        if id < self.train_trajectories {
            Split::Train
        } else if id < self.train_trajectories + self.validation_trajectories {
            Split::Validation
        } else {
            Split::Test
        }
    }
}

/// Dataset plus the generation metadata needed for reproduction.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub config: DataConfig,
    pub rows: Vec<Transition>,
}
impl Dataset {
    /// Generate independent initial states with ChaCha8 and RK4 reference labels.
    pub fn generate(config: DataConfig) -> Result<Self> {
        config.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut rows = Vec::with_capacity(config.trajectories * config.steps);
        for trajectory in 0..config.trajectories {
            let mut x = [
                rng.random_range(config.theta_range[0]..config.theta_range[1]),
                rng.random_range(config.omega_range[0]..config.omega_range[1]),
            ];
            for step in 0..config.steps {
                let next = config.physics.reference_step(x);
                rows.push(Transition {
                    trajectory,
                    split: config.split(trajectory),
                    t: step as f64 * config.physics.dt,
                    theta: x[0],
                    omega: x[1],
                    next_theta: next[0],
                    next_omega: next[1],
                });
                x = next;
            }
        }
        Ok(Self { config, rows })
    }

    /// Persist data and recipe as transitions.csv and config.json.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut csv = csv::Writer::from_path(dir.join("transitions.csv"))?;
        for row in &self.rows {
            csv.serialize(row)?;
        }
        csv.flush()?;
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&self.config)?,
        )?;
        Ok(())
    }

    /// Load a dataset and reject inconsistent splits, time grids or invalid data.
    pub fn load(dir: &Path) -> Result<Self> {
        let config: DataConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        config.validate()?;
        let rows: Vec<Transition> = csv::Reader::from_path(dir.join("transitions.csv"))?
            .deserialize()
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            rows.len() == config.trajectories * config.steps,
            "Dataset row count differs from recipe"
        );
        for (i, row) in rows.iter().enumerate() {
            ensure!(
                row.trajectory == i / config.steps && row.split == config.split(row.trajectory),
                "Trajectory ordering or split membership invalid"
            );
            ensure!(
                (row.t - (i % config.steps) as f64 * config.physics.dt).abs() < 1e-9,
                "Dataset time grid invalid"
            );
            ensure!(
                row.state()
                    .iter()
                    .chain(row.next().iter())
                    .all(|v| v.is_finite()),
                "Non-finite data"
            );
        }
        Ok(Self { config, rows })
    }

    /// Raw state and increment matrices for one split, in deterministic row order.
    pub fn matrices(&self, split: Split) -> Result<(Array2<f64>, Array2<f64>)> {
        let rows: Vec<_> = self.rows.iter().filter(|r| r.split == split).collect();
        ensure!(!rows.is_empty(), "Requested split is empty");
        let x = Array2::from_shape_vec(
            (rows.len(), 2),
            rows.iter().flat_map(|r| r.state()).collect(),
        )?;
        let y = Array2::from_shape_vec(
            (rows.len(), 2),
            rows.iter().flat_map(|r| r.delta()).collect(),
        )?;
        Ok((x, y))
    }
}

/// Population mean and standard deviation fitted exclusively to training rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Normalizer {
    pub mean: State,
    pub std: State,
}
impl Normalizer {
    /// Compute per-column population statistics with a 1e-8 zero-variance floor.
    pub fn fit(values: &Array2<f64>) -> Result<Self> {
        ensure!(
            values.nrows() > 0 && values.ncols() == 2 && values.iter().all(|v| v.is_finite()),
            "Invalid normalization data"
        );
        let mean = std::array::from_fn(|i| values.column(i).sum() / values.nrows() as f64);
        let std = std::array::from_fn(|i| {
            (values.column(i).mapv(|v| (v - mean[i]).powi(2)).sum() / values.nrows() as f64)
                .sqrt()
                .max(1e-8)
        });
        Ok(Self { mean, std })
    }
    /// Checks serialized normalization statistics.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.mean.iter().all(|v| v.is_finite())
                && self.std.iter().all(|v| v.is_finite() && *v > 0.0),
            "Invalid normalization statistics"
        );
        Ok(())
    }
    /// Normalize a physical two-component value.
    pub fn encode(&self, x: State) -> State {
        std::array::from_fn(|i| (x[i] - self.mean[i]) / self.std[i])
    }
    /// Inverse of encode.
    pub fn decode(&self, x: State) -> State {
        std::array::from_fn(|i| x[i] * self.std[i] + self.mean[i])
    }
    /// Encode a two-column sample matrix.
    pub fn matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        ensure!(x.ncols() == 2, "Normalization requires two columns");
        self.validate().context("Cannot normalize matrix")?;
        Ok(Array2::from_shape_fn(x.dim(), |(r, c)| {
            (x[[r, c]] - self.mean[c]) / self.std[c]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn trajectory_splits_and_generation_are_reproducible() {
        let c = DataConfig {
            trajectories: 8,
            train_trajectories: 4,
            validation_trajectories: 2,
            steps: 5,
            ..DataConfig::default()
        };
        let a = Dataset::generate(c.clone()).unwrap();
        let b = Dataset::generate(c).unwrap();
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.matrices(Split::Train).unwrap().0.nrows(), 20);
        assert_eq!(a.matrices(Split::Test).unwrap().0.nrows(), 10);
        for group in a.rows.chunks(5) {
            assert!(group.iter().all(|r| r.split == group[0].split));
        }
    }
    #[test]
    fn normalization_round_trip_and_zero_variance() {
        let n = Normalizer::fit(&array![[1.0, 2.0], [3.0, 2.0]]).unwrap();
        assert_eq!(n.mean, [2.0, 2.0]);
        assert_eq!(n.std, [1.0, 1e-8]);
        assert_eq!(n.decode(n.encode([2.2, 2.0])), [2.2, 2.0]);
    }
}
