//! Shared f64 fixtures, numerical parity evidence and bounded training benchmarks.

use crate::{
    data::{Dataset, Normalizer, Split},
    nn::{Activation, Network},
};
use anyhow::{Result, ensure};
use ndarray::{Array2, Axis};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};

/// Serialized once and consumed unchanged by both implementations.
#[derive(Debug, Serialize, Deserialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub x: Array2<f64>,
    pub y: Array2<f64>,
    pub initial: Network,
    pub orders: Vec<Vec<usize>>,
}
#[derive(Serialize)]
struct Parity {
    prediction: Array2<f64>,
    loss: f64,
    gradient_weights: Vec<Array2<f64>>,
    gradient_bias: Vec<ndarray::Array1<f64>>,
    updated: Network,
}
#[derive(Debug, Serialize)]
struct Trial {
    trial: usize,
    seconds: f64,
    final_loss: f64,
}
#[derive(Serialize)]
struct Timing {
    implementation: String,
    epochs: usize,
    samples: usize,
    batch_size: usize,
    learning_rate: f64,
    dtype: String,
    warmups: usize,
    repetitions: usize,
    threads: usize,
    backend: String,
    median_seconds: f64,
    trials: Vec<Trial>,
}

fn run(f: &Fixture) -> Result<(f64, Network)> {
    let mut network = f.initial.clone();
    let start = Instant::now();
    for order in &f.orders {
        for batch in order.chunks(f.batch_size) {
            network.train_batch(
                &f.x.select(Axis(0), batch),
                &f.y.select(Axis(0), batch),
                f.learning_rate,
            )?;
        }
    }
    Ok((start.elapsed().as_secs_f64(), network))
}

/// Prepare shared inputs and benchmark Rust only. Python is run as a separate process.
pub fn benchmark(root: &Path, epochs: usize) -> Result<()> {
    ensure!(
        (1..=400).contains(&epochs),
        "Benchmark epochs must be 1 to 400"
    );
    let dataset = Dataset::load(&root.join("data"))?;
    let (x, y) = dataset.matrices(Split::Train)?;
    let x = Normalizer::fit(&x)?.matrix(&x)?;
    let y = Normalizer::fit(&y)?.matrix(&y)?;
    let mut rng = ChaCha8Rng::seed_from_u64(43);
    let mut order: Vec<usize> = (0..x.nrows()).collect();
    let orders = (0..epochs)
        .map(|_| {
            order.shuffle(&mut rng);
            order.clone()
        })
        .collect();
    let f = Fixture {
        schema_version: 1,
        epochs,
        batch_size: 128,
        learning_rate: 0.01,
        x,
        y,
        initial: Network::new(&[2, 32, 32, 2], Activation::Tanh, 42)?,
        orders,
    };
    let fixture_dir = root.join("benchmark_fixture");
    std::fs::create_dir_all(&fixture_dir)?;
    let results = root.join("results");
    std::fs::create_dir_all(&results)?;
    std::fs::write(fixture_dir.join("shared.json"), serde_json::to_vec(&f)?)?;
    let px = f.x.select(Axis(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
    let py = f.y.select(Axis(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
    let (loss, gradients) = f.initial.gradients(&px, &py)?;
    let mut updated = f.initial.clone();
    updated.train_batch(&px, &py, f.learning_rate)?;
    let parity = Parity {
        prediction: f.initial.forward(&px)?,
        loss,
        gradient_weights: gradients.iter().map(|g| g.weights.clone()).collect(),
        gradient_bias: gradients.iter().map(|g| g.bias.clone()).collect(),
        updated,
    };
    std::fs::write(
        fixture_dir.join("parity_rust.json"),
        serde_json::to_vec(&parity)?,
    )?;
    println!(
        "Benchmark: {} epochs, {} samples, one warmup plus five timed runs",
        epochs,
        f.x.nrows()
    );
    run(&f)?;
    let mut trials = Vec::new();
    let mut csv = csv::Writer::from_path(results.join("benchmark_rust.csv"))?;
    for trial in 1..=5 {
        let (seconds, network) = run(&f)?;
        let row = Trial {
            trial,
            seconds,
            final_loss: network.loss(&f.x, &f.y)?,
        };
        csv.serialize(&row)?;
        println!("Rust trial {trial}: {seconds:.4}s");
        trials.push(row);
    }
    csv.flush()?;
    let mut times: Vec<_> = trials.iter().map(|r| r.seconds).collect();
    times.sort_by(f64::total_cmp);
    let summary = Timing {
        implementation: "Rust ndarray".into(),
        epochs,
        samples: f.x.nrows(),
        batch_size: f.batch_size,
        learning_rate: f.learning_rate,
        dtype: "float64".into(),
        warmups: 1,
        repetitions: 5,
        threads: 1,
        backend: "ndarray matrixmultiply (no external BLAS, no rayon)".into(),
        median_seconds: times[2],
        trials,
    };
    std::fs::write(
        results.join("benchmark_rust.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}
