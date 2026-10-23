//! Held-out one-step and autonomous rollout evaluation with explicit baselines.

use crate::{
    data::{Dataset, Split, Transition},
    model::Model,
    physics::{Physics, State},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::path::Path;

/// Errors are kept separate in physical units; standardized RMSE is dimensionless.
#[derive(Debug, Serialize)]
pub struct OneStep {
    pub method: String,
    pub theta_rmse: f64,
    pub omega_rmse: f64,
    pub standardized_rmse: f64,
}
/// A 10-second autonomous trajectory summary. No teacher forcing is applied.
#[derive(Debug, Serialize)]
pub struct Rollout {
    pub trajectory: usize,
    pub scope: String,
    pub method: String,
    pub initial: State,
    pub theta_rmse: f64,
    pub omega_rmse: f64,
    pub final_theta_error: f64,
    pub final_omega_error: f64,
    pub max_abs_theta_error: f64,
}
/// Per-step trajectories used directly by the plots and the report.
#[derive(Debug, Serialize)]
pub struct Trace {
    pub trajectory: usize,
    pub scope: String,
    pub method: String,
    pub t: f64,
    pub theta: f64,
    pub omega: f64,
    pub reference_theta: f64,
    pub reference_omega: f64,
    pub theta_error: f64,
    pub omega_error: f64,
    pub energy: f64,
}
/// Machine-readable acceptance result, plus all experiment scores.
#[derive(Debug, Serialize)]
pub struct Evaluation {
    pub schema_version: u32,
    pub test_samples: usize,
    pub one_step: Vec<OneStep>,
    pub standardized_improvement_fraction: f64,
    pub acceptance_passed: bool,
    pub rollouts: Vec<Rollout>,
}

fn predict(
    method: &str,
    x: State,
    physics: Physics,
    model: &Model,
    tiny: Option<&Model>,
) -> Result<State> {
    match method {
        "NN" => model.next(x),
        "Tiny NN" => tiny
            .expect("method included only when model exists")
            .next(x),
        "Euler" => Ok(physics.euler_step(x)),
        "RK4" => Ok(physics.rk4(x, physics.dt)),
        "Persistence" => Ok(x),
        _ => anyhow::bail!("Unknown evaluation method"),
    }
}

/// Test the frozen validation-selected model; write raw evidence and summaries.
pub fn evaluate(
    dataset: &Dataset,
    model: &Model,
    tiny: Option<&Model>,
    out: &Path,
) -> Result<Evaluation> {
    ensure!(
        dataset.config.physics == model.physics,
        "Dataset/model physics mismatch"
    );
    if let Some(tiny) = tiny {
        ensure!(tiny.physics == model.physics, "Tiny model physics mismatch");
    }
    std::fs::create_dir_all(out)?;
    let test: Vec<&Transition> = dataset
        .rows
        .iter()
        .filter(|r| r.split == Split::Test)
        .collect();
    ensure!(!test.is_empty(), "No held-out test data");
    let mut methods = vec!["Persistence", "Euler", "RK4", "NN"];
    if tiny.is_some() {
        methods.push("Tiny NN");
    }
    let mut scores = Vec::new();
    for method in &methods {
        let mut squared = [0.0; 2];
        let mut standardized = 0.0;
        for row in &test {
            let p = predict(method, row.state(), model.physics, model, tiny)?;
            for i in 0..2 {
                let e = p[i] - row.next()[i];
                squared[i] += e * e;
                standardized += (e / model.delta_normalization.std[i]).powi(2);
            }
        }
        scores.push(OneStep {
            method: (*method).into(),
            theta_rmse: (squared[0] / test.len() as f64).sqrt(),
            omega_rmse: (squared[1] / test.len() as f64).sqrt(),
            standardized_rmse: (standardized / (2 * test.len()) as f64).sqrt(),
        });
    }
    let persistence = scores[0].standardized_rmse;
    let nn = scores
        .iter()
        .find(|s| s.method == "NN")
        .expect("NN score exists")
        .standardized_rmse;
    let improvement = 1.0 - nn / persistence;
    let mut trace_csv = csv::Writer::from_path(out.join("rollouts.csv"))?;
    let mut rollout_scores = Vec::new();
    let mut initials: Vec<(usize, &str, State)> = test
        .iter()
        .filter(|r| r.t == 0.0)
        .map(|r| (r.trajectory, "held_out", r.state()))
        .collect();
    initials.push((
        dataset.config.trajectories,
        "out_of_distribution",
        [2.5, 0.0],
    ));
    for (id, scope, initial) in initials {
        for method in &methods {
            let mut reference = initial;
            let mut predicted = initial;
            let mut sum = [0.0; 2];
            let mut last = [0.0; 2];
            let mut max_theta = 0.0_f64;
            for step in 0..=dataset.config.steps {
                last = [predicted[0] - reference[0], predicted[1] - reference[1]];
                ensure!(
                    predicted.iter().all(|v| v.is_finite() && v.abs() < 1e6),
                    "{method} rollout diverged on trajectory {id}"
                );
                trace_csv.serialize(Trace {
                    trajectory: id,
                    scope: scope.into(),
                    method: (*method).into(),
                    t: step as f64 * model.physics.dt,
                    theta: predicted[0],
                    omega: predicted[1],
                    reference_theta: reference[0],
                    reference_omega: reference[1],
                    theta_error: last[0],
                    omega_error: last[1],
                    energy: model.physics.energy(predicted),
                })?;
                if step > 0 {
                    for i in 0..2 {
                        sum[i] += last[i].powi(2);
                    }
                }
                max_theta = max_theta.max(last[0].abs());
                if step < dataset.config.steps {
                    predicted = predict(method, predicted, model.physics, model, tiny)?;
                    reference = model.physics.reference_step(reference);
                }
            }
            rollout_scores.push(Rollout {
                trajectory: id,
                scope: scope.into(),
                method: (*method).into(),
                initial,
                theta_rmse: (sum[0] / dataset.config.steps as f64).sqrt(),
                omega_rmse: (sum[1] / dataset.config.steps as f64).sqrt(),
                final_theta_error: last[0],
                final_omega_error: last[1],
                max_abs_theta_error: max_theta,
            });
        }
    }
    trace_csv.flush()?;
    let result = Evaluation {
        schema_version: 1,
        test_samples: test.len(),
        one_step: scores,
        standardized_improvement_fraction: improvement,
        acceptance_passed: improvement >= 0.5,
        rollouts: rollout_scores,
    };
    std::fs::write(
        out.join("evaluation.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    std::fs::write(
        out.join("dataset.json"),
        serde_json::to_vec_pretty(&dataset.config)?,
    )?;
    println!(
        "Held-out improvement over persistence: {:.3}% (pass={})",
        100.0 * improvement,
        result.acceptance_passed
    );
    for s in &result.one_step {
        println!(
            "{}: theta RMSE={:.6e} rad, omega RMSE={:.6e} rad/s",
            s.method, s.theta_rmse, s.omega_rmse
        );
    }
    Ok(result)
}
