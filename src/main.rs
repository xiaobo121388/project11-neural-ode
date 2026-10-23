//! Command-line workflow for the Team 11 reproducible experiment.

use anyhow::Result;
use clap::{Parser, Subcommand};
use project11::{
    data::{DataConfig, Dataset},
    evaluation,
    model::Model,
    nn::Activation,
    training::{self, TrainConfig},
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Hand-written neural ODE surrogate: Team 11")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Open the offline desktop lab (also the default with no subcommand).
    Gui {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 1280.0)]
        width: f32,
        #[arg(long, default_value_t = 800.0)]
        height: f32,
        #[arg(long)]
        paused: bool,
    },
    /// Generate the default 80-trajectory reference dataset.
    Generate {
        #[arg(long, default_value = ".")]
        out: PathBuf,
    },
    /// Train both networks; use validation loss only for checkpoint selection.
    Train {
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long, default_value_t = 200)]
        epochs: usize,
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
    },
    /// Evaluate saved models on held-out data and autonomous rollouts.
    Evaluate {
        #[arg(long, default_value = ".")]
        out: PathBuf,
    },
    /// Export the shared fixture, check Rust parity outputs and time five runs.
    Benchmark {
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
    },
    /// Generate, train and evaluate the entire deterministic experiment.
    Reproduce {
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long, default_value_t = 200)]
        epochs: usize,
    },
}
fn train(out: &std::path::Path, epochs: usize, learning_rate: f64) -> Result<()> {
    let data = Dataset::load(&out.join("data"))?;
    let config = TrainConfig {
        epochs,
        learning_rate,
        ..TrainConfig::default()
    };
    println!("Training main model (2-32-32-2 Tanh)");
    training::train(
        &data,
        &[2, 32, 32, 2],
        Activation::Tanh,
        &config,
        &out.join("models/main.json"),
        &out.join("results/history_main.csv"),
    )?;
    println!("Training small baseline (2-3-2 Sigmoid)");
    training::train(
        &data,
        &[2, 3, 2],
        Activation::Sigmoid,
        &config,
        &out.join("models/tiny.json"),
        &out.join("results/history_tiny.csv"),
    )?;
    Ok(())
}
fn evaluate(out: &std::path::Path) -> Result<()> {
    let data = Dataset::load(&out.join("data"))?;
    let model = Model::load(&out.join("models/main.json"))?;
    let tiny = Model::load(&out.join("models/tiny.json"))?;
    evaluation::evaluate(&data, &model, Some(&tiny), &out.join("results"))?;
    Ok(())
}
fn run() -> Result<()> {
    let command = Cli::parse().command.unwrap_or(Command::Gui {
        model: None,
        width: 1280.0,
        height: 800.0,
        paused: false,
    });
    match command {
        Command::Gui {
            model,
            width,
            height,
            paused,
        } => {
            #[cfg(feature = "gui")]
            project11::gui::launch(model.as_deref(), [width, height], paused)?;
            #[cfg(not(feature = "gui"))]
            {
                let _ = (model, width, height, paused);
                anyhow::bail!("GUI not compiled; rebuild with default features");
            }
        }
        Command::Generate { out } => {
            let data = Dataset::generate(DataConfig::default())?;
            data.save(&out.join("data"))?;
            println!("Generated {} transitions", data.rows.len());
        }
        Command::Train {
            out,
            epochs,
            learning_rate,
        } => train(&out, epochs, learning_rate)?,
        Command::Evaluate { out } => evaluate(&out)?,
        Command::Benchmark { out, epochs } => project11::benchmark::benchmark(&out, epochs)?,
        Command::Reproduce { out, epochs } => {
            Dataset::generate(DataConfig::default())?.save(&out.join("data"))?;
            train(&out, epochs, 0.01)?;
            evaluate(&out)?;
        }
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}
