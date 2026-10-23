//! Console-free offline Windows desktop entry point.
#![cfg_attr(windows, windows_subsystem = "windows")]

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Team 11 offline pendulum lab")]
struct Options {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 1280.0)]
    width: f32,
    #[arg(long, default_value_t = 800.0)]
    height: f32,
    #[arg(long)]
    paused: bool,
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long, default_value_t = 5.8)]
    capture_time: f64,
    #[arg(long, default_value_t = 0)]
    preset: usize,
    #[arg(long)]
    tiny: bool,
}
fn main() {
    let options = Options::parse();
    let capture = options.snapshot.map(|path| project11::gui::Capture {
        path,
        time: options.capture_time,
        preset: options.preset,
        tiny: options.tiny,
    });
    if let Err(error) = project11::gui::launch_with_capture(
        options.model.as_deref(),
        [options.width, options.height],
        options.paused,
        capture,
    ) {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}
