//! Hand-written neural state-transition models and a reproducible ODE experiment.
//!
//! Reference trajectories use a fourth-order Runge-Kutta integrator. The learned
//! model predicts state increments without invoking the physical equation.

pub mod data;
pub mod evaluation;
pub mod model;
pub mod nn;
pub mod physics;
pub mod training;
