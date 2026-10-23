//! Pendulum equations and explicit numerical reference integrators.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// State in radians and radians per second. Angles are not wrapped or clipped.
pub type State = [f64; 2];

/// Physical constants and the public prediction / internal reference time steps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Physics {
    pub gravity: f64,
    pub length: f64,
    pub damping: f64,
    pub dt: f64,
    pub reference_dt: f64,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            length: 1.0,
            damping: 0.1,
            dt: 0.02,
            reference_dt: 0.002,
        }
    }
}

impl Physics {
    /// Rejects nonphysical constants and inconsistent reference time steps.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.gravity,
                self.length,
                self.damping,
                self.dt,
                self.reference_dt
            ]
            .iter()
            .all(|v| v.is_finite()),
            "Physical parameters must be finite"
        );
        ensure!(
            self.gravity > 0.0 && self.length > 0.0,
            "g and length must be positive"
        );
        ensure!(self.damping >= 0.0, "Damping must be nonnegative");
        ensure!(
            self.reference_dt > 0.0 && self.dt >= self.reference_dt,
            "Invalid time steps"
        );
        let steps = self.dt / self.reference_dt;
        ensure!(
            (steps - steps.round()).abs() < 1e-9 && steps <= 10000.0,
            "Prediction dt must be an integer multiple of reference dt (at most 10000 substeps)"
        );
        Ok(())
    }

    /// The first-order damped nonlinear pendulum ODE.
    pub fn derivative(&self, state: State) -> State {
        [
            state[1],
            -self.gravity / self.length * state[0].sin() - self.damping * state[1],
        ]
    }

    /// A single classical fourth-order Runge-Kutta step of size h.
    pub fn rk4(&self, state: State, h: f64) -> State {
        let k1 = self.derivative(state);
        let k2 = self.derivative(add_scaled(state, k1, h * 0.5));
        let k3 = self.derivative(add_scaled(state, k2, h * 0.5));
        let k4 = self.derivative(add_scaled(state, k3, h));
        std::array::from_fn(|i| state[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    /// Advances exactly one prediction interval using validated RK4 substeps.
    pub fn reference_step(&self, mut state: State) -> State {
        for _ in 0..(self.dt / self.reference_dt).round() as usize {
            state = self.rk4(state, self.reference_dt);
        }
        state
    }

    /// The explicit Euler baseline at the public prediction time step.
    pub fn euler_step(&self, state: State) -> State {
        add_scaled(state, self.derivative(state), self.dt)
    }

    /// Mechanical energy per unit mass, with downward equilibrium as zero.
    pub fn energy(&self, state: State) -> f64 {
        0.5 * self.length.powi(2) * state[1].powi(2)
            + self.gravity * self.length * (1.0 - state[0].cos())
    }
}

fn add_scaled(state: State, derivative: State, h: f64) -> State {
    [state[0] + h * derivative[0], state[1] + h * derivative[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_remains_fixed() {
        assert_eq!(Physics::default().reference_step([0.0, 0.0]), [0.0, 0.0]);
    }

    #[test]
    fn rk4_converges_at_fourth_order() {
        let p = Physics::default();
        let integrate = |h: f64| {
            let mut x = [1.0, 0.4];
            for _ in 0..(1.0 / h).round() as usize {
                x = p.rk4(x, h);
            }
            x
        };
        let truth = integrate(0.00025);
        let error = |h| {
            let x = integrate(h);
            ((x[0] - truth[0]).powi(2) + (x[1] - truth[1]).powi(2)).sqrt()
        };
        assert!(error(0.04) / error(0.02) > 14.0);
        assert!(error(0.002) < 1e-9);
    }

    #[test]
    fn damped_energy_decreases() {
        let p = Physics::default();
        let mut x = [1.2, 1.0];
        for _ in 0..500 {
            let next = p.reference_step(x);
            assert!(p.energy(next) <= p.energy(x) + 1e-10);
            x = next;
        }
    }

    #[test]
    fn small_angle_agrees_with_linear_analytic_solution() {
        let p = Physics {
            damping: 0.0,
            ..Physics::default()
        };
        let amplitude = 1e-5;
        let mut x = [amplitude, 0.0];
        for _ in 0..50 {
            x = p.reference_step(x);
        }
        let expected = amplitude * (p.gravity / p.length).sqrt().cos();
        assert!((x[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(
            Physics {
                dt: f64::NAN,
                ..Physics::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Physics {
                reference_dt: 0.003,
                ..Physics::default()
            }
            .validate()
            .is_err()
        );
    }
}
