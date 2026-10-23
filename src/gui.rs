//! Offline desktop lab with embedded checkpoints and autonomous, fixed-step rollouts.

use crate::{model::Model, physics::State};
use anyhow::{Context, Result};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

/// Deterministic native GPU readback for visual QA, not an image mockup.
#[derive(Clone)]
pub struct Capture {
    pub path: PathBuf,
    pub time: f64,
    pub preset: usize,
    pub tiny: bool,
}
type CaptureStatus = Arc<Mutex<Option<std::result::Result<(), String>>>>;

const BLUE: Color32 = Color32::from_rgb(36, 91, 194);
const ORANGE: Color32 = Color32::from_rgb(203, 83, 44);
const INK: Color32 = Color32::from_rgb(30, 35, 43);
const MUTED: Color32 = Color32::from_rgb(101, 109, 119);

/// One independently generated comparison frame.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub t: f64,
    pub reference: State,
    pub neural: State,
}

/// Compute the learned rollout recursively; no true state enters the NN after t=0.
pub fn frames(model: &Model, initial: State, steps: usize) -> Result<Vec<Frame>> {
    model.validate()?;
    let mut reference = initial;
    let mut neural = initial;
    let mut frames = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        anyhow::ensure!(
            neural.iter().all(|x| x.is_finite() && x.abs() < 1e6),
            "Prediction diverged; reset the initial state"
        );
        frames.push(Frame {
            t: step as f64 * model.physics.dt,
            reference,
            neural,
        });
        if step < steps {
            reference = model.physics.reference_step(reference);
            neural = model.next(neural)?;
        }
    }
    Ok(frames)
}

struct Lab {
    models: Vec<(String, Model)>,
    selected: usize,
    initial: State,
    preset: usize,
    trace: Vec<Frame>,
    time: f64,
    running: bool,
    speed: f64,
    last_tick: Instant,
    error: Option<String>,
    capture: Option<Capture>,
    capture_status: CaptureStatus,
    rendered_frames: usize,
    screenshot_requested: bool,
}
impl Lab {
    fn new(
        cc: &eframe::CreationContext<'_>,
        path: Option<&Path>,
        paused: bool,
        capture: Option<Capture>,
        capture_status: CaptureStatus,
    ) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(10.0, 8.0);
        style.visuals.panel_fill = Color32::from_rgb(250, 251, 252);
        cc.egui_ctx.set_style(style);
        let loaded = match path {
            Some(path) => Model::load(path).map(|m| vec![("Custom checkpoint".into(), m)]),
            None => (|| -> Result<_> {
                Ok(vec![
                    (
                        "Main / 2-32-32-2 Tanh".into(),
                        Model::from_json(include_str!("../models/main.json"))?,
                    ),
                    (
                        "Tiny / 2-3-2 Sigmoid".into(),
                        Model::from_json(include_str!("../models/tiny.json"))?,
                    ),
                ])
            })(),
        };
        let (models, error) = match loaded {
            Ok(m) => (m, None),
            Err(e) => (Vec::new(), Some(format!("Cannot load model: {e:#}"))),
        };
        let mut app = Self {
            models,
            selected: 0,
            initial: [0.29242229484062987, 1.2044954808830122],
            preset: 0,
            trace: Vec::new(),
            time: 0.0,
            running: !paused,
            speed: 1.0,
            last_tick: Instant::now(),
            error,
            capture,
            capture_status,
            rendered_frames: 0,
            screenshot_requested: false,
        };
        if !app.models.is_empty() {
            if let Some(capture) = &app.capture {
                app.preset = capture.preset;
                app.initial = match capture.preset {
                    0 => app.initial,
                    1 => [1.400726226143501, -1.443007949655112],
                    _ => [2.5, 0.0],
                };
                if capture.tiny && app.models.len() > 1 {
                    app.selected = 1;
                }
            }
            app.reset_trace();
            if let Some(capture) = &app.capture {
                app.time = capture.time;
                app.running = false;
            }
        }
        app
    }
    fn reset_trace(&mut self) {
        self.time = 0.0;
        self.last_tick = Instant::now();
        let model = &self.models[self.selected].1;
        let steps = (10.0 / model.physics.dt).round() as usize;
        if steps == 0 || steps > 100000 {
            self.error = Some("Prediction dt is too small for the desktop preview".into());
            self.trace.clear();
            return;
        }
        match frames(model, self.initial, steps) {
            Ok(trace) => {
                self.trace = trace;
                self.error = None;
            }
            Err(e) => {
                self.trace.clear();
                self.error = Some(format!("Simulation stopped: {e:#}"));
                self.running = false;
            }
        }
    }
    fn toggle(&mut self) {
        if self.time >= 10.0 {
            self.time = 0.0;
        }
        self.running = !self.running;
        self.last_tick = Instant::now();
    }
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(12.0);
        ui.heading("Experiment");
        ui.add_space(8.0);
        let mut changed = false;
        ui.label("Checkpoint");
        egui::ComboBox::from_id_salt("model")
            .selected_text(&self.models[self.selected].0)
            .width(212.0)
            .show_ui(ui, |ui| {
                for (i, (name, _)) in self.models.iter().enumerate() {
                    changed |= ui.selectable_value(&mut self.selected, i, name).changed();
                }
            });
        ui.label(format!(
            "{} trainable parameters",
            self.models[self.selected].1.network.parameter_count()
        ));
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);
        ui.strong("Initial state");
        let names = [
            "Held-out 68",
            "Held-out 74 / high amplitude",
            "Outside training range",
            "Custom",
        ];
        egui::ComboBox::from_id_salt("preset")
            .selected_text(names[self.preset])
            .width(212.0)
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate().take(3) {
                    if ui.selectable_value(&mut self.preset, i, *name).changed() {
                        self.initial = match i {
                            0 => [0.29242229484062987, 1.2044954808830122],
                            1 => [1.400726226143501, -1.443007949655112],
                            _ => [2.5, 0.0],
                        };
                        changed = true;
                    }
                }
            });
        ui.label("Angle / rad");
        if ui
            .add(
                egui::Slider::new(&mut self.initial[0], -2.8..=2.8)
                    .custom_formatter(|v, _| format!("{v:.2}")),
            )
            .changed()
        {
            self.preset = 3;
            changed = true;
        }
        ui.label("Angular velocity / rad/s");
        if ui
            .add(
                egui::Slider::new(&mut self.initial[1], -4.0..=4.0)
                    .custom_formatter(|v, _| format!("{v:.2}")),
            )
            .changed()
        {
            self.preset = 3;
            changed = true;
        }
        if self.initial[0].abs() > 1.5 || self.initial[1].abs() > 2.0 {
            ui.colored_label(ORANGE, "Outside training initial-state range");
        }
        if changed {
            self.reset_trace();
        }
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);
        ui.strong("Playback");
        ui.horizontal(|ui| {
            let icon = if self.running {
                egui::include_image!("../assets/pause.svg")
            } else {
                egui::include_image!("../assets/play.svg")
            };
            if ui
                .add(egui::Button::image(
                    egui::Image::new(icon).fit_to_exact_size(Vec2::splat(19.0)),
                ))
                .on_hover_text(if self.running { "Pause" } else { "Play" })
                .clicked()
            {
                self.toggle();
            }
            if ui
                .add(egui::Button::image(
                    egui::Image::new(egui::include_image!("../assets/rotate-ccw.svg"))
                        .fit_to_exact_size(Vec2::splat(19.0)),
                ))
                .on_hover_text("Reset")
                .clicked()
            {
                self.time = 0.0;
                self.running = false;
            }
            if ui
                .add(egui::Button::image(
                    egui::Image::new(egui::include_image!("../assets/step-forward.svg"))
                        .fit_to_exact_size(Vec2::splat(19.0)),
                ))
                .on_hover_text("Advance one prediction step")
                .clicked()
            {
                self.running = false;
                self.time = (self.time + self.models[self.selected].1.physics.dt).min(10.0);
            }
        });
        ui.label("Speed");
        ui.add(egui::Slider::new(&mut self.speed, 0.25..=2.0).suffix("x"));
        ui.label("Time / s");
        if ui
            .add(
                egui::Slider::new(&mut self.time, 0.0..=10.0)
                    .custom_formatter(|v, _| format!("{v:.2}")),
            )
            .changed()
        {
            self.running = false;
        }
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);
        let p = self.models[self.selected].1.physics;
        ui.strong("Physical model");
        egui::Grid::new("physics")
            .num_columns(2)
            .spacing([16.0, 7.0])
            .show(ui, |ui| {
                for (key, value) in [
                    ("Gravity", format!("{:.2} m/s2", p.gravity)),
                    ("Length", format!("{:.2} m", p.length)),
                    ("Damping", format!("{:.2} /s", p.damping)),
                    ("Prediction dt", format!("{:.3} s", p.dt)),
                    ("RK4 substep", format!("{:.3} s", p.reference_dt)),
                ] {
                    ui.label(key);
                    ui.label(value);
                    ui.end_row();
                }
            });
    }
}
impl eframe::App for Lab {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(capture) = &self.capture {
            self.rendered_frames += 1;
            for event in ctx.input(|i| i.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = event {
                    let rgba: Vec<u8> = image
                        .pixels
                        .iter()
                        .flat_map(|pixel| pixel.to_array())
                        .collect();
                    let result = (|| -> Result<()> {
                        if let Some(parent) = capture.path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        image::save_buffer(
                            &capture.path,
                            &rgba,
                            image.size[0] as u32,
                            image.size[1] as u32,
                            image::ColorType::Rgba8,
                        )?;
                        Ok(())
                    })()
                    .map_err(|e| format!("Native screenshot export failed: {e:#}"));
                    *self.capture_status.lock().expect("capture status lock") = Some(result);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            if self.rendered_frames >= 5 && !self.screenshot_requested {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                self.screenshot_requested = true;
            }
            if self.rendered_frames > 500 {
                *self.capture_status.lock().expect("capture status lock") =
                    Some(Err("Native screenshot timed out".into()));
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ctx.request_repaint();
        }
        let now = Instant::now();
        let wall = (now - self.last_tick).as_secs_f64().min(0.1);
        self.last_tick = now;
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) && !self.models.is_empty() {
            self.toggle();
        }
        if self.running {
            self.time = (self.time + wall * self.speed).min(10.0);
            if self.time >= 10.0 {
                self.running = false;
            }
        }
        if self.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        egui::TopBottomPanel::top("header")
            .exact_height(73.0)
            .show(ctx, |ui| {
                ui.add_space(9.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("TEAM 11  /  NEURAL STATE-TRANSITION MODEL")
                                .size(11.0)
                                .color(MUTED),
                        );
                        ui.label(
                            egui::RichText::new("Pendulum Dynamics")
                                .size(24.0)
                                .strong()
                                .color(INK),
                        );
                    });
                });
            });
        if self.models.is_empty() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Model unavailable");
                ui.colored_label(ORANGE, self.error.as_deref().unwrap_or("No model loaded"));
            });
            return;
        }
        egui::SidePanel::left("controls")
            .exact_width(257.0)
            .resizable(false)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(error) = &self.error {
                ui.colored_label(ORANGE, error);
                return;
            }
            let index = ((self.time / self.models[self.selected].1.physics.dt).floor() as usize)
                .min(self.trace.len() - 1);
            let frame = self.trace[index];
            let rmse = (self
                .trace
                .iter()
                .skip(1)
                .map(|f| (f.neural[0] - f.reference[0]).powi(2))
                .sum::<f64>()
                / (self.trace.len() - 1) as f64)
                .sqrt();
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(BLUE, "Reference / RK4");
                ui.colored_label(ORANGE, "Neural prediction");
                ui.separator();
                ui.label(format!("t = {:.2} s", frame.t));
                ui.separator();
                ui.label(format!("10 s angle RMSE = {rmse:.4} rad"));
            });
            egui::ScrollArea::vertical().show(ui, |ui| {
                let width = ui.available_width();
                let sim_height = ui.available_height().clamp(180.0, 290.0);
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(width, sim_height), egui::Sense::hover());
                pendulum(ui, rect, frame.reference[0], BLUE, 0.28, "RK4 reference");
                pendulum(ui, rect, frame.neural[0], ORANGE, 0.73, "Learned dynamics");
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!(
                        "Angle: {:.4} / {:.4} rad",
                        frame.reference[0], frame.neural[0]
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Current angle error: {:.5} rad",
                        (frame.neural[0] - frame.reference[0]).abs()
                    ));
                });
                ui.add_space(10.0);
                ui.columns(2, |columns| {
                    columns[0].strong("Angle over time / rad");
                    let time_ref: Vec<_> =
                        self.trace.iter().map(|f| [f.t, f.reference[0]]).collect();
                    let time_nn: Vec<_> = self.trace.iter().map(|f| [f.t, f.neural[0]]).collect();
                    chart(&mut columns[0], &time_ref, &time_nn, Some(frame.t), false);
                    columns[1].strong("Phase portrait / rad, rad/s");
                    let phase_ref: Vec<_> = self.trace.iter().map(|f| f.reference).collect();
                    let phase_nn: Vec<_> = self.trace.iter().map(|f| f.neural).collect();
                    chart(&mut columns[1], &phase_ref, &phase_nn, None, true);
                });
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{} steps  /  f64  /  autonomous rollout",
                        self.trace.len() - 1
                    ))
                    .size(12.0)
                    .color(MUTED),
                );
            });
        });
    }
}
fn pendulum(ui: &egui::Ui, rect: Rect, angle: f64, color: Color32, fraction: f32, label: &str) {
    let painter = ui.painter_at(rect);
    let origin = Pos2::new(rect.left() + rect.width() * fraction, rect.top() + 34.0);
    let length = (rect.height() * 0.64).min(rect.width() * 0.215);
    let bob = origin + Vec2::new(angle.sin() as f32, angle.cos() as f32) * length;
    painter.line_segment(
        [
            origin - Vec2::new(0.0, 8.0),
            origin + Vec2::new(0.0, length + 16.0),
        ],
        Stroke::new(1.0_f32, Color32::from_gray(225)),
    );
    painter.line_segment(
        [origin - Vec2::new(28.0, 0.0), origin + Vec2::new(28.0, 0.0)],
        Stroke::new(4.0_f32, INK),
    );
    painter.line_segment([origin, bob], Stroke::new(3.5_f32, color));
    painter.circle_filled(origin, 4.0, INK);
    painter.circle_filled(bob, 13.0, color);
    painter.text(
        Pos2::new(origin.x, rect.bottom() - 20.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(14.0),
        color,
    );
}
fn chart(
    ui: &mut egui::Ui,
    reference: &[[f64; 2]],
    neural: &[[f64; 2]],
    cursor: Option<f64>,
    phase: bool,
) {
    let (outer, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 203.0), egui::Sense::hover());
    let r = Rect::from_min_max(
        outer.min + Vec2::new(43.0, 12.0),
        outer.max - Vec2::new(22.0, 29.0),
    );
    let painter = ui.painter_at(outer);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for point in reference.iter().chain(neural) {
        for i in 0..2 {
            lo[i] = lo[i].min(point[i]);
            hi[i] = hi[i].max(point[i]);
        }
    }
    for i in 0..2 {
        let pad = ((hi[i] - lo[i]) * 0.08).max(0.05);
        lo[i] -= pad;
        hi[i] += pad;
    }
    if !phase {
        lo[0] = 0.0;
        hi[0] = 10.0;
    }
    let point = |p: [f64; 2]| {
        Pos2::new(
            r.left() + ((p[0] - lo[0]) / (hi[0] - lo[0])) as f32 * r.width(),
            r.bottom() - ((p[1] - lo[1]) / (hi[1] - lo[1])) as f32 * r.height(),
        )
    };
    for k in 0..=4 {
        let f = k as f32 / 4.0;
        let y = r.top() + f * r.height();
        let x = r.left() + f * r.width();
        painter.line_segment(
            [Pos2::new(r.left(), y), Pos2::new(r.right(), y)],
            Stroke::new(1.0_f32, Color32::from_gray(226)),
        );
        painter.line_segment(
            [Pos2::new(x, r.top()), Pos2::new(x, r.bottom())],
            Stroke::new(1.0_f32, Color32::from_gray(235)),
        );
        painter.text(
            Pos2::new(r.left() - 6.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{:.1}", hi[1] - (hi[1] - lo[1]) * f as f64),
            egui::FontId::monospace(10.5),
            MUTED,
        );
        painter.text(
            Pos2::new(x, r.bottom() + 12.0),
            egui::Align2::CENTER_CENTER,
            format!("{:.1}", lo[0] + (hi[0] - lo[0]) * f as f64),
            egui::FontId::monospace(10.5),
            MUTED,
        );
    }
    for (series, color) in [(reference, BLUE), (neural, ORANGE)] {
        painter.add(egui::Shape::line(
            series.iter().map(|&p| point(p)).collect(),
            Stroke::new(1.7_f32, color),
        ));
    }
    if let Some(t) = cursor {
        let x = point([t, 0.0]).x;
        painter.line_segment(
            [Pos2::new(x, r.top()), Pos2::new(x, r.bottom())],
            Stroke::new(1.0_f32, INK),
        );
    }
}

/// Open an embedded or custom checkpoint. Invalid models remain visible as errors.
pub fn launch(path: Option<&Path>, size: [f32; 2], paused: bool) -> Result<()> {
    launch_with_capture(path, size, paused, None)
}

/// Same native UI with optional deterministic snapshot and automatic exit.
pub fn launch_with_capture(
    path: Option<&Path>,
    size: [f32; 2],
    paused: bool,
    capture: Option<Capture>,
) -> Result<()> {
    anyhow::ensure!(
        size.iter()
            .all(|v| v.is_finite() && *v >= 320.0 && *v <= 4096.0),
        "Invalid viewport size"
    );
    if let Some(c) = &capture {
        anyhow::ensure!(
            c.time.is_finite() && (0.0..=10.0).contains(&c.time) && c.preset < 3,
            "Invalid capture time or preset"
        );
    }
    let status: CaptureStatus = Arc::new(Mutex::new(None));
    let result_status = status.clone();
    let capture_requested = capture.is_some();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(size)
            .with_min_inner_size([960.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Project 11 - Pendulum Dynamics",
        options,
        Box::new(move |cc| Ok(Box::new(Lab::new(cc, path, paused, capture, status)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
    .context("Cannot open desktop window")?;
    if capture_requested {
        let result = result_status
            .lock()
            .map_err(|_| anyhow::anyhow!("Capture lock poisoned"))?
            .take();
        match result {
            Some(Ok(())) => {}
            Some(Err(error)) => anyhow::bail!(error),
            None => anyhow::bail!("Screenshot was not captured"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedded_rollout_uses_only_its_own_previous_predictions() {
        let model = Model::from_json(include_str!("../models/main.json")).unwrap();
        let f = frames(&model, [0.7, 0.2], 10).unwrap();
        for pair in f.windows(2) {
            assert_eq!(pair[1].neural, model.next(pair[0].neural).unwrap());
            assert_eq!(
                pair[1].reference,
                model.physics.reference_step(pair[0].reference)
            );
        }
    }
}
