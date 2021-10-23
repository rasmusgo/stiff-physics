use std::{
    mem,
    sync::{atomic::Ordering, Arc},
};

use eframe::{
    egui::{self, mutex::Mutex, vec2, Color32, Sense, Stroke},
    epi,
};
use egui::plot::{Line, Plot, Value, Values};
use nalgebra::{self, DMatrix, DVector, Point2};

use crate::audio_player::AudioPlayer;
use crate::stiff_physics::{create_diff_eq_system, Spring, D};

const SPRINGS: [Spring; 3] = [
    Spring {
        p1: 0,
        p2: 1,
        length: 1.0,
        k: 1000.0,
        d: 0.1,
    },
    Spring {
        p1: 1,
        p2: 2,
        length: 1.0,
        k: 1000.0,
        d: 0.1,
    },
    Spring {
        p1: 2,
        p2: 0,
        length: 1.0,
        k: 1000.0,
        d: 0.1,
    },
];

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))] // if we add new fields, give them default values when deserializing old state
pub struct StiffPhysicsApp {
    point_mass: f32,
    relaxation_iterations: usize,
    points: Vec<Point2<f64>>,
    springs: Vec<Spring>,
    relaxed_points: Vec<Point2<f64>>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    mat_a: DMatrix<f64>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    simulation_state: DVector<f64>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    exp_a_sim_step: DMatrix<f64>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    exp_a_audio_step: DMatrix<f64>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    enable_simulation: bool,
    #[cfg_attr(feature = "persistence", serde(skip))]
    pub audio_player: Arc<Mutex<Option<anyhow::Result<AudioPlayer>>>>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    audio_history: Vec<(f32, f32)>,
    #[cfg_attr(feature = "persistence", serde(skip))]
    audio_history_index: usize,
}

impl Default for StiffPhysicsApp {
    fn default() -> Self {
        let points = vec![
            Point2::new(-0.6, 0.2),
            Point2::new(0., -(0.75_f64.sqrt()) + 0.2),
            Point2::new(0.6, 0.2),
        ];
        let springs = SPRINGS.to_vec();
        let relaxed_points = vec![
            Point2::new(-0.5, 0.2),
            Point2::new(0.0, -(0.75_f64.sqrt()) + 0.2),
            Point2::new(0.5, 0.2),
        ];
        assert_eq!(points.len(), relaxed_points.len());
        let num_points = points.len();
        let block_size = num_points * D;
        let system_size = block_size * 2 + 1;

        Self {
            point_mass: 0.001,
            relaxation_iterations: 1,
            points,
            springs,
            relaxed_points,
            mat_a: DMatrix::zeros(system_size, system_size),
            simulation_state: DVector::zeros(system_size),
            exp_a_sim_step: DMatrix::zeros(system_size, system_size),
            exp_a_audio_step: DMatrix::zeros(system_size, system_size),
            enable_simulation: false,
            audio_player: Default::default(),
            audio_history: Vec::new(),
            audio_history_index: 0,
        }
    }
}

impl epi::App for StiffPhysicsApp {
    fn name(&self) -> &str {
        "Stiff Physics"
    }

    /// Called once before the first frame.
    fn setup(
        &mut self,
        _ctx: &egui::CtxRef,
        _frame: &mut epi::Frame<'_>,
        _storage: Option<&dyn epi::Storage>,
    ) {
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        #[cfg(feature = "persistence")]
        if let Some(storage) = _storage {
            *self = epi::get_value(storage, epi::APP_KEY).unwrap_or_default()
        }
    }

    /// Called by the frame work to save state before shutdown.
    /// Note that you must enable the `persistence` feature for this to work.
    #[cfg(feature = "persistence")]
    fn save(&mut self, storage: &mut dyn epi::Storage) {
        epi::set_value(storage, epi::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        let Self {
            point_mass,
            relaxation_iterations,
            points,
            springs,
            relaxed_points,
            mat_a,
            simulation_state,
            exp_a_sim_step,
            exp_a_audio_step,
            enable_simulation,
            audio_player,
            audio_history,
            audio_history_index,
        } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        // egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
        //     // The top panel is often a good place for a menu bar:
        //     egui::menu::bar(ui, |ui| {
        //         egui::menu::menu(ui, "File", |ui| {
        //             if ui.button("Quit").clicked() {
        //                 frame.quit();
        //             }
        //         });
        //     });
        // });

        if *enable_simulation {
            // Advance simulation
            *simulation_state = &*exp_a_sim_step * &*simulation_state;
            ctx.request_repaint();
        }

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            for spring in &mut *springs {
                ui.add(egui::Slider::new(&mut spring.k, 0.0..=1000.0).text("spring_constant"));
                ui.add(egui::Slider::new(&mut spring.d, 0.0..=1000.0).text("damping"));
            }
            ui.add(egui::Slider::new(point_mass, 0.01..=10.0).text("point_mass"));

            if ui.button("Store as relaxed").clicked() {
                for spring in &mut *springs {
                    let p1 = points[spring.p1];
                    let p2 = points[spring.p2];
                    let diff = p1 - p2;
                    let norm2 = diff.x * diff.x + diff.y * diff.y;
                    spring.length = norm2.sqrt();
                }
                *relaxed_points = points.clone();
            }

            ui.add(
                egui::Slider::new(relaxation_iterations, 0..=10)
                    .text("Relaxation iterations before linearization"),
            );

            let point_masses = [*point_mass as f64].repeat(points.len());
            let (mut a, y0) = create_diff_eq_system(points, &point_masses, &*springs);

            // Re-linearize around estimated relaxed state
            for _i in 0..*relaxation_iterations {
                // Solve for zero velocity, zero acceleration and minimal distance to y0 (TODO: weighted by mass)
                // https://math.stackexchange.com/questions/1318637/projection-of-a-vector-onto-the-null-space-of-a-matrix
                let y_relaxed = &y0
                    - &a.tr_mul(&((&a * &a.transpose()).pseudo_inverse(1.0e-10).unwrap() * &a))
                        * &y0;
                relaxed_points.clear();
                for i in 0..points.len() {
                    relaxed_points.push(Point2::new(y_relaxed[i * D], y_relaxed[i * D + 1]));
                }
                a = create_diff_eq_system(relaxed_points.as_slice(), &point_masses, &*springs).0;
            }
            *mat_a = a;

            if ui.button("Simulate").clicked() {
                *simulation_state = y0;
                *exp_a_sim_step = (mat_a.clone() * 0.01).exp();
                *exp_a_audio_step = (mat_a.clone() / 44100.0).exp();
                *enable_simulation = true;
                // println!("{:?}", mat_a);
                // println!("{:?}", &*mat_a * &*simulation_state);

                // Generate audio to find max_value for normalization
                let p0_vel_loc = points.len() * D;
                let mut y = simulation_state.clone();
                let mut max_value: f32 = 0.0;
                for _i in 0..44100 * 3 {
                    let y_next = &*exp_a_audio_step * &y;
                    let value = (y_next[p0_vel_loc] - y[p0_vel_loc]) as f32;
                    max_value = f32::max(max_value, value.abs());
                    y = y_next;
                }

                // Send audio generator functor to audio player
                let mut audio_player_ref = audio_player.lock();
                if audio_player_ref.is_none() {
                    *audio_player_ref = Some(AudioPlayer::new());
                }
                if let Some(Ok(player)) = audio_player_ref.as_mut() {
                    let sample_rate = player.config.sample_rate().0 as f64;
                    let exp_a_audio_step = (mat_a.clone() / sample_rate).exp();
                    let fade_in_rate = 50.0 / sample_rate as f32;

                    // Produce a waveform by advancing the simulation.
                    let mut y = simulation_state.clone();
                    let mut y_next = simulation_state.clone();
                    let mut fade = 0.0;
                    let next_sample = move || {
                        y_next.gemv(1.0, &exp_a_audio_step, &y, 0.0);
                        let value = fade * ((y_next[p0_vel_loc] - y[p0_vel_loc]) as f32);
                        mem::swap(&mut y, &mut y_next);
                        if fade < 1.0 {
                            fade = (fade + fade_in_rate).min(1.0);
                        }
                        value / max_value
                    };

                    player.play_audio(Box::new(next_sample)).unwrap();
                }
            }

            if let Some(Ok(player)) = audio_player.lock().as_mut() {
                let mut enable_band_pass_filter =
                    player.enable_band_pass_filter.load(Ordering::Relaxed);
                if ui
                    .checkbox(&mut enable_band_pass_filter, "Enable band-pass filter")
                    .clicked()
                {
                    player
                        .enable_band_pass_filter
                        .fetch_xor(true, Ordering::SeqCst);
                }
                let sample_rate = player.config.sample_rate().0 as usize;
                if let Some(consumer) = &mut player.to_ui_consumer {
                    while let Ok(data) = consumer.pop() {
                        if audio_history.len() < sample_rate {
                            audio_history.push(data);
                        } else {
                            audio_history[*audio_history_index] = data;
                            *audio_history_index = (*audio_history_index + 1) % audio_history.len();
                        }
                    }
                }

                let (left, right) = audio_history.split_at(*audio_history_index);
                let line_raw =
                    Line::new(Values::from_values_iter(
                        right.iter().chain(left).enumerate().map(|(i, val)| {
                            Value::new(i as f64 / sample_rate as f64, val.0 as f64)
                        }),
                    ));
                let line_filtered =
                    Line::new(Values::from_values_iter(
                        right.iter().chain(left).enumerate().map(|(i, val)| {
                            Value::new(i as f64 / sample_rate as f64, val.1 as f64)
                        }),
                    ));
                ui.add(
                    Plot::new("Audio")
                        .line(line_raw)
                        .line(line_filtered)
                        .view_aspect(1.0),
                );
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                ui.add(
                    egui::Hyperlink::new("https://github.com/emilk/egui/").text("powered by egui"),
                );
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's

            ui.heading("Stiff physics");

            let size = vec2(ui.available_width(), ui.available_width());
            let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
            let rect = response.rect;
            let c = rect.center();
            let r = rect.width() / 2. - 1.;
            let line_width = f32::max(r / 500., 1.0);

            if let Some(pos) = response.interact_pointer_pos() {
                let mut best_norm2 = 15. * 15.;
                let mut best_point = None;
                for (i, &mut p) in points.iter_mut().enumerate() {
                    let point_in_pixels = c + vec2(p.x as f32, p.y as f32) * r;
                    let diff = pos - point_in_pixels;
                    let norm2 = diff.x * diff.x + diff.y * diff.y;
                    if norm2 <= best_norm2 {
                        best_norm2 = norm2;
                        best_point = Some(i);
                    }
                }
                if let Some(i) = best_point {
                    let p = (pos - c) / r;
                    points[i].x = p.x as f64;
                    points[i].y = p.y as f64;
                }
            }
            if *enable_simulation {
                let num_simulated_points = simulation_state.len() / 4;
                let mut simulated_points = Vec::<Point2<f64>>::with_capacity(num_simulated_points);
                for i in 0..num_simulated_points {
                    simulated_points.push(Point2::new(
                        simulation_state[i * D],
                        simulation_state[i * D + 1],
                    ));
                }
                draw_particle_system(&simulated_points[..], &*springs, line_width, &painter, c, r);
            }
            if *relaxation_iterations > 0 {
                draw_particle_system(&relaxed_points[..], &*springs, line_width, &painter, c, r);
            }
            draw_particle_system(points, &*springs, line_width, &painter, c, r);

            ui.hyperlink("https://github.com/rasmusgo/stiff-physics");
            ui.add(egui::github_link_file!(
                "https://github.com/rasmusgo/stiff-physics/blob/master/",
                "Source code."
            ));
            egui::warn_if_debug_build(ui);
        });

        if false {
            egui::Window::new("Window").show(ctx, |ui| {
                ui.label("Windows can be moved by dragging them.");
                ui.label("They are automatically sized based on contents.");
                ui.label("You can turn on resizing and scrolling if you like.");
                ui.label("You would normally chose either panels OR windows.");
            });
        }
    }
}

fn draw_particle_system(
    points: &[Point2<f64>],
    springs: &[Spring],
    line_width: f32,
    painter: &egui::Painter,
    c: egui::Pos2,
    r: f32,
) {
    let circle_radius = r / 100.;
    let color = Color32::BLACK;
    for spring in springs {
        let p1 = points[spring.p1];
        let p2 = points[spring.p2];
        let diff = p1 - p2;
        let stress = f64::min((spring.length - diff.norm()).abs() / spring.length, 1.0);
        let line_color = egui::lerp(egui::Rgba::GREEN..=egui::Rgba::RED, stress as f32);
        let stroke = Stroke::new(line_width, line_color);
        let p1 = vec2(p1.x as f32, p1.y as f32);
        let p2 = vec2(p2.x as f32, p2.y as f32);
        painter.line_segment([c + p1 * r, c + p2 * r], stroke);
    }
    for &p in points {
        let p = vec2(p.x as f32, p.y as f32);
        painter.circle_filled(c + p * r, circle_radius, color);
    }
}
