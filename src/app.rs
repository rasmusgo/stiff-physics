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

use crate::{
    audio_player::AudioPlayer,
    stiff_physics::{create_diff_eq_system_around_y0, new_state_vector_from_points, Spring, D},
};

const SPRINGS: [Spring; 3] = [
    Spring {
        p1: 0,
        p2: 1,
        length: 1.0,
        k: 500.0,
        d: 0.1,
    },
    Spring {
        p1: 1,
        p2: 2,
        length: 1.0,
        k: 5000.0,
        d: 0.1,
    },
    Spring {
        p1: 2,
        p2: 0,
        length: 1.0,
        k: 200.0,
        d: 0.1,
    },
];

pub struct StiffPhysicsApp {
    point_mass: f32,
    points: Vec<Point2<f64>>,
    springs: Vec<Spring>,
    relaxed_points: Vec<Point2<f64>>,
    simulation_state: DVector<f64>,
    exp_a_sim_step: DMatrix<f64>,
    exp_a_audio_step: DMatrix<f64>,
    enable_simulation: bool,
    pub audio_player: Arc<Mutex<Option<anyhow::Result<AudioPlayer>>>>,
    audio_history: Vec<(f32, f32, f32, f32)>,
    audio_history_index: usize,
    audio_history_resolution: usize,
    state_vector_producer: rtrb::Producer<DVector<f64>>,
    state_vector_consumer: rtrb::Consumer<DVector<f64>>,
    grabbed_point: Option<usize>,
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

        // Dummy short-circuit producer/consumer
        let (state_vector_producer, state_vector_consumer) = rtrb::RingBuffer::new(1);

        Self {
            point_mass: 0.01,
            points,
            springs,
            relaxed_points,
            simulation_state: DVector::zeros(system_size),
            exp_a_sim_step: DMatrix::zeros(system_size, system_size),
            exp_a_audio_step: DMatrix::zeros(system_size, system_size),
            enable_simulation: false,
            audio_player: Default::default(),
            audio_history: Vec::new(),
            audio_history_index: 0,
            audio_history_resolution: 1000,
            state_vector_producer,
            state_vector_consumer,
            grabbed_point: None,
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
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        puffin::profile_function!();
        puffin::GlobalProfiler::lock().new_frame();
        if puffin::are_scopes_on() && !puffin_egui::profiler_window(ctx) {
            puffin::set_scopes_on(false);
        }

        let Self {
            point_mass,
            points,
            springs,
            relaxed_points,
            simulation_state,
            exp_a_sim_step,
            exp_a_audio_step,
            enable_simulation,
            audio_player,
            audio_history,
            audio_history_index,
            audio_history_resolution,
            state_vector_producer,
            state_vector_consumer,
            grabbed_point,
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
            // Get updated simulation state from the audio thread.
            if let Ok(state_storage) = state_vector_consumer.pop() {
                // Pass back the old vector so that the audio thread doesn't need to allocate.
                state_vector_producer
                    .push(mem::replace(simulation_state, state_storage))
                    .unwrap();
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            if ui.button("Start profiler").clicked() {
                puffin::set_scopes_on(true);
            }
            for spring in &mut *springs {
                ui.add(egui::Slider::new(&mut spring.k, 0.0..=100000.0).text("spring_constant"));
                ui.add(egui::Slider::new(&mut spring.d, 0.0..=100000.0).text("damping"));
            }
            ui.add(egui::Slider::new(point_mass, 0.01..=10.0).text("point_mass"));

            ui.horizontal(|ui| {
                if ui.button("Store as relaxed").clicked() {
                    for spring in &mut *springs {
                        spring.length = (points[spring.p1] - points[spring.p2]).norm();
                    }
                    *relaxed_points = points.clone();
                }
                if ui.button("Load relaxed").clicked() {
                    *points = relaxed_points.clone();
                }
            });

            let clicked_simulate = ui
                .vertical_centered_justified(|ui| {
                    ui.add(
                        egui::Button::new("\nSimulate\n")
                            .text_style(egui::TextStyle::Heading)
                            .fill(egui::Color32::GREEN),
                    )
                    .clicked()
                })
                .inner;
            if clicked_simulate {
                puffin::profile_scope!("start simulation");
                let point_masses = [*point_mass as f64].repeat(points.len());
                let y0 = new_state_vector_from_points(relaxed_points);
                let mat_a = create_diff_eq_system_around_y0(&y0, &point_masses, springs);
                let p0_vel_loc = points.len() * D;
                *simulation_state = new_state_vector_from_points(points);
                {
                    puffin::profile_scope!("exp_a_sim_step");
                    *exp_a_sim_step = (mat_a.clone() * 0.01).exp();
                }
                {
                    puffin::profile_scope!("exp_a_audio_step");
                    *exp_a_audio_step = (mat_a.clone() / 44100.0).exp();
                }
                *enable_simulation = true;
                // println!("{:?}", mat_a);
                // println!("{:?}", &*mat_a * &*simulation_state);

                // Create communication channel to send back simulation state to UI.
                //      UI thread               Audio thread
                //      ---------               ------------
                //  Allocate state vector            .
                //         |                         .
                // to_audio_producer -------> to_audio_consumer
                //         .                         |
                //         .               Swap with up to date vector
                //         .                         |
                //   to_ui_consumer <--------- to_ui_producer
                //         |                         .
                //    Draw graphics                  .
                let (to_audio_producer, mut to_audio_consumer) = rtrb::RingBuffer::new(1);
                let (mut to_ui_producer, to_ui_consumer) = rtrb::RingBuffer::new(1);

                // Store these so we can communicate with the audio thread.
                *state_vector_producer = to_audio_producer;
                *state_vector_consumer = to_ui_consumer;

                // Create some state vector storage to pass back and forth.
                state_vector_producer
                    .push(simulation_state.clone())
                    .unwrap();

                // Send audio generator functor to audio player
                let mut audio_player_ref = audio_player.lock();
                if audio_player_ref.is_none() {
                    *audio_player_ref = Some(AudioPlayer::new());
                }
                if let Some(Ok(player)) = audio_player_ref.as_mut() {
                    puffin::profile_scope!("create sampling function");
                    let sample_rate = player.config.sample_rate().0 as f64;
                    let exp_a_audio_step = {
                        puffin::profile_scope!("exp(mat_a / sample_rate)");
                        (mat_a / sample_rate).exp()
                    };
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
                        if !to_ui_producer.is_full() {
                            if let Ok(mut state_storage) = to_audio_consumer.pop() {
                                // Put an (almost) up to state vector in the provided storage and send it back.
                                mem::swap(&mut state_storage, &mut y_next);
                                to_ui_producer.push(state_storage).unwrap();
                            }
                        }
                        value
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

                ui.add(
                    egui::Slider::new(audio_history_resolution, 10..=sample_rate)
                        .text("Graph resolution"),
                );
                let step = (audio_history.len() / *audio_history_resolution).max(1);
                let (left, right) = audio_history.split_at(*audio_history_index);
                let iter = right.iter().chain(left).enumerate().step_by(step);
                let line_raw = Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                    Value::new(
                        (i as f64 - audio_history.len() as f64) / sample_rate as f64,
                        val.0 as f64,
                    )
                })));
                let line_filtered =
                    Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                        Value::new(
                            (i as f64 - audio_history.len() as f64) / sample_rate as f64,
                            val.1 as f64,
                        )
                    })));
                let line_power_raw =
                    Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                        Value::new(
                            (i as f64 - audio_history.len() as f64) / sample_rate as f64,
                            val.2 as f64,
                        )
                    })));
                let line_power_filtered =
                    Line::new(Values::from_values_iter(iter.map(|(i, val)| {
                        Value::new(
                            (i as f64 - audio_history.len() as f64) / sample_rate as f64,
                            val.3 as f64,
                        )
                    })));
                ui.add(
                    Plot::new("Audio")
                        .line(line_raw)
                        .line(line_filtered)
                        .line(line_power_raw)
                        .line(line_power_filtered)
                        .view_aspect(1.0)
                        .include_x(-1.0)
                        .include_x(0.0)
                        .include_y(-1.0)
                        .include_y(1.0),
                );
                let num_samples = player.num_frames_per_callback.load(Ordering::Relaxed);
                ui.label(format!(
                    "{:?} samples = {:.3} ms = {:.2} hz",
                    num_samples,
                    1000.0 * (num_samples as f32) / (player.config.sample_rate().0 as f32),
                    (player.config.sample_rate().0 as f32) / (num_samples as f32),
                ));
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
                if response.drag_started() {
                    let mut best_norm2 = 15. * 15.;
                    for (i, &mut p) in points.iter_mut().enumerate() {
                        let point_in_pixels = c + vec2(p.x as f32, p.y as f32) * r;
                        let diff = pos - point_in_pixels;
                        let norm2 = diff.x * diff.x + diff.y * diff.y;
                        if norm2 <= best_norm2 {
                            best_norm2 = norm2;
                            *grabbed_point = Some(i);
                        }
                    }
                }
                if response.drag_released() {
                    *grabbed_point = None;
                }
                if let Some(i) = *grabbed_point {
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
                draw_particle_system(
                    &simulated_points[..],
                    &*springs,
                    line_width * 3.0,
                    &painter,
                    c,
                    r,
                );
            }
            draw_particle_system(&relaxed_points[..], &*springs, line_width, &painter, c, r);
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
