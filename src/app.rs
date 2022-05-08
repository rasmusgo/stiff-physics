use std::{
    mem,
    sync::{atomic::Ordering, Arc},
};

use eframe::{
    egui::{self, mutex::Mutex, vec2, Color32, Sense, Stroke},
    epi,
};
use egui::plot::{Line, Plot, VLine, Value, Values};
use nalgebra::{self, DMatrix, DVector, Matrix2, Point2, Vector2};

use crate::{
    audio_player::AudioPlayer,
    stiff_physics::{create_diff_eq_system_around_y0, new_state_vector_from_points, Spring, D},
};

const SPRINGS: [Spring; 3] = [
    Spring {
        p1: 0,
        p2: 1,
        length: 1.0,
        k: 5000.0,
        d: 0.5,
    },
    Spring {
        p1: 1,
        p2: 2,
        length: 1.0,
        k: 5000.0,
        d: 0.5,
    },
    Spring {
        p1: 2,
        p2: 0,
        length: 1.0,
        k: 10000.0,
        d: 0.5,
    },
];

const MAP_SIZE: f64 = 5.0;
const MAP_HALF_SIZE: f64 = MAP_SIZE / 2.0;

#[derive(Clone, Copy)]
enum GrabbedPoint {
    None,
    PointMass(usize),
    SimulatedPoint(usize),
    Listener,
}

pub struct StiffPhysicsApp {
    listener_pos: Point2<f64>,
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
    mouse_event_producer: rtrb::Producer<(f64, f64, GrabbedPoint)>,
    grabbed_point: GrabbedPoint,
}

impl Default for StiffPhysicsApp {
    fn default() -> Self {
        let listener_pos = Point2::new(0.0, 1.0);

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
        let (mouse_event_producer, _) = rtrb::RingBuffer::new(1);

        Self {
            listener_pos,
            point_mass: 0.1,
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
            mouse_event_producer,
            grabbed_point: GrabbedPoint::None,
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            *self.audio_player.lock() = Some(AudioPlayer::new());
        }
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        puffin::profile_function!();
        puffin::GlobalProfiler::lock().new_frame();
        #[cfg(feature = "profiler")]
        if puffin::are_scopes_on() && !puffin_egui::profiler_window(ctx) {
            puffin::set_scopes_on(false);
        }

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

        if self.enable_simulation {
            // Get updated simulation state from the audio thread.
            if let Ok(state_storage) = self.state_vector_consumer.pop() {
                // Pass back the old vector so that the audio thread doesn't need to allocate.
                self.state_vector_producer
                    .push(mem::replace(&mut self.simulation_state, state_storage))
                    .unwrap();
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Settings");

            #[cfg(feature = "profiler")]
            if ui.button("Start profiler").clicked() {
                puffin::set_scopes_on(true);
            }
            for spring in &mut self.springs {
                ui.add(egui::Slider::new(&mut spring.k, 0.0..=100000.0).text("spring_constant"));
                ui.add(egui::Slider::new(&mut spring.d, 0.0..=10.0).text("damping"));
            }
            ui.add(egui::Slider::new(&mut self.point_mass, 0.01..=10.0).text("point_mass"));

            ui.horizontal(|ui| {
                if ui.button("Store as relaxed").clicked() {
                    for spring in &mut *self.springs {
                        spring.length = (self.points[spring.p1] - self.points[spring.p2]).norm();
                    }
                    self.relaxed_points = self.points.clone();
                }
                if ui.button("Load relaxed").clicked() {
                    self.points = self.relaxed_points.clone();
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
            if clicked_simulate || ctx.input().key_pressed(egui::Key::Space) {
                self.start_simulation();
            }

            if let Some(Ok(player)) = self.audio_player.lock().as_mut() {
                ui.heading("Audio");

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
                while let Some(data) = player.get_audio_history_entry() {
                    if self.audio_history.len() < sample_rate {
                        self.audio_history.push(data);
                        self.audio_history_index = self.audio_history.len() % sample_rate;
                    } else {
                        self.audio_history[self.audio_history_index] = data;
                        self.audio_history_index =
                            (self.audio_history_index + 1) % self.audio_history.len();
                    }
                }

                ui.add(
                    egui::Slider::new(&mut self.audio_history_resolution, 10..=sample_rate)
                        .text("Graph resolution"),
                );
                let step = (self.audio_history.len() / self.audio_history_resolution).max(1);
                let iter = self.audio_history.iter().enumerate().step_by(step);
                let line_raw =
                    Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                        Value::new((i as f64) / sample_rate as f64, val.0 as f64)
                    })));
                let line_filtered =
                    Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                        Value::new((i as f64) / sample_rate as f64, val.1 as f64)
                    })));
                let line_power_raw =
                    Line::new(Values::from_values_iter(iter.clone().map(|(i, val)| {
                        Value::new((i as f64) / sample_rate as f64, val.2 as f64)
                    })));
                let line_power_filtered =
                    Line::new(Values::from_values_iter(iter.map(|(i, val)| {
                        Value::new((i as f64) / sample_rate as f64, val.3 as f64)
                    })));
                let vline = VLine::new(self.audio_history_index as f64 / sample_rate as f64);
                ui.add(
                    Plot::new("Audio")
                        .line(line_raw)
                        .line(line_filtered)
                        .line(line_power_raw)
                        .line(line_power_filtered)
                        .view_aspect(1.0)
                        .vline(vline)
                        .include_x(0.0)
                        .include_x(1.0)
                        .center_y_axis(true),
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

            ui.heading("Simulation");

            let size = vec2(ui.available_width(), ui.available_width());
            let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
            let rect = response.rect;
            let c = rect.center();
            let r = rect.width() / MAP_SIZE as f32 - MAP_HALF_SIZE as f32;
            let line_width = f32::max(r / 500., 1.0);

            if let Some(pos) = response.interact_pointer_pos() {
                if response.drag_started() {
                    let grabbed_point = &mut self.grabbed_point;
                    let mut best_norm2 = 15. * 15.;
                    let mut test_point = |p: &Point2<f64>, id: GrabbedPoint| {
                        let point_in_pixels = c + vec2(p.x as f32, p.y as f32) * r;
                        let diff = pos - point_in_pixels;
                        let norm2 = diff.x * diff.x + diff.y * diff.y;
                        if norm2 <= best_norm2 {
                            best_norm2 = norm2;
                            *grabbed_point = id;
                        }
                    };
                    if self.enable_simulation {
                        let num_simulated_points = self.simulation_state.len() / 4;
                        for i in 0..num_simulated_points {
                            test_point(
                                &Point2::new(
                                    self.simulation_state[i * D],
                                    self.simulation_state[i * D + 1],
                                ),
                                GrabbedPoint::SimulatedPoint(i),
                            );
                        }
                    }
                    for (i, p) in self.points.iter().enumerate() {
                        test_point(p, GrabbedPoint::PointMass(i));
                    }
                    test_point(&self.listener_pos, GrabbedPoint::Listener);
                }
                if response.drag_released() {
                    self.grabbed_point = GrabbedPoint::None;
                }
                let p = (pos - c) / r;
                match self.grabbed_point {
                    GrabbedPoint::PointMass(i) => {
                        self.points[i].x = p.x as f64;
                        self.points[i].y = p.y as f64;
                    }
                    GrabbedPoint::Listener => {
                        self.listener_pos.x = p.x as f64;
                        self.listener_pos.y = p.y as f64;
                        if let Some(Ok(player)) = self.audio_player.lock().as_mut() {
                            player.set_listener_pos(self.listener_pos).unwrap();
                        }
                    }
                    _ => self
                        .mouse_event_producer
                        .push((p.x as f64, p.y as f64, self.grabbed_point))
                        .unwrap_or_default(),
                }
            }
            if self.enable_simulation {
                let num_simulated_points = self.simulation_state.len() / 4;
                let mut simulated_points = Vec::<Point2<f64>>::with_capacity(num_simulated_points);
                for i in 0..num_simulated_points {
                    simulated_points.push(Point2::new(
                        self.simulation_state[i * D],
                        self.simulation_state[i * D + 1],
                    ));
                }
                draw_particle_system(
                    &simulated_points[..],
                    &self.springs,
                    line_width * 3.0,
                    &painter,
                    c,
                    r,
                );
            }
            draw_particle_system(
                &self.relaxed_points[..],
                &self.springs,
                line_width,
                &painter,
                c,
                r,
            );
            draw_particle_system(&self.points, &self.springs, line_width, &painter, c, r);

            let p = egui::Vec2::new(self.listener_pos.x as f32, self.listener_pos.y as f32);
            let circle_radius = r * 0.1;
            let stroke = Stroke::new(line_width, Color32::BLACK);
            painter.circle(c + p * r, circle_radius, Color32::GOLD, stroke);

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

impl StiffPhysicsApp {
    fn start_simulation(&mut self) {
        puffin::profile_function!();
        let point_masses = [self.point_mass as f64].repeat(self.points.len());
        let y0 = new_state_vector_from_points(&self.relaxed_points);
        let mat_a = create_diff_eq_system_around_y0(&y0, &point_masses, &self.springs);
        self.simulation_state = new_state_vector_from_points(&self.points);
        {
            puffin::profile_scope!("exp_a_sim_step");
            self.exp_a_sim_step = (mat_a.clone() * 0.01).exp();
        }
        {
            puffin::profile_scope!("exp_a_audio_step");
            self.exp_a_audio_step = (mat_a.clone() / 44100.0).exp();
        }
        self.enable_simulation = true;
        let (to_audio_producer, mut to_audio_consumer) = rtrb::RingBuffer::new(1);
        let (mut to_ui_producer, to_ui_consumer) = rtrb::RingBuffer::new(1);
        self.state_vector_producer = to_audio_producer;
        self.state_vector_consumer = to_ui_consumer;
        self.state_vector_producer
            .push(self.simulation_state.clone())
            .unwrap();

        // Precompute demeaned relaxed points to use for determining rotation during simulation.
        let relaxed_points_mean = Point2::from(
            self.relaxed_points
                .iter()
                .map(|&p| p.coords)
                .sum::<Vector2<f64>>()
                / (self.relaxed_points.len() as f64),
        );
        let demeaned_relaxed_points: Vec<Vector2<f64>> = self
            .relaxed_points
            .iter()
            .map(|&p| p - relaxed_points_mean)
            .collect();

        let mut audio_player_ref = self.audio_player.lock();
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

            // Channel for mouse events coming from the UI thread
            let (mouse_event_producer, mut mouse_event_consumer) = rtrb::RingBuffer::new(100);
            self.mouse_event_producer = mouse_event_producer;
            // Get smooth acceleration of mouse by doing moving average twice.
            const MOUSE_SMOOTHING_ALPHA: f64 = 0.001;
            let mut mouse_state = GrabbedPoint::None;
            let mut mouse_target1 = Point2::new(0.0, 0.0);
            let mut mouse_target2 = Point2::new(0.0, 0.0);
            let mut mouse_pos = Point2::new(0.0, 0.0);

            // Keep a history of states in a circular buffer so that we can create a waveform by combining contributions over time.
            const SAMPLES_IN_BUFFER: usize = 1024;
            const SPEED_OF_SOUND: f64 = 343.0;
            let num_points = self.points.len();
            let meters_per_sample = SPEED_OF_SOUND / sample_rate;
            let mut state_history = self
                .simulation_state
                .clone()
                .resize_horizontally(SAMPLES_IN_BUFFER, 0.0);
            let mut acc_history = DMatrix::<f64>::zeros(num_points * D, SAMPLES_IN_BUFFER);
            let mut index_of_newest: usize = 0;
            let mut num_samples_recorded: usize = 1;
            // Storage space for state vectors with rotation undone
            let mut y_unrot = y0.clone();
            let mut y_next_unrot = y0;
            let next_sample = move |update_state: bool, listener_pos: Point2<f64>| {
                if update_state {
                    puffin::profile_scope!("update_state");
                    let read_index = index_of_newest;
                    let write_index = (index_of_newest + 1) % SAMPLES_IN_BUFFER;
                    let (y, mut y_next) =
                        state_history.columns_range_pair_mut(read_index, write_index);
                    let mouse_vel;
                    {
                        // Update mouse position and velocity
                        puffin::profile_scope!("update_mouse");
                        while let Ok(event) = mouse_event_consumer.pop() {
                            mouse_target1[0] = event.0;
                            mouse_target1[1] = event.1;
                            if let GrabbedPoint::None = mouse_state {
                                if let GrabbedPoint::SimulatedPoint(point_index) = event.2 {
                                    let point_pos_loc = point_index * D;
                                    let point_pos =
                                        Point2::from(y.fixed_rows::<D>(point_pos_loc).xy());
                                    mouse_target2 = point_pos;
                                    mouse_pos = point_pos;
                                } else {
                                    mouse_target2 = mouse_target1;
                                    mouse_pos = mouse_target1;
                                }
                            }
                            mouse_state = event.2;
                        }
                        mouse_target2.coords = mouse_target1.coords * MOUSE_SMOOTHING_ALPHA
                            + mouse_target2.coords * (1.0 - MOUSE_SMOOTHING_ALPHA);
                        mouse_vel =
                            (mouse_target2 - mouse_pos) * MOUSE_SMOOTHING_ALPHA * sample_rate;
                        mouse_pos.coords = mouse_target2.coords * MOUSE_SMOOTHING_ALPHA
                            + mouse_pos.coords * (1.0 - MOUSE_SMOOTHING_ALPHA);
                    }
                    {
                        // Advance the simulation and record history
                        puffin::profile_scope!("advance_simulation");

                        // Find rotation compared to relaxed points
                        let mut mean_pos = Vector2::<f64>::zeros();
                        let mut mean_vel = Vector2::<f64>::zeros();
                        for point_index in 0..num_points {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            mean_pos += y.fixed_rows::<D>(point_pos_loc);
                            mean_vel += y.fixed_rows::<D>(point_vel_loc);
                        }
                        mean_pos /= num_points as f64;
                        mean_vel /= num_points as f64;
                        let mut mat_apq = Matrix2::<f64>::zeros();
                        let mut angular_momentum = 0.0;
                        let mut angular_inertia = 0.0;
                        for (point_index, demeaned_relaxed_pos) in
                            demeaned_relaxed_points.iter().enumerate()
                        {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            let demeaned_pos = y.fixed_rows::<D>(point_pos_loc) - mean_pos;
                            let demeaned_vel = y.fixed_rows::<D>(point_vel_loc) - mean_vel;
                            mat_apq += demeaned_pos * demeaned_relaxed_pos.transpose();
                            angular_momentum += demeaned_pos.perp(&demeaned_vel); // The 2D counterpart of a cross product
                            angular_inertia += demeaned_pos.norm_squared();
                        }
                        let angular_velocity = angular_momentum / angular_inertia;

                        // A_pq = ⅀pqᵀ
                        // A_pq = UDVᵀ = U(VᵀV)DVᵀ = (UVᵀ)(VDVᵀ) = RS, R = UVᵀ, S = VDVᵀ
                        let svd = mat_apq.svd(true, true);
                        let mut rot = svd.u.unwrap() * svd.v_t.unwrap();
                        let rot_direction = if rot.determinant() < 0.0 { -1.0 } else { 1.0 };
                        if rot.determinant() < 0.0 {
                            rot.column_mut(svd.singular_values.imin()).neg_mut();
                        }
                        for point_index in 0..num_points {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            rot.tr_mul_to(
                                &(y.fixed_rows::<D>(point_pos_loc) - mean_pos),
                                &mut y_unrot.fixed_rows_mut::<D>(point_pos_loc),
                            );
                            rot.tr_mul_to(
                                &(y.fixed_rows::<D>(point_vel_loc) - mean_vel),
                                &mut y_unrot.fixed_rows_mut::<D>(point_vel_loc),
                            );
                            // Remove angular velocity
                            y_unrot[point_vel_loc] +=
                                y_unrot[point_pos_loc + 1] * angular_velocity * rot_direction;
                            y_unrot[point_vel_loc + 1] -=
                                y_unrot[point_pos_loc] * angular_velocity * rot_direction;
                        }
                        y_next_unrot.gemv(1.0, &exp_a_audio_step, &y_unrot, 0.0);

                        // Apply one iteration of angular and linear velocity
                        rot *= nalgebra::Rotation2::new(angular_velocity / sample_rate);
                        mean_pos += mean_vel / sample_rate;

                        // Compute new inertia and angular momentum in order to properly conserve angular momentum
                        let mut angular_momentum2 = 0.0;
                        let mut angular_inertia2 = 0.0;
                        for point_index in 0..num_points {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            let demeaned_pos = y_next_unrot.fixed_rows::<D>(point_pos_loc);
                            let demeaned_vel = y_next_unrot.fixed_rows::<D>(point_vel_loc);
                            angular_momentum2 += demeaned_pos.perp(&demeaned_vel); // The 2D counterpart of a cross product
                            angular_inertia2 += demeaned_pos.norm_squared();
                        }
                        let angular_velocity_correction =
                            (angular_momentum - angular_momentum2) / angular_inertia2;

                        for point_index in 0..num_points {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;

                            // Conserve angular momentum
                            y_next_unrot[point_vel_loc] -=
                                y_next_unrot[point_pos_loc + 1] * angular_velocity_correction;
                            y_next_unrot[point_vel_loc + 1] +=
                                y_next_unrot[point_pos_loc] * angular_velocity_correction;

                            // Apply new rotation
                            rot.mul_to(
                                &y_next_unrot.fixed_rows::<D>(point_pos_loc),
                                &mut y_next.fixed_rows_mut::<D>(point_pos_loc),
                            );
                            rot.mul_to(
                                &y_next_unrot.fixed_rows::<D>(point_vel_loc),
                                &mut y_next.fixed_rows_mut::<D>(point_vel_loc),
                            );

                            // Apply new position and conserve linear velocity
                            for j in 0..D {
                                y_next[point_pos_loc + j] += mean_pos[j];
                                y_next[point_vel_loc + j] += mean_vel[j];
                            }
                        }

                        // Compute acceleration before collision detection to avoid strong clicks.
                        acc_history.set_column(
                            write_index,
                            &(sample_rate
                                * (y_next.rows(num_points * D, num_points * D)
                                    - y.rows(num_points * D, num_points * D))),
                        );

                        if let GrabbedPoint::SimulatedPoint(point_index) = mouse_state {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            y_next[point_pos_loc] = mouse_pos[0];
                            y_next[point_pos_loc + 1] = mouse_pos[1];
                            y_next[point_vel_loc] = mouse_vel[0];
                            y_next[point_vel_loc + 1] = mouse_vel[1];
                        }
                        for point_index in 0..num_points {
                            let point_pos_loc = point_index * D;
                            let point_vel_loc = num_points * D + point_index * D;
                            for j in 0..D {
                                if y_next[point_pos_loc + j] > MAP_HALF_SIZE {
                                    y_next[point_pos_loc + j] = MAP_HALF_SIZE;
                                    if y_next[point_vel_loc + j] > 0.0 {
                                        y_next[point_vel_loc + j] = 0.0;
                                    }
                                }
                                if y_next[point_pos_loc + j] < -MAP_HALF_SIZE {
                                    y_next[point_pos_loc + j] = -MAP_HALF_SIZE;
                                    if y_next[point_vel_loc + j] < 0.0 {
                                        y_next[point_vel_loc + j] = 0.0;
                                    }
                                }
                            }
                        }

                        index_of_newest = write_index;
                        num_samples_recorded = SAMPLES_IN_BUFFER.min(num_samples_recorded + 1);
                    }
                    if !to_ui_producer.is_full() {
                        puffin::profile_scope!("Send state to UI");
                        if let Ok(mut state_storage) = to_audio_consumer.pop() {
                            // Put an up to date state vector in the provided storage and send it back.
                            state_storage.set_column(0, &state_history.column(index_of_newest));
                            to_ui_producer.push(state_storage).unwrap();
                        }
                    }
                }

                // Traverse history to find the waves that are contributing to what the listener should be hearing right now.
                puffin::profile_scope!("compute_audio_sample");
                let mut value = 0.0;
                for point_index in 0..num_points {
                    let point_pos_loc = point_index * D;

                    // Use latest sample to guess how far back in time we need to go
                    let mut i = {
                        let y = &state_history.column(index_of_newest);
                        let relative_position = Vector2::new(
                            y[point_pos_loc] - listener_pos[0],
                            y[point_pos_loc + 1] - listener_pos[1],
                        );
                        let distance_by_state = relative_position.norm();
                        let guess_i = (distance_by_state / meters_per_sample).ceil() as usize;
                        guess_i.max(2).min(num_samples_recorded - 1)
                    };

                    loop {
                        // Sample from back in time
                        let read_index =
                            (index_of_newest + SAMPLES_IN_BUFFER - i) % SAMPLES_IN_BUFFER;
                        let y = &state_history.column(read_index);
                        let relative_position = Vector2::new(
                            y[point_pos_loc] - listener_pos[0],
                            y[point_pos_loc + 1] - listener_pos[1],
                        );
                        let distance_by_time = i as f64 * meters_per_sample;
                        let distance_by_time_squared = distance_by_time * distance_by_time;
                        let distance_by_state_squared = relative_position.norm_squared();

                        // Do we need to go further back in time?
                        if distance_by_time_squared < distance_by_state_squared {
                            i += 1;
                            if i >= num_samples_recorded {
                                break;
                            }
                            continue;
                        }

                        // Sample from slightly less far back in time
                        let read_index_next = (read_index + 1) % SAMPLES_IN_BUFFER;
                        let y_next = &state_history.column(read_index_next);
                        let relative_position_next = Vector2::new(
                            y_next[point_pos_loc] - listener_pos[0],
                            y_next[point_pos_loc + 1] - listener_pos[1],
                        );
                        let distance_by_time_next = (i - 1) as f64 * meters_per_sample;
                        let distance_by_time_next_squared =
                            distance_by_time_next * distance_by_time_next;
                        let distance_by_state_next_squared = relative_position_next.norm_squared();

                        // Do we need to go forwards in time?
                        if distance_by_time_next_squared > distance_by_state_next_squared {
                            i -= 1;
                            if i < 2 {
                                break;
                            }
                            continue;
                        }

                        // We should now have a sample before and after the information horizon.
                        // Interpolate between these to find the value at the horizon.
                        let distance_by_state = distance_by_state_squared.sqrt();
                        let distance_by_state_prev = distance_by_state_next_squared.sqrt();
                        let t = (distance_by_time - distance_by_state)
                            / (distance_by_state_prev - distance_by_state + meters_per_sample);

                        let acc_next =
                            &acc_history.fixed_slice::<2, 1>(point_pos_loc, read_index_next);
                        let acc = &acc_history.fixed_slice::<2, 1>(point_pos_loc, read_index);
                        let interpolated_relative_position =
                            relative_position + t * (relative_position_next - relative_position);
                        let interpolated_acc = acc + t * (acc_next - acc);
                        let direction = interpolated_relative_position.normalize();
                        value += interpolated_acc.dot(&direction)
                            / interpolated_relative_position.norm_squared();
                        break;
                    }
                }
                value as f32
            };

            player.set_listener_pos(self.listener_pos).unwrap();
            player.play_audio(Box::new(next_sample)).unwrap();
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
