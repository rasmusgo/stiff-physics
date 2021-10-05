use std::convert::TryInto;

use eframe::{
    egui::{self, vec2, Color32, Sense, Stroke, Vec2},
    epi,
};
use hyperdual::{Float, Hyperdual};
use nalgebra::{self, DMatrix, DVector};

use crate::play_audio_buffer;

trait OneHot {
    type H;
    fn from_one_hot(index: usize) -> Self::H;
}

impl<T: hyperdual::Zero + hyperdual::One + Copy + nalgebra::Scalar, const N: usize> OneHot
    for Hyperdual<T, N>
{
    type H = Hyperdual<T, N>;
    /// Create a new dual number from a real number and set the real value or a derivative to one.
    ///
    /// All other parts are set to zero.
    #[inline]
    fn from_one_hot(index: usize) -> Hyperdual<T, N>
    where
        T: hyperdual::Zero,
    {
        let mut dual = Hyperdual::<T, N>::from_real(T::zero());
        dual[index] = T::one();
        dual
    }
}

const NUM_POINTS: usize = 3;
const NUM_SPRINGS: usize = NUM_POINTS - 1;
const D: usize = 2;
const STRIDE: usize = D * 2;
const N: usize = NUM_POINTS * STRIDE + 1;

fn spring_force<const S: usize>(
    p1_px: Hyperdual<f64, S>,
    p1_py: Hyperdual<f64, S>,
    p1_vx: Hyperdual<f64, S>,
    p1_vy: Hyperdual<f64, S>,
    p2_px: Hyperdual<f64, S>,
    p2_py: Hyperdual<f64, S>,
    p2_vx: Hyperdual<f64, S>,
    p2_vy: Hyperdual<f64, S>,
    relaxed_length: f64,
    k: f64, // Spring constant
    d: f64,
) -> [Hyperdual<f64, S>; D] {
    let dx = p2_px - p1_px;
    let dy = p2_py - p1_py;
    let dvx = p2_vx - p1_vx;
    let dvy = p2_vy - p1_vy;
    let spring_length = (dx * dx + dy * dy).sqrt();
    let spring_dirx = dx / spring_length;
    let spring_diry = dy / spring_length;
    let force_magnitude: Hyperdual<f64, S> = Hyperdual::from_real(k)
        * (spring_length - Hyperdual::from_real(relaxed_length))
        + Hyperdual::from_real(d) * (spring_dirx * dvx + spring_diry * dvy);

    let force_x = spring_dirx * -force_magnitude;
    let force_y = spring_diry * -force_magnitude;

    return [force_x, force_y];
}

fn create_diff_eq_system(
    points: &[Vec2; NUM_POINTS],
    point_masses: &[f32; NUM_POINTS],
    lengths: &[f32; NUM_SPRINGS],
    k: f32,
    d: f32,
) -> (DMatrix<f64>, DVector<f64>) {
    // Initial state
    let y0 = DVector::<f64>::from_vec(vec![
        points[0].x.into(),
        points[0].y.into(),
        0.0, // vx
        0.0, // vy
        points[1].x.into(),
        points[1].y.into(),
        0.0, // vx
        0.0, // vy
        points[2].x.into(),
        points[2].y.into(),
        0.0, // vx
        0.0, // vy
        1.0, // For transformation to homogenous system.
    ]);

    // Dual numbers for automatic differentiation of springs. (Spatial derivatives, not time derivatives).
    let mut p1_px: Hyperdual<f64, 9> = Hyperdual::from_one_hot(1);
    let mut p1_py: Hyperdual<f64, 9> = Hyperdual::from_one_hot(2);
    let mut p1_vx: Hyperdual<f64, 9> = Hyperdual::from_one_hot(3);
    let mut p1_vy: Hyperdual<f64, 9> = Hyperdual::from_one_hot(4);
    let mut p2_px: Hyperdual<f64, 9> = Hyperdual::from_one_hot(5);
    let mut p2_py: Hyperdual<f64, 9> = Hyperdual::from_one_hot(6);
    let mut p2_vx: Hyperdual<f64, 9> = Hyperdual::from_one_hot(7);
    let mut p2_vy: Hyperdual<f64, 9> = Hyperdual::from_one_hot(8);

    // Construct A matrix for y' = Ay. (Time derivative of state vector).
    let mut mat_a = DMatrix::<f64>::zeros(N, N);

    // Equations for variable substitutions
    for i in 0..NUM_POINTS {
        let start_index = i * STRIDE;
        for j in 0..D {
            // "velocity is velocity"
            mat_a[(start_index + j, start_index + D + j)] = 1.0;
        }
    }

    // Equations for spring forces
    for (i, &relaxed_length) in lengths.iter().enumerate() {
        let p1_loc = i * STRIDE;
        let p2_loc = (i + 1) * STRIDE;

        // Set parameters to spring function.
        p1_px[0] = y0[p1_loc];
        p1_py[0] = y0[p1_loc + 1];
        p1_vx[0] = y0[p1_loc + 2];
        p1_vy[0] = y0[p1_loc + 3];
        p2_px[0] = y0[p2_loc];
        p2_py[0] = y0[p2_loc + 1];
        p2_vx[0] = y0[p2_loc + 2];
        p2_vy[0] = y0[p2_loc + 3];

        let p1_mass: f64 = point_masses[i].into();
        let p2_mass: f64 = point_masses[i + 1].into();

        let force = spring_force(
            p1_px,
            p1_py,
            p1_vx,
            p1_vy,
            p2_px,
            p2_py,
            p2_vx,
            p2_vy,
            relaxed_length.into(),
            k.into(),
            d.into(),
        );

        for j in 0..D {
            let mut constant_term = force[j][0];
            for k in 0..STRIDE {
                // Acceleration based on position and velocity.
                mat_a[(p1_loc + D + j, p1_loc + k)] -= force[j][1 + k] / p1_mass; // p1 acc from pos and vel of p1.
                mat_a[(p1_loc + D + j, p2_loc + k)] -= force[j][1 + STRIDE + k] / p1_mass; // p1 acc from pos and vel of p2.
                mat_a[(p2_loc + D + j, p1_loc + k)] += force[j][1 + k] / p2_mass; // p2 acc from pos and vel of p1.
                mat_a[(p2_loc + D + j, p2_loc + k)] += force[j][1 + STRIDE + k] / p2_mass; // p2 acc from pos and vel of p2.

                // Offset for linearization around y0.
                constant_term -=
                    force[j][1 + k] * y0[p1_loc + k] + force[j][1 + STRIDE + k] * y0[p2_loc + k];
            }
            // Constant acceleration term.
            mat_a[(p1_loc + D + j, N - 1)] -= constant_term / p1_mass;
            mat_a[(p2_loc + D + j, N - 1)] += constant_term / p2_mass;
        }
    }

    return (mat_a, y0);
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))] // if we add new fields, give them default values when deserializing old state
pub struct StiffPhysicsApp {
    // Example stuff:
    label: String,

    // this how you opt-out of serialization of a member
    #[cfg_attr(feature = "persistence", serde(skip))]
    spring_constant: f32,
    damping: f32,
    point_mass: f32,
    relaxation_iterations: usize,
    points: [Vec2; NUM_POINTS],
    lengths: [f32; NUM_SPRINGS],
    relaxed_points: [Vec2; NUM_POINTS],
    mat_a: DMatrix<f64>,
    simulation_state: DVector<f64>,
    exp_a_sim_step: DMatrix<f64>,
    exp_a_audio_step: DMatrix<f64>,
    enable_simulation: bool,
}

impl Default for StiffPhysicsApp {
    fn default() -> Self {
        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            spring_constant: 1000.0,
            damping: 0.1,
            point_mass: 0.001,
            relaxation_iterations: 1,
            points: [vec2(-0.5, 0.), vec2(0., 0.5), vec2(0.5, 0.)],
            lengths: [0.5, 0.5],
            relaxed_points: [vec2(-0.5, 0.), vec2(0., 0.5), vec2(0.5, 0.)],
            mat_a: DMatrix::<f64>::zeros(N, N),
            simulation_state: DVector::<f64>::zeros(N),
            exp_a_sim_step: DMatrix::<f64>::zeros(N, N),
            exp_a_audio_step: DMatrix::<f64>::zeros(N, N),
            enable_simulation: false,
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
            label,
            spring_constant,
            damping,
            point_mass,
            relaxation_iterations,
            points,
            lengths,
            relaxed_points,
            mat_a,
            simulation_state,
            exp_a_sim_step,
            exp_a_audio_step,
            enable_simulation,
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

            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(label);
            });

            ui.add(egui::Slider::new(spring_constant, 0.0..=1000.0).text("spring_constant"));
            ui.add(egui::Slider::new(damping, 0.0..=1000.0).text("damping"));
            ui.add(egui::Slider::new(point_mass, 0.01..=10.0).text("point_mass"));

            if ui.button("Store as relaxed").clicked() {
                for (i, length) in lengths.into_iter().enumerate() {
                    let p1 = points[i];
                    let p2 = points[i + 1];
                    let diff = p1 - p2;
                    let norm2 = diff.x * diff.x + diff.y * diff.y;
                    *length = norm2.sqrt();
                }
            }

            ui.add(
                egui::Slider::new(relaxation_iterations, 0..=10)
                    .text("Relaxation iterations before linearization"),
            );

            let point_masses: [f32; 3] = [*point_mass, *point_mass, *point_mass];
            let (mut a, y0) =
                create_diff_eq_system(points, &point_masses, lengths, *spring_constant, *damping);

            // Re-linearize around estimated relaxed state
            for _i in 0..*relaxation_iterations {
                // Solve for zero velocity, zero acceleration and minimal distance to y0 (TODO: weighted by mass)
                // https://math.stackexchange.com/questions/1318637/projection-of-a-vector-onto-the-null-space-of-a-matrix
                let y_relaxed = &y0
                    - &a.tr_mul(&((&a * &a.transpose()).pseudo_inverse(1.0e-10).unwrap() * &a))
                        * &y0;

                *relaxed_points = [
                    Vec2::new(
                        y_relaxed[0 * STRIDE] as f32,
                        y_relaxed[0 * STRIDE + 1] as f32,
                    ),
                    Vec2::new(
                        y_relaxed[1 * STRIDE] as f32,
                        y_relaxed[1 * STRIDE + 1] as f32,
                    ),
                    Vec2::new(
                        y_relaxed[2 * STRIDE] as f32,
                        y_relaxed[2 * STRIDE + 1] as f32,
                    ),
                ];
                a = create_diff_eq_system(
                    &*relaxed_points,
                    &point_masses,
                    lengths,
                    *spring_constant,
                    *damping,
                )
                .0;
            }
            *mat_a = a;

            if ui.button("Simulate").clicked() {
                *simulation_state = y0;
                *exp_a_sim_step = (mat_a.clone() * 0.01).exp();
                *exp_a_audio_step = (mat_a.clone() / 44100.0).exp();
                *enable_simulation = true;
                // println!("{:?}", mat_a);
                // println!("{:?}", &*mat_a * &*simulation_state);

                // Generate audio
                let mut data = Vec::new();
                let mut y = simulation_state.clone();
                let mut max_value: f32 = 0.0;
                for _i in 0..44100 * 3 {
                    let y_next = &*exp_a_audio_step * &y;
                    let value = (&y_next[D] - &y[D]) as f32;
                    max_value = f32::max(max_value, value.abs());
                    data.push(value);
                    y = y_next;
                }
                // Normalize volume
                if max_value > 0.0 {
                    for value in data.iter_mut() {
                        *value /= max_value;
                    }
                }
                // Fade in to remove click at the start
                const FADE_IN_SAMPLES: usize = 44100 / 50;
                for (i, value) in data.iter_mut().enumerate() {
                    if i == FADE_IN_SAMPLES {
                        break;
                    }
                    *value *= i as f32 / FADE_IN_SAMPLES as f32;
                }
                // Fade out to remove click at the end
                const FADE_OUT_SAMPLES: usize = 44100 / 50;
                for (i, value) in data.iter_mut().rev().enumerate() {
                    if i == FADE_OUT_SAMPLES {
                        break;
                    }
                    *value *= i as f32 / FADE_OUT_SAMPLES as f32;
                }

                // Send data to audio player (javascript when running via wasm)
                play_audio_buffer(data);
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
            let circle_radius = r / 100.;
            let color = Color32::BLACK;

            if let Some(pos) = response.interact_pointer_pos() {
                let mut best_norm2 = 15. * 15.;
                let mut best_point = None;
                for (i, &mut p) in points.into_iter().enumerate() {
                    let point_in_pixels = c + p * r;
                    let diff = pos - point_in_pixels;
                    let norm2 = diff.x * diff.x + diff.y * diff.y;
                    if norm2 <= best_norm2 {
                        best_norm2 = norm2;
                        best_point = Some(i);
                    }
                }
                if let Some(i) = best_point {
                    points[i] = (pos - c) / r;
                }
            }
            if *enable_simulation {
                let simulated_points: [Vec2; NUM_POINTS] = (0..NUM_POINTS)
                    .map(|i| {
                        vec2(
                            simulation_state[i * STRIDE] as f32,
                            simulation_state[i * STRIDE + 1] as f32,
                        )
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                draw_particle_system(
                    lengths,
                    &simulated_points,
                    line_width,
                    &painter,
                    c,
                    r,
                    circle_radius,
                    color,
                );
            }
            if *relaxation_iterations > 0 {
                draw_particle_system(
                    lengths,
                    relaxed_points,
                    line_width,
                    &painter,
                    c,
                    r,
                    circle_radius,
                    color,
                );
            }
            draw_particle_system(
                lengths,
                points,
                line_width,
                &painter,
                c,
                r,
                circle_radius,
                color,
            );

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
    lengths: &[f32],
    points: &[Vec2],
    line_width: f32,
    painter: &egui::Painter,
    c: egui::Pos2,
    r: f32,
    circle_radius: f32,
    color: Color32,
) {
    for (i, &length) in lengths.into_iter().enumerate() {
        let p1 = points[i];
        let p2 = points[i + 1];
        let diff: Vec2 = p1 - p2;
        let norm2 = diff.x * diff.x + diff.y * diff.y;
        let stress = f32::min((length - norm2.sqrt()).abs() / length, 1.0);
        let line_color = egui::lerp(egui::Rgba::GREEN..=egui::Rgba::RED, stress);
        let stroke = Stroke::new(line_width, line_color);
        painter.line_segment([c + p1 * r, c + p2 * r], stroke);
    }
    for &p in points {
        painter.circle_filled(c + p * r, circle_radius, color);
    }
}
