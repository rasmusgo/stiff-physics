use eframe::{egui::{self, Color32, Sense, Stroke, Vec2, vec2}, epi};

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
    points: [Vec2; 3],
    lengths: [f32; 2],
}

impl Default for StiffPhysicsApp {
    fn default() -> Self {
        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            spring_constant: 1000.0,
            damping: 1.0,
            point_mass: 0.01,
            points: [
                vec2(-0.5, 0.),
                vec2(0., 0.),
                vec2(0.5, 0.),
            ],
            lengths: [
                0.5,
                0.5,
            ]
        }
    }
}

impl epi::App for StiffPhysicsApp {
    fn name(&self) -> &str {
        "egui template"
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
        let Self { label, spring_constant, damping, point_mass, points, lengths } = self;

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
                let mut best_norm2 = 15.*15.;
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
            for (i, &mut length) in lengths.into_iter().enumerate() {
                let p1 = points[i];
                let p2 = points[i + 1];
                let diff = p1 - p2;
                let norm2 = diff.x * diff.x + diff.y * diff.y;
                let stress = f32::min((length - norm2.sqrt()).abs() / length, 1.0);
                let line_color = egui::lerp(egui::Rgba::GREEN..=egui::Rgba::RED, stress);
                let stroke = Stroke::new(line_width, line_color);
                painter.line_segment([
                    c + p1 * r,
                    c + p2 * r,
                    ], stroke);
            }
            for &mut p in points {
                painter.circle_filled(c + p * r, circle_radius, color);
            }

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
