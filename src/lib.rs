#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), deny(warnings))] // Forbid warnings in release builds
#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod audio_player;

pub use app::StiffPhysicsApp;

// ----------------------------------------------------------------------------
// When compiling for web:

#[cfg(target_arch = "wasm32")]
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use eframe::{
    egui::mutex::Mutex,
    wasm_bindgen::{self, prelude::*, JsCast, JsValue},
};

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), JsValue> {
    let app = StiffPhysicsApp::default();
    install_touch_handler(canvas_id, app.audio_player.clone())?;
    eframe::start_web(canvas_id, Box::new(app))
}

#[cfg(target_arch = "wasm32")]
fn install_touch_handler(
    canvas_id: &str,
    audio_player: Arc<Mutex<Option<anyhow::Result<audio_player::AudioPlayer>>>>,
) -> Result<(), JsValue> {
    let document = web_sys::window()
        .ok_or("Failed to get window")?
        .document()
        .ok_or("Failed to get document")?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Failed to get canvas")?;

    {
        let event_name = "touchstart";
        let closure = Closure::wrap(Box::new(move |_event: web_sys::TouchEvent| {
            let mut audio_player_ref = audio_player.lock();
            if audio_player_ref.is_none() {
                *audio_player_ref = Some(audio_player::AudioPlayer::new());
            }
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback(event_name, closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    Ok(())
}
