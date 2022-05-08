# Stiff Physics

[![dependency status](https://deps.rs/repo/github/rasmusgo/stiff-physics/status.svg)](https://deps.rs/repo/github/rasmusgo/stiff-physics)
[![Build Status](https://github.com/rasmusgo/stiff-physics/workflows/CI/badge.svg)](https://github.com/rasmusgo/stiff-physics/actions?workflow=CI)

This is a proof of concept of physically based audio generation from simulation of stiff springs and light point masses.

### Testing online

Try it out by visiting <https://rasmusgo.github.io/stiff-physics/>.

### Testing locally

On Linux you need to first run:

`sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libspeechd-dev libxkbcommon-dev libssl-dev libasound2-dev`

On Fedora Rawhide you need to run:

`dnf install clang clang-devel clang-tools-extra speech-dispatcher-devel libxkbcommon-devel pkg-config openssl-devel alsa-lib-devel`

Make sure you are using the latest version of stable rust by running

`rustup update`

Then you should be able to compile and run it with

`cargo run --release`

### Compiling for the web

The app can be compiled to [WASM](https://en.wikipedia.org/wiki/WebAssembly) and published as a web page. There are a few simple scripts that manages this:

``` sh
./setup_web.sh
./build_web.sh
./start_server.sh
open http://127.0.0.1:8080/
```

* `setup_web.sh` installs the tools required to build for web
* `build_web.sh` compiles the code to wasm and puts it in the `docs/` folder (see below)
* `start_server.sh` starts a local HTTP server so you can test before you publish
* Open http://127.0.0.1:8080/ in a web browser to view

The finished web app is found in the `docs/` folder (this is so that it can be easily shared with [GitHub Pages](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)). It consists of three files:

* `index.html`: A few lines of HTML, CSS and JS that loads the app.
* `stiff_physics_bg.wasm`: What the Rust code compiles to.
* `stiff_physics.js`: Auto-generated binding between Rust and JS.

You can test the template app at <https://rasmusgo.github.io/stiff-physics/>.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
