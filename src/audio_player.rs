use anyhow;
use cpal;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> anyhow::Result<cpal::Stream>
where
    T: cpal::Sample,
{
    let sample_rate = config.sample_rate.0 as f32;
    let channels = config.channels as usize;

    // Produce a sinusoid of maximum amplitude.
    let mut sample_clock = 0f32;
    let mut next_value = move || {
        sample_clock = (sample_clock + 1.0) % sample_rate;
        (sample_clock * 440.0 * std::f32::consts::TAU / sample_rate).sin()
    };

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value)
        },
        err_fn,
    )?;
    stream.play()?;
    Ok(stream)
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: cpal::Sample,
{
    for frame in output.chunks_mut(channels) {
        let value: T = cpal::Sample::from::<f32>(&next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

pub struct AudioPlayer {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    stream: Option<anyhow::Result<cpal::Stream>>,
}

impl AudioPlayer {
    pub fn new() -> anyhow::Result<AudioPlayer> {
        let host = cpal::default_host();

        let optional_device = host.default_output_device();
        if let None = optional_device {
            anyhow::bail!("No output device is available");
        }
        let device = optional_device.unwrap();
        println!("Output device: {}", device.name()?);

        let config = device.default_output_config()?;
        println!("Default output config: {:?}", config);

        Ok(AudioPlayer {
            device,
            config,
            stream: None,
        })
    }

    pub fn play_audio_buffer(&mut self, _data: Vec<f32>) -> anyhow::Result<()> {
        self.stream = Some(match self.config.sample_format() {
            cpal::SampleFormat::F32 => run::<f32>(&self.device, &self.config.clone().into()),
            cpal::SampleFormat::I16 => run::<i16>(&self.device, &self.config.clone().into()),
            cpal::SampleFormat::U16 => run::<u16>(&self.device, &self.config.clone().into()),
        });
        Ok(())
    }
}
