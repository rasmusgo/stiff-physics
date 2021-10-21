use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use anyhow::anyhow;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

type SamplingFunction = Box<dyn Send + FnMut() -> f32>;

pub struct AudioPlayer {
    device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    stream: Option<anyhow::Result<cpal::Stream>>,
    producer: Option<rtrb::Producer<SamplingFunction>>,
    pub to_ui_consumer: Option<rtrb::Consumer<(f32, f32)>>,
    pub enable_band_pass_filter: Arc<AtomicBool>,
}

impl AudioPlayer {
    pub fn new() -> anyhow::Result<AudioPlayer> {
        let host = cpal::default_host();

        let optional_device = host.default_output_device();
        if optional_device.is_none() {
            anyhow::bail!("No output device is available");
        }
        let device = optional_device.unwrap();
        println!("Output device: {}", device.name()?);

        let config = device.default_output_config()?;
        println!("Default output config: {:?}", config);

        let mut audio_player = AudioPlayer {
            device,
            config,
            stream: None,
            producer: None,
            to_ui_consumer: None,
            enable_band_pass_filter: Arc::new(AtomicBool::new(true)),
        };
        audio_player.start_output_stream()?;
        Ok(audio_player)
    }

    fn start_output_stream(&mut self) -> anyhow::Result<()> {
        let (producer, mut consumer) = rtrb::RingBuffer::new(2);
        let (mut to_ui_producer, to_ui_consumer) =
            rtrb::RingBuffer::new(self.config.sample_rate().0 as usize);
        let mut stored_sampling_function: Option<SamplingFunction> = None;
        // Exponential moving average band-pass filtering
        const ALPHA1: f32 = 0.01;
        const ALPHA2: f32 = 0.001;
        let mut moving_average1 = 0_f32;
        let mut moving_average2 = 0_f32;
        let enable_band_pass_filter = self.enable_band_pass_filter.clone();
        let next_sample = move || {
            if let Ok(new_sampling_function) = consumer.pop() {
                stored_sampling_function = Some(new_sampling_function);
            }

            let value = if let Some(sampling_function) = &mut stored_sampling_function {
                let sample = sampling_function();
                if sample.is_finite() {
                    sample
                } else {
                    0_f32
                }
            } else {
                0_f32
            };
            moving_average1 = moving_average1 * (1.0 - ALPHA1) + value * ALPHA1;
            moving_average2 = moving_average2 * (1.0 - ALPHA2) + value * ALPHA2;
            let filtered_value = moving_average1 - moving_average2;

            // Try to push but ignore if it works or not.
            let _ = to_ui_producer.push((value, filtered_value));

            if enable_band_pass_filter.load(Ordering::SeqCst) {
                filtered_value
            } else {
                value
            }
        };
        self.producer = Some(producer);
        self.to_ui_consumer = Some(to_ui_consumer);
        self.stream = Some(match self.config.sample_format() {
            cpal::SampleFormat::F32 => run::<f32>(
                &self.device,
                &self.config.clone().into(),
                Box::new(next_sample),
            ),
            cpal::SampleFormat::I16 => run::<i16>(
                &self.device,
                &self.config.clone().into(),
                Box::new(next_sample),
            ),
            cpal::SampleFormat::U16 => run::<u16>(
                &self.device,
                &self.config.clone().into(),
                Box::new(next_sample),
            ),
        });
        Ok(())
    }

    pub fn play_audio(&mut self, next_sample: SamplingFunction) -> anyhow::Result<()> {
        match &mut self.producer {
            Some(producer) => producer
                .push(next_sample)
                .or(Err(anyhow!("Failed to push"))),
            None => Err(anyhow!("Audio not initialized")),
        }
    }
}

pub fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut next_sample: SamplingFunction,
) -> anyhow::Result<cpal::Stream>
where
    T: cpal::Sample,
{
    let channels = config.channels as usize;
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_sample)
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
