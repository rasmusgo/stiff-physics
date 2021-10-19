use anyhow::anyhow;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

type SamplingFunction = Box<dyn Send + FnMut() -> f32>;

pub struct AudioPlayer {
    device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    stream: Option<anyhow::Result<cpal::Stream>>,
    producer: Option<rtrb::Producer<SamplingFunction>>,
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

        Ok(AudioPlayer {
            device,
            config,
            stream: None,
            producer: None,
        })
    }

    pub fn start_output_stream(&mut self) -> anyhow::Result<()> {
        let (producer, mut consumer) = rtrb::RingBuffer::new(2);
        let mut stored_sampling_function: Option<SamplingFunction> = None;
        let next_sample = move || {
            if let Ok(new_sampling_function) = consumer.pop() {
                stored_sampling_function = Some(new_sampling_function);
            }

            if let Some(sampling_function) = &mut stored_sampling_function {
                return sampling_function();
            }
            0_f32
        };
        self.producer = Some(producer);
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
