use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut next_sample: Box<dyn Send + FnMut() -> f32>,
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

pub struct AudioPlayer {
    device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    stream: Option<anyhow::Result<cpal::Stream>>,
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
        })
    }

    pub fn play_audio_buffer(
        &mut self,
        next_sample: Box<dyn Send + FnMut() -> f32>,
    ) -> anyhow::Result<()> {
        self.stream = Some(match self.config.sample_format() {
            cpal::SampleFormat::F32 => {
                run::<f32>(&self.device, &self.config.clone().into(), next_sample)
            }
            cpal::SampleFormat::I16 => {
                run::<i16>(&self.device, &self.config.clone().into(), next_sample)
            }
            cpal::SampleFormat::U16 => {
                run::<u16>(&self.device, &self.config.clone().into(), next_sample)
            }
        });
        Ok(())
    }
}
