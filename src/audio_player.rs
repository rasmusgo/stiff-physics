use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use anyhow::anyhow;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

type SamplingFunction = Box<dyn Send + FnMut() -> f32>;

pub struct AudioPlayer {
    device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    stream: Option<anyhow::Result<cpal::Stream>>,
    pub enable_band_pass_filter: Arc<AtomicBool>,
    pub num_frames_per_callback: Arc<AtomicUsize>,
    ring_buffers: Option<RingBuffersNonRealTimeSides>,
}

struct RingBuffersNonRealTimeSides {
    disposal_queue_consumer: rtrb::Consumer<SamplingFunction>,
    sampling_function_producer: rtrb::Producer<SamplingFunction>,
    to_ui_consumer: rtrb::Consumer<(f32, f32, f32, f32)>,
}

struct RingBuffersRealTimeSides {
    disposal_queue_producer: rtrb::Producer<SamplingFunction>,
    sampling_function_consumer: rtrb::Consumer<SamplingFunction>,
    to_ui_producer: rtrb::Producer<(f32, f32, f32, f32)>,
}

impl AudioPlayer {
    pub fn new() -> anyhow::Result<AudioPlayer> {
        puffin::profile_function!();
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
            ring_buffers: None,
            enable_band_pass_filter: Arc::new(AtomicBool::new(true)),
            num_frames_per_callback: Arc::new(AtomicUsize::new(0)),
        };
        audio_player.start_output_stream()?;
        Ok(audio_player)
    }

    fn start_output_stream(&mut self) -> anyhow::Result<()> {
        puffin::profile_function!();
        let (disposal_queue_producer, disposal_queue_consumer) = rtrb::RingBuffer::new(2);
        let (sampling_function_producer, sampling_function_consumer) = rtrb::RingBuffer::new(2);
        let (to_ui_producer, to_ui_consumer) =
            rtrb::RingBuffer::new(self.config.sample_rate().0 as usize);
        self.ring_buffers = Some(RingBuffersNonRealTimeSides {
            disposal_queue_consumer,
            sampling_function_producer,
            to_ui_consumer,
        });
        let realtime_sides = RingBuffersRealTimeSides {
            disposal_queue_producer,
            sampling_function_consumer,
            to_ui_producer,
        };
        self.stream = Some(match self.config.sample_format() {
            cpal::SampleFormat::F32 => run::<f32>(
                &self.device,
                &self.config.clone().into(),
                self.enable_band_pass_filter.clone(),
                self.num_frames_per_callback.clone(),
                realtime_sides,
            ),
            cpal::SampleFormat::I16 => run::<i16>(
                &self.device,
                &self.config.clone().into(),
                self.enable_band_pass_filter.clone(),
                self.num_frames_per_callback.clone(),
                realtime_sides,
            ),
            cpal::SampleFormat::U16 => run::<u16>(
                &self.device,
                &self.config.clone().into(),
                self.enable_band_pass_filter.clone(),
                self.num_frames_per_callback.clone(),
                realtime_sides,
            ),
        });
        Ok(())
    }

    pub fn play_audio(&mut self, next_sample: SamplingFunction) -> anyhow::Result<()> {
        puffin::profile_function!();
        if let Some(RingBuffersNonRealTimeSides {
            disposal_queue_consumer,
            ..
        }) = &mut self.ring_buffers
        {
            // Old sampling functions are implicitly dropped here.
            while disposal_queue_consumer.pop().is_ok() {}
        }
        match &mut self.ring_buffers {
            Some(RingBuffersNonRealTimeSides {
                sampling_function_producer,
                ..
            }) => sampling_function_producer
                .push(next_sample)
                .map_err(|_| anyhow!("Failed to push")),
            None => Err(anyhow!("Audio not initialized")),
        }
    }

    pub fn get_audio_history_entry(&mut self) -> Option<(f32, f32, f32, f32)> {
        if let Some(RingBuffersNonRealTimeSides { to_ui_consumer, .. }) = &mut self.ring_buffers {
            to_ui_consumer.pop().ok()
        } else {
            None
        }
    }
}

fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    enable_band_pass_filter: Arc<AtomicBool>,
    num_frames_per_callback: Arc<AtomicUsize>,
    RingBuffersRealTimeSides {
        mut sampling_function_consumer,
        mut disposal_queue_producer,
        mut to_ui_producer,
    }: RingBuffersRealTimeSides,
) -> anyhow::Result<cpal::Stream>
where
    T: cpal::Sample,
{
    puffin::profile_function!();
    let channels = config.channels as usize;
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let mut stored_sampling_function: Option<SamplingFunction> = None;

    // Exponential moving average band-pass filtering
    const ALPHA1: f32 = 0.01;
    const ALPHA2: f32 = 0.001;
    const ALPHA_ATTACK: f32 = 0.01;
    const ALPHA_RELEASE: f32 = 0.0001;
    const BASELINE: f32 = 0.1;
    const HEADROOM_FRACTION: f32 = 0.25;
    const HEADROOM_FACTOR: f32 = 1.0 - HEADROOM_FRACTION;
    let mut moving_average1 = 0_f32;
    let mut moving_average2 = 0_f32;
    let mut moving_power_average_filtered = BASELINE;
    let mut moving_power_average_raw = BASELINE;

    puffin::profile_function!();
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            puffin::profile_function!();
            if let Ok(new_sampling_function) = sampling_function_consumer.pop() {
                if let Some(old_sampling_function) =
                    std::mem::replace(&mut stored_sampling_function, Some(new_sampling_function))
                {
                    disposal_queue_producer.push(old_sampling_function).unwrap();
                }
            }

            // Report num_frames_per_callback
            num_frames_per_callback.store(data.len() / channels, Ordering::Relaxed);

            let mut next_sample = || {
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
                let filtered_tall_puppy = filtered_value.abs() + BASELINE;
                if filtered_tall_puppy > moving_power_average_filtered {
                    moving_power_average_filtered = moving_power_average_filtered
                        * (1.0 - ALPHA_ATTACK)
                        + filtered_tall_puppy * ALPHA_ATTACK;
                } else {
                    moving_power_average_filtered = moving_power_average_filtered
                        * (1.0 - ALPHA_RELEASE)
                        + filtered_tall_puppy * ALPHA_RELEASE;
                }
                let raw_tall_puppy = value.abs() + BASELINE;
                if raw_tall_puppy > moving_power_average_raw {
                    moving_power_average_raw = moving_power_average_raw * (1.0 - ALPHA_ATTACK)
                        + raw_tall_puppy * ALPHA_ATTACK;
                } else {
                    moving_power_average_raw = moving_power_average_raw * (1.0 - ALPHA_RELEASE)
                        + raw_tall_puppy * ALPHA_RELEASE;
                }

                let normalized_filtered_value =
                    HEADROOM_FACTOR * filtered_value / moving_power_average_filtered;
                let normalized_value = HEADROOM_FACTOR * value / moving_power_average_raw;

                // Try to push but ignore if it works or not.
                let _ = to_ui_producer.push((
                    normalized_value,
                    normalized_filtered_value,
                    moving_power_average_raw,
                    moving_power_average_filtered,
                ));

                if enable_band_pass_filter.load(Ordering::Relaxed) {
                    normalized_filtered_value
                } else {
                    normalized_value
                }
            };

            for frame in data.chunks_mut(channels) {
                let value: T = cpal::Sample::from::<f32>(&next_sample());
                for sample in frame.iter_mut() {
                    *sample = value;
                }
            }
        },
        err_fn,
    )?;
    stream.play()?;
    Ok(stream)
}
