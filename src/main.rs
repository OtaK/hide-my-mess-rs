use crate::error::HideError;

mod error;
pub use self::error::*;

mod rvm;

#[derive(Debug, clap::Parser)]
#[clap(about, version, author)]
/// Virtual camera that blurs your background
struct Args {
    /// Lists video capture devices present on the system along with their details
    #[clap(short, long = "list")]
    list_devices: bool,
    /// Choose a video capture device index manually, in case you have several connected
    #[clap(long)]
    camera_index: Option<usize>,
    /// Desired frame capture width. If not available, it'll fallback to the highest mode detected
    #[clap(short, long)]
    width: Option<u32>,
    /// Desired frame capture height. If not available, it'll fallback to the highest mode detected
    #[clap(short, long)]
    height: Option<u32>,
    /// Desired capture framerate. If not available, it'll fallback to the highest mode detected
    #[clap(long)]
    fps: Option<u32>,
    /// Used variant of RobustVideoMatting. resnet is more accurate & faster but heavier, mobilenet is lighter
    /// The two possible choices are `resnet50` and `mobilenetv3`
    #[clap(long, default_value_t)]
    model: rvm::RvmModelKind,
}

#[cfg(not(target_os = "linux"))]
fn main() -> error::HideResult<()> {
    panic!("Not compatible!");

    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|granted| {
        if !granted {
            log::info!("Permission denied, exiting...");
            std::process::exit(0);
        }

        if let Err(e) = main_next() {
            log::error!("{}", e);
        }
    });

    #[cfg(not(target_os = "macos"))]
    main_next()
}

#[cfg(target_os = "linux")]
fn main() -> error::HideResult<()> {
    pretty_env_logger::init();
    main_next()
}

#[inline(always)]
fn main_next() -> error::HideResult<()> {
    use clap::StructOpt as _;
    let args = Args::parse();

    let mut devices = nokhwa::query_devices(nokhwa::CaptureAPIBackend::Auto)?;
    devices.sort_by(|a, b| {
        a.index()
            .as_index()
            .unwrap()
            .cmp(&b.index().as_index().unwrap())
    });

    if args.list_devices {
        if devices.is_empty() {
            log::info!("No devices found on the system!");
        } else {
            log::info!(
                "Found {} device(s) currently connected to this system: ",
                devices.len()
            );
            for d in devices {
                log::info!(
                    "Index #{}: {} via [{}]@[{}]",
                    d.index(),
                    d.human_name(),
                    d.description(),
                    d.misc(),
                );
            }
        }

        return Ok(());
    }

    log::debug!("devices found: {devices:#?}");
    if devices.is_empty() {
        log::error!("No devices found on the system!");
        return Ok(());
    }

    let camera_index = args
        .camera_index
        .map(|idx| nokhwa::CameraIndex::Index(idx as u32))
        .unwrap_or_else(|| devices[0].index().clone());

    log::debug!("Selected device index: #{camera_index}");

    let mut camera = nokhwa::Camera::new(&camera_index, None)?;

    let mut format = camera.camera_format();

    log::info!(
        "Selected camera: #{} - {}",
        camera.info().index(),
        camera.info().human_name()
    );

    let mut compatible_formats = camera
        .compatible_list_by_resolution(nokhwa::FrameFormat::MJPEG)?
        .into_iter()
        .filter(|(_res, fps_list)| fps_list.iter().any(|f| *f >= 24))
        .collect::<Vec<(nokhwa::Resolution, Vec<u32>)>>();

    compatible_formats.sort_by(|a, b| b.0.cmp(&a.0));

    if compatible_formats.is_empty() {
        log::error!("Your capture device somehow supports NO capture formats! :(");
        return Ok(());
    }

    let (mut resolution, mut fps) = compatible_formats.pop().unwrap();
    if let Some(w) = args.width {
        resolution.width_x = w;
    }
    if let Some(h) = args.height {
        resolution.height_y = h;
    }

    fps.sort_by(|a, b| b.cmp(a));

    format.set_resolution(resolution);
    if let Some(fps) = args.fps {
        format.set_frame_rate(fps);
    } else {
        format.set_frame_rate(fps.pop().unwrap())
    }

    log::info!(
        "Camera Format selected: {}x{}@{}fps",
        format.resolution().width(),
        format.resolution().height(),
        format.frame_rate()
    );

    camera.set_camera_format(format)?;

    let (w, h) = (
        camera.camera_format().resolution().width(),
        camera.camera_format().resolution().height(),
    );

    log::info!(
        "Active Camera Format: {w}x{h}@{}fps",
        camera.camera_format().frame_rate(),
    );

    let fake_cam_info = match devices.iter().find(|info| info.human_name() == "fake-cam") {
        Some(info) => info,
        None => {
            return Err(HideError::FakeCameraMissing);
        }
    };

    let mut fake_camera = v4l::Device::new(fake_cam_info.index().as_index().unwrap() as usize)?;
    use v4l::video::Output as _;
    let format = v4l::Format::new(w, h, v4l::FourCC::new(b"RGB3"));
    let fake_fmt = fake_camera.set_format(&format)?;
    log::info!("Fake Camera found at index #{}", fake_cam_info.index());
    log::debug!(
        "Fake camera format: {:?}",
        std::str::from_utf8(&fake_fmt.fourcc.repr).unwrap()
    );

    let buf_size = camera.min_buffer_size(true);
    log::debug!("Buffer size: {buf_size}");
    let mut frame =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(w, h, vec![0; buf_size]).unwrap();

    log::info!("Loading ML Inference model...");
    let mut rvm = rvm::RobustVideoMatting::try_init(args.model)?;

    camera.open_stream()?;

    // Warmup
    for _ in 0..5 {
        let _ = camera.frame()?;
    }

    let background_pixel = [127, 212, 255, 255].into();
    let result = image::ImageBuffer::from_pixel(w, h, background_pixel);
    let mut result = image::DynamicImage::ImageRgba8(result);

    loop {
        camera.frame_to_buffer(&mut frame, true)?;

        // let mut blurred_bg = image::imageops::blur(&buf_u8, 12.);

        log::debug!("Got camera frame [len = {}]", frame.len());

        rvm.infer(&mut frame, (w, h))?;

        use image::GenericImageView as _;
        image::imageops::overlay(&mut result, &frame.view(0, 0, w, h), 0, 0);

        use std::io::Write as _;
        fake_camera.write_all(&result.to_rgb8())?;
        // Reset result
        for pixel in result.as_mut_rgba8().unwrap().enumerate_pixels_mut() {
            *pixel.2 = background_pixel;
        }

        frame.fill(0);
    }

    //camera.stop_stream()?;
    //Ok(())
}
