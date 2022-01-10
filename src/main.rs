use crate::error::HideError;

mod error;
pub use self::error::*;

mod rvm;

#[inline(always)]
#[allow(dead_code)]
// Don't look it's scary
fn rgb_to_yuv(pixel: &image::Rgb<u8>) -> (u8, u8, u8) {
    let [r, g, b] = pixel.0;
    let (r, g, b) = (r as f32 / 255., g as f32 / 255., b as f32 / 255.);
    let y = ((0.299 * r + 0.587 * g + 0.114 * b) * 255.).floor() as u8;
    let u = ((-0.168736 * r - 0.331264 * g + 0.5 * b) * 255. + 128.).floor() as u8;
    let v = ((0.5 * r - 0.418688 * g - 0.081312 * b) * 255. + 128.).floor() as u8;
    (y, u, v)
}

#[derive(Debug, clap::Parser)]
#[clap(about, version, author)]
/// Virtual camera that blurs your background
struct Args {
    #[clap(short, long = "list")]
    /// Lists video capture devices present on the system along with their details
    list_devices: bool,
    #[clap(long)]
    /// Choose a video capture device index manually, in case you have several connected
    camera_index: Option<usize>,
    #[clap(short, long)]
    /// Desired frame capture width. If not available, it'll fallback to the highest mode detected
    width: Option<u32>,
    #[clap(short, long)]
    /// Desired frame capture height. If not available, it'll fallback to the highest mode detected
    height: Option<u32>,
    #[clap(long)]
    /// Desired capture framerate. If not available, it'll fallback to the highest mode detected
    fps: Option<u32>,
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

fn main_next() -> error::HideResult<()> {
    use clap::StructOpt as _;
    let args = Args::parse();

    let mut devices = nokhwa::query_devices(nokhwa::CaptureAPIBackend::Auto)?;
    devices.sort_by(|a, b| a.index().cmp(&b.index()));

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

    log::debug!("devices found: {:#?}", devices);
    if devices.is_empty() {
        log::error!("No devices found on the system!");
        return Ok(());
    }

    let camera_index = args.camera_index.unwrap_or_else(|| devices[0].index());
    log::debug!("Selected device index: #{}", camera_index);

    let mut camera = nokhwa::Camera::new(camera_index, None)?;

    log::info!(
        "Selected camera: #{} - {}",
        camera.info().index(),
        camera.info().human_name()
    );

    let mut compatible_formats = camera
        .compatible_camera_formats()?
        .into_iter()
        .filter(|f| f.format() == nokhwa::FrameFormat::MJPEG && f.frame_rate() >= 24)
        .collect::<Vec<nokhwa::CameraFormat>>();

    compatible_formats.sort_by(|a, b| b.resolution().cmp(&a.resolution()));

    if compatible_formats.is_empty() {
        log::error!("Your capture device somehow supports NO capture formats! :(");
        return Ok(());
    }

    let mut format = compatible_formats[0];
    let mut resolution = format.resolution();
    if let Some(w) = args.width {
        resolution.width_x = w;
    }
    if let Some(h) = args.height {
        resolution.height_y = h;
    }
    format.set_resolution(resolution);
    if let Some(fps) = args.fps {
        format.set_frame_rate(fps);
    }

    log::info!(
        "Camera Format selected: {}x{}@{}fps",
        format.resolution().width(),
        format.resolution().height(),
        format.frame_rate()
    );

    camera
        .set_camera_format(format)
        .or_else(|_| camera.set_camera_format(compatible_formats[0]))?;

    let (w, h) = (
        camera.camera_format().resolution().width(),
        camera.camera_format().resolution().height(),
    );

    let channels = 3u32;

    log::info!(
        "Active Camera Format: {}x{}@{}fps",
        w,
        h,
        camera.camera_format().frame_rate(),
    );

    let fake_cam_info = match devices.iter().find(|info| info.human_name() == "fake-cam") {
        Some(info) => info,
        None => {
            return Err(HideError::FakeCameraMissing);
        }
    };

    let mut fake_camera = v4l::Device::new(fake_cam_info.index())?;
    v4l::video::Output::set_format(
        &fake_camera,
        &v4l::Format::new(w, h, v4l::FourCC::new(b"MJPG")),
    )?;
    let fake_fmt = v4l::video::Output::format(&fake_camera)?;
    log::info!("Fake Camera found at index #{}", fake_cam_info.index());
    log::debug!(
        "Fake camera format: {:?}",
        std::str::from_utf8(&fake_fmt.fourcc.repr).unwrap()
    );

    let buf_size = camera.min_buffer_size(false);
    log::debug!("Buffer size: {}", buf_size);
    let mut frame: Vec<f32> = Vec::with_capacity(buf_size);
    // let mut yuv_buffer: Vec<u8> = Vec::with_capacity(buf_size);

    log::debug!("Loading ML model...");
    let mut rvm = rvm::RobustVideoMatting::try_init()?;

    camera.open_stream()?;
    let mut jpg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut fake_camera, 90);
    loop {
        let buf_u8 = camera.frame()?;
        let mut blurred_bg = image::imageops::blur(&buf_u8, 12.);

        // Normalize u8 to 0..1 f32 pixels
        frame.clear();
        frame = buf_u8.into_iter().map(|pix| *pix as f32 / 255.).collect();
        // for pix in buf_u8.into_iter() {
        //     frame.push(*pix as f32 / 255.);
        // }

        log::debug!("Got camera frame [len = {}]", frame.len());

        let fgr: Vec<u8> = rvm
            .run(&frame, (channels, w, h))?
            .into_iter()
            .map(|px| (px * 255.) as u8)
            .collect();

        let foreground = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w, h, fgr).unwrap();
        use image::GenericImageView as _;
        image::imageops::overlay(&mut blurred_bg, &foreground.view(0, 0, w, h), 0, 0);

        jpg_encoder.encode_image(&blurred_bg)?;
    }

    //camera.stop_stream()?;
    //Ok(())
}
