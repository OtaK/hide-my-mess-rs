mod error;
mod rvm;

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

fn main() -> error::HideResult<()> {
    pretty_env_logger::init();

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

fn main_next() -> error::HideResult<()> {
    use clap::StructOpt as _;
    let args = Args::parse();

    let devices = nokhwa::query_devices(nokhwa::CaptureAPIBackend::Auto)?;

    if args.list_devices {
        if devices.is_empty() {
            log::info!("No devices found on the system!");
        } else {
            log::info!("Found {} device(s) currently connected to this system: ", devices.len());
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

    let mut camera = nokhwa::Camera::new(
        camera_index,
        None,
    )?;

    log::info!("Selected camera: #{} - {}", camera.info().index(), camera.info().human_name());

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

    log::info!(
        "Active Camera Format: {}x{}@{}fps",
        camera.camera_format().resolution().width(),
        camera.camera_format().resolution().height(),
        camera.camera_format().frame_rate(),
    );

    log::debug!("Loading ML model...");
    let rvm = rvm::RobustVideoMatting::try_init()?;

    camera.open_stream()?;

    loop {
        let frame = camera.frame()?;
        rvm.run(frame)?;
        buf.clear();
    }

    camera.stop_stream()?;
    Ok(())
}
