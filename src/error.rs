#[derive(Debug, thiserror::Error)]
pub enum HideError {
    #[error("vl42loopback device missing! Did you run `sudo modprobe v4l2loopback devices=1 exclusive_caps=1 max_buffers=2 video_nr=2 card_label=\"fake-cam\"`?")]
    FakeCameraMissing,
    #[error("Your capture device somehow supports NO capture formats! :(")]
    NoCameraFormats,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    ImageError(#[from] image::ImageError),
    #[error(transparent)]
    IntError(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    TensorflowError(#[from] tensorflow::Status),
    #[error(transparent)]
    Webcam(#[from] nokhwa::NokhwaError),
    #[error("Cannot get access to cache directory")]
    CacheDirError,
    #[error(transparent)]
    ModelDownloadError(#[from] attohttpc::Error),
    #[error("Invalid selected model: {0}. Possible choices are `resnet50` or `mobilenetv3`")]
    InvalidModel(String),
    #[error(transparent)]
    ModelUnzipError(#[from] zip::result::ZipError),
    #[error(transparent)]
    Other(#[from] eyre::Report),
}

pub type HideResult<T> = Result<T, HideError>;
