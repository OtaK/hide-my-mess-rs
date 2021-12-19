#[derive(Debug, thiserror::Error)]
pub enum HideError {
    #[error(transparent)]
    TensorflowError(#[from] tensorflow::Status),
    #[error(transparent)]
    Webcam(#[from] nokhwa::NokhwaError),
    #[error(transparent)]
    Other(#[from] eyre::Report),
}

pub type HideResult<T> = Result<T, HideError>;
