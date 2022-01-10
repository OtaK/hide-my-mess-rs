#![allow(dead_code)]
use crate::{HideError, HideResult};

const CACHE_DIR_NAME: &str = "hide-my-mess";

const REPOSITORY_URL: &str = "https://github.com/PeterL1n/RobustVideoMatting";
const TARGET_VERSION: &str = "v1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvmModelKind {
    MobileNetV3,
    Resnet50,
}

impl std::fmt::Display for RvmModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RvmModelKind::MobileNetV3 => "rvm_mobilenetv3_tf",
                RvmModelKind::Resnet50 => "rvm_resnet50_tf",
            }
        )
    }
}

pub fn download_rvm_model(kind: RvmModelKind) -> HideResult<std::path::PathBuf> {
    let target_dir = dirs::cache_dir()
        .ok_or(HideError::CacheDirError)?
        .join(CACHE_DIR_NAME)
        .join("models");

    let model_folder_name = kind.to_string();
    let model_folder = target_dir.join(kind.to_string());
    std::fs::create_dir_all(&model_folder)?;

    let download_url = format!(
        "{}/releases/download/{}/{}.zip",
        REPOSITORY_URL, TARGET_VERSION, model_folder_name
    );

    let res = ureq::get(&download_url).call()?;

    let progress_bar =
        if let Some(Ok(length)) = res.header("content-length").map(|len| len.parse::<u64>()) {
            indicatif::ProgressBar::new(length * 2) // we multiply by 2 because we count read & write as progress
        } else {
            indicatif::ProgressBar::new_spinner() // No content length so...we just spin
        };

    progress_bar.set_message(format!("Downloading {}...", download_url));

    // 8KB buffer
    let mut buf = Vec::with_capacity(8192);
    let zipfile_path = model_folder.join(format!("{}.zip", model_folder_name));
    let mut file = std::fs::File::create(&zipfile_path)?;

    let mut reader = res.into_reader();
    // Download the file into the zip file
    loop {
        use std::io::{Read as _, Write as _};
        reader.read_exact(&mut buf)?;
        let len = buf.len();
        progress_bar.inc(len as _);
        file.write_all(&buf)?;
        progress_bar.inc(len as _);
        if len < buf.capacity() {
            progress_bar.finish();
            break;
        }
        buf.clear();
    }

    let zip_reader = std::fs::File::open(&zipfile_path)?;
    // Unzip that bad boy
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    zip.extract(&model_folder)?;

    // Remove zip file
    std::fs::remove_file(&zipfile_path)?;

    Ok(model_folder)
}

#[cfg(test)]
mod tests {
    use super::{download_rvm_model, RvmModelKind};

    #[test]
    fn can_unzip_mobilenet_model() {
        let path = download_rvm_model(RvmModelKind::MobileNetV3).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn can_unzip_resnet_model() {
        let path = download_rvm_model(RvmModelKind::Resnet50).unwrap();
        assert!(path.exists());
    }
}
