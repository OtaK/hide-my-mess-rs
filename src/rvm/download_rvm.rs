use crate::{HideError, HideResult};

const CACHE_DIR_NAME: &str = "hide-my-mess";

const REPOSITORY_URL: &str = "https://github.com/PeterL1n/RobustVideoMatting";
const TARGET_VERSION: &str = "v1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvmModelKind {
    MobileNetV3,
    Resnet50,
}

impl Default for RvmModelKind {
    fn default() -> Self {
        Self::MobileNetV3
    }
}

impl RvmModelKind {
    fn to_filename(&self) -> &str {
        match self {
            RvmModelKind::MobileNetV3 => "rvm_mobilenetv3_tf",
            RvmModelKind::Resnet50 => "rvm_resnet50_tf",
        }
    }
}

impl std::fmt::Display for RvmModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RvmModelKind::MobileNetV3 => "mobilenetv3",
                RvmModelKind::Resnet50 => "resnet50",
            }
        )
    }
}

impl std::str::FromStr for RvmModelKind {
    type Err = HideError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "mobilenetv3" => Self::MobileNetV3,
            "resnet50" => Self::Resnet50,
            i => return Err(HideError::InvalidModel(i.to_string())),
        })
    }
}

pub fn download_rvm_model(kind: RvmModelKind) -> HideResult<std::path::PathBuf> {
    let target_dir = dirs::cache_dir()
        .ok_or(HideError::CacheDirError)?
        .join(CACHE_DIR_NAME)
        .join("models");

    let model_folder_name = kind.to_filename();
    let model_folder = target_dir.join(model_folder_name);

    if model_folder.exists() {
        return Ok(model_folder);
    }

    std::fs::create_dir_all(&target_dir)?;

    let download_url = format!(
        "{}/releases/download/{}/{}.zip",
        REPOSITORY_URL, TARGET_VERSION, model_folder_name
    );

    let res = attohttpc::get(&download_url).send()?;

    log::info!("Downloading model at {download_url}...");

    let zipfile_path = target_dir.join(format!("{}.zip", model_folder_name));
    let file = std::fs::File::create(&zipfile_path)?;
    // Pipe the download to the file
    res.write_to(file)?;

    let zip_reader = std::fs::File::open(&zipfile_path)?;
    // Unzip that bad boy
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    zip.extract(&target_dir)?;

    // Remove zip file
    std::fs::remove_file(&zipfile_path)?;

    Ok(model_folder)
}

#[cfg(test)]
mod tests {
    use super::{download_rvm_model, RvmModelKind};

    #[test]
    fn can_unzip_mobilenet_model() {
        let _ = pretty_env_logger::try_init();
        let path = download_rvm_model(RvmModelKind::MobileNetV3).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn can_unzip_resnet_model() {
        let _ = pretty_env_logger::try_init();
        let path = download_rvm_model(RvmModelKind::Resnet50).unwrap();
        assert!(path.exists());
    }
}
