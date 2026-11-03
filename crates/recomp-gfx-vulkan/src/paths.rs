//! Host paths for the files the backend writes when debugging
//! (`RECOMP_VULKAN_DUMP`, `RECOMP_VULKAN_CAPTURE`).
//!
//! A relative path is taken from the working directory, as before, except
//! when the launcher sets `RECOMP_BASE_DIR` (the Windows build script does)
//! or the host is Windows: there the executable's directory is the base, so
//! a title started from Explorer or a shortcut writes next to itself rather
//! than wherever the shell happened to be. The assets and save data keep
//! their existing rules: `[runtime.ipc] romfs_image` (and a `save/`
//! directory beside it) or `RECOMP_SAVE_DIR`.

use std::path::{Path, PathBuf};

/// The directory relative paths are resolved against, if the platform has
/// one: `RECOMP_BASE_DIR` when set, else the executable's directory on
/// Windows. Other hosts keep the working directory.
fn base_dir() -> Option<PathBuf> {
    let configured = std::env::var_os("RECOMP_BASE_DIR")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from);
    configured.or_else(|| {
        if cfg!(windows) {
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_path_buf))
        } else {
            None
        }
    })
}

/// `path` as an absolute path: unchanged when it already is one, else
/// joined to [`base_dir`] when there is one (the working directory is used
/// otherwise, by returning the relative path).
pub fn resolve(path: impl AsRef<Path>) -> PathBuf {
    resolve_under(base_dir().as_deref(), path.as_ref())
}

fn resolve_under(base: Option<&Path>, path: &Path) -> PathBuf {
    match base {
        Some(base) if path.is_relative() => base.join(path),
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_paths_are_kept() {
        let path = if cfg!(windows) {
            Path::new(r"C:\captures")
        } else {
            Path::new("/captures")
        };
        assert_eq!(resolve_under(Some(Path::new("base")), path), path);
    }

    #[test]
    fn relative_paths_use_the_base_directory_when_there_is_one() {
        assert_eq!(
            resolve_under(Some(Path::new("base")), Path::new("out/frames")),
            Path::new("base").join("out/frames")
        );
    }

    #[test]
    fn relative_paths_keep_the_working_directory_without_a_base() {
        assert_eq!(
            resolve_under(None, Path::new("out/frames")),
            Path::new("out/frames")
        );
    }
}
