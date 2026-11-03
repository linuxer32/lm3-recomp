//! Where the runtime looks for user-supplied files.
//!
//! Generated programs record the paths they were built with, but a packaged
//! build cannot use a build machine's `/home/...` path. Launchers place the
//! files beside the executable (or in its `assets/` directory) instead, or
//! name them through environment variables. Nothing here reads file
//! contents: proprietary title data stays outside the repository and the
//! generated project.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Directory that holds packaged assets, overriding the executable's
/// directory.
pub const ASSETS_DIR_ENV: &str = "RECOMP_ASSETS_DIR";
/// Android's SDL/JNI launcher sets this spelling; accept it as an alias.
pub const ASSETS_ENV: &str = "RECOMP_ASSETS";

/// Directories searched for a relative asset, most specific first:
/// `RECOMP_ASSETS_DIR`, the executable's directory, its `assets/`
/// subdirectory, then the current directory.
pub fn asset_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for name in [ASSETS_DIR_ENV, ASSETS_ENV] {
        if let Some(dir) = std::env::var_os(name).filter(|value| !value.is_empty()) {
            roots.push(PathBuf::from(dir));
            break;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
            roots.push(dir.join("assets"));
        }
    }
    if let Ok(dir) = std::env::current_dir() {
        roots.push(dir);
    }
    roots
}

/// Resolve a file such as a script or a RomFS image.
///
/// `env_var` is the override (`RECOMP_SCRIPT`, `RECOMP_ROMFS`): when set, its
/// value wins and an empty value disables the asset. Otherwise the relative
/// name is looked up under [`asset_roots`], and finally the build-time
/// `fallback` path is returned so an unpackaged build keeps working. The
/// fallback is returned even if it does not exist; the caller reports the
/// read error, as before.
pub fn resolve_asset(
    env_var: &str,
    relative: Option<&str>,
    fallback: Option<&str>,
) -> Option<PathBuf> {
    resolve_asset_from(
        &asset_roots(),
        std::env::var_os(env_var),
        relative,
        fallback,
    )
}

/// Resolve a file that has no environment override of its own (a memory
/// segment). Always returns a path: the first existing candidate, else the
/// build-time `fallback`, else `relative` itself, so the caller's read error
/// names a useful location.
pub fn resolve_relative_asset(relative: &str, fallback: Option<&str>) -> PathBuf {
    resolve_asset_from(&asset_roots(), None, Some(relative), fallback)
        .unwrap_or_else(|| PathBuf::from(relative))
}

/// The pure resolution step, with the process environment and roots passed
/// in so it can be tested without touching global state.
fn resolve_asset_from(
    roots: &[PathBuf],
    env_value: Option<OsString>,
    relative: Option<&str>,
    fallback: Option<&str>,
) -> Option<PathBuf> {
    if let Some(value) = env_value {
        return if value.is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        };
    }
    if let Some(relative) = relative.filter(|relative| !relative.is_empty()) {
        if Path::new(relative).is_absolute() {
            return Some(PathBuf::from(relative));
        }
        for root in roots {
            let candidate = root.join(relative);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    fallback
        .filter(|fallback| !fallback.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("recomp-assets-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn an_env_override_wins_and_an_empty_one_disables_the_asset() {
        let roots = vec![PathBuf::from("/nonexistent")];
        assert_eq!(
            resolve_asset_from(
                &roots,
                Some(OsString::from("/user/romfs.bin")),
                Some("romfs.bin"),
                Some("/build/romfs.bin")
            ),
            Some(PathBuf::from("/user/romfs.bin"))
        );
        assert_eq!(
            resolve_asset_from(
                &roots,
                Some(OsString::new()),
                Some("romfs.bin"),
                Some("/build/romfs.bin")
            ),
            None
        );
    }

    #[test]
    fn a_packaged_copy_wins_over_a_stale_build_path() {
        let root = temp_dir("packaged");
        std::fs::create_dir_all(root.join("scripts")).expect("create scripts");
        let packaged = root.join("scripts/game.rhai");
        std::fs::write(&packaged, b"// script").expect("write script");
        let roots = vec![root.clone()];
        assert_eq!(
            resolve_asset_from(
                &roots,
                None,
                Some("scripts/game.rhai"),
                Some("/home/old/scripts/game.rhai")
            ),
            Some(packaged)
        );
        std::fs::remove_dir_all(&root).expect("remove temp dir");
    }

    #[test]
    fn the_build_path_is_the_fallback_when_nothing_is_packaged() {
        let roots = vec![PathBuf::from("/nonexistent")];
        assert_eq!(
            resolve_asset_from(
                &roots,
                None,
                Some("segments/text.bin"),
                Some("/build/segments/text.bin")
            ),
            Some(PathBuf::from("/build/segments/text.bin"))
        );
        assert_eq!(
            resolve_asset_from(&roots, None, Some("segments/text.bin"), None),
            None
        );
    }

    #[test]
    fn a_relative_asset_falls_back_to_the_relative_path() {
        let path = resolve_relative_asset(
            "recomp-assets-missing/file.bin",
            Some("/nonexistent/fallback.bin"),
        );
        assert_eq!(path, PathBuf::from("/nonexistent/fallback.bin"));
        let path = resolve_relative_asset("recomp-assets-missing/file.bin", None);
        assert_eq!(path, PathBuf::from("recomp-assets-missing/file.bin"));
    }
}
