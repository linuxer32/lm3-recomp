use recomp_pipeline::bundle::{
    bundle_manifest_self_hash, package_bundle, BundleManifest, PackageOptions,
};
use sha2::{Digest, Sha256};
use std::fs;
#[test]
fn bundle_manifest_checksums_match_contents() {
    let temp = tempfile::tempdir().expect("tempdir");
    let project_dir = temp.path().join("project");
    let provenance_path = temp.path().join("provenance.toml");
    let out_dir = temp.path().join("bundle");

    fs::create_dir_all(project_dir.join("src")).expect("create project");
    fs::write(project_dir.join("Cargo.toml"), "name = \"demo\"").expect("write Cargo.toml");
    fs::write(project_dir.join("src/main.rs"), "fn main() {}").expect("write main.rs");
    fs::write(project_dir.join("manifest.json"), "{}").expect("write manifest.json");
    fs::write(&provenance_path, "schema_version = \"1\"").expect("write provenance");

    let report = package_bundle(PackageOptions {
        project_dir: project_dir.clone(),
        provenance_path: provenance_path.clone(),
        out_dir: out_dir.clone(),
        assets_dir: None,
    })
    .expect("package bundle");

    let manifest_src = fs::read_to_string(&report.manifest_path).expect("read manifest");
    let manifest: BundleManifest = serde_json::from_str(&manifest_src).expect("parse manifest");

    assert_eq!(manifest.manifest_self_hash_basis, "files_self_placeholder");

    let mut saw_self = false;
    let manifest_size = manifest_src.len() as u64;
    for entry in &manifest.files {
        let path = out_dir.join(&entry.path);
        let bytes = fs::read(&path).expect("read file");
        if entry.path == "metadata/bundle-manifest.json" {
            saw_self = true;
            let expected_hash = bundle_manifest_self_hash(&manifest).expect("compute self hash");
            assert_eq!(
                expected_hash,
                entry.sha256,
                "checksum mismatch for {}",
                path.display()
            );
            assert_eq!(
                manifest_size,
                entry.size,
                "size mismatch for {}",
                path.display()
            );
            continue;
        }

        let sha = sha256_bytes(&bytes);
        assert_eq!(
            sha,
            entry.sha256,
            "checksum mismatch for {}",
            path.display()
        );
        assert_eq!(
            bytes.len() as u64,
            entry.size,
            "size mismatch for {}",
            path.display()
        );
    }

    assert!(saw_self, "bundle-manifest.json entry listed");
}

#[test]
fn bundle_copies_chunks_and_cargo_config() {
    let temp = tempfile::tempdir().expect("tempdir");
    let project_dir = temp.path().join("project");
    let provenance_path = temp.path().join("provenance.toml");
    let out_dir = temp.path().join("bundle");

    fs::create_dir_all(project_dir.join("src")).expect("create src");
    fs::create_dir_all(project_dir.join("chunks/chunk_0/src")).expect("create chunk");
    fs::create_dir_all(project_dir.join(".cargo")).expect("create cargo dir");
    fs::write(
        project_dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\".\", \"chunks/chunk_0\"]\n",
    )
    .expect("write Cargo.toml");
    fs::write(project_dir.join("manifest.json"), "{}").expect("write manifest.json");
    fs::write(project_dir.join("src/main.rs"), "fn main() {}").expect("write main.rs");
    fs::write(
        project_dir.join("chunks/chunk_0/Cargo.toml"),
        "[package]\nname = \"chunk_0\"\n",
    )
    .expect("write chunk Cargo.toml");
    fs::write(project_dir.join("chunks/chunk_0/src/lib.rs"), "").expect("write chunk lib.rs");
    fs::write(
        project_dir.join(".cargo/config.toml"),
        "[build]\nrustflags = [\"-C\", \"target-cpu=generic\"]\n",
    )
    .expect("write cargo config");
    fs::write(&provenance_path, "schema_version = \"1\"").expect("write provenance");

    let report = package_bundle(PackageOptions {
        project_dir,
        provenance_path,
        out_dir: out_dir.clone(),
        assets_dir: None,
    })
    .expect("package bundle");

    for path in [
        "code/chunks/chunk_0/Cargo.toml",
        "code/chunks/chunk_0/src/lib.rs",
        "code/.cargo/config.toml",
    ] {
        assert!(
            out_dir.join(path).is_file(),
            "{path} was not packaged; wrote {:?}",
            report.files_written
        );
    }

    let manifest_src = fs::read_to_string(&report.manifest_path).expect("read manifest");
    let manifest: BundleManifest = serde_json::from_str(&manifest_src).expect("parse manifest");
    let paths: Vec<&str> = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    assert!(
        paths.contains(&"code/chunks/chunk_0/Cargo.toml"),
        "{paths:?}"
    );
    assert!(paths.contains(&"code/.cargo/config.toml"), "{paths:?}");
    // Segments are title data and must never be copied into the code bundle.
    assert!(
        !paths.iter().any(|path| path.contains("segments/")),
        "{paths:?}"
    );
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}
