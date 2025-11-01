// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::env;
use std::path::Path;
use std::path::PathBuf;

use anki_io::create_dir_all;
use anki_io::read_file;
use anki_io::write_file_if_changed;
use anki_proto_gen::add_must_use_annotations;
use anki_proto_gen::determine_if_message_is_empty;
use anyhow::Context;
use anyhow::Result;
use prost_reflect::DescriptorPool;

pub fn write_rust_protos(descriptors_path: PathBuf) -> Result<DescriptorPool> {
    set_protoc_path();
    let proto_dir = PathBuf::from("../../proto");
    let paths = gather_proto_paths(&proto_dir)?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tmp_descriptors = out_dir.join("descriptors.tmp");
    prost_build::Config::new()
        .out_dir(&out_dir)
        .file_descriptor_set_path(&tmp_descriptors)
        .type_attribute(
            "Deck.Filtered.SearchTerm.Order",
            "#[derive(strum::EnumIter)]",
        )
        .type_attribute(
            "Deck.Normal.DayLimit",
            "#[derive(Eq, serde::Deserialize, serde::Serialize)]",
        )
        .type_attribute("HelpPageLinkRequest.HelpPage", "#[derive(strum::EnumIter)]")
        .type_attribute("CsvMetadata.Delimiter", "#[derive(strum::EnumIter)]")
        .type_attribute(
            "Preferences.BackupLimits",
            "#[derive(serde::Deserialize, serde::Serialize)]",
        )
        .type_attribute(
            "CsvMetadata.DupeResolution",
            "#[derive(serde::Deserialize, serde::Serialize)]",
        )
        .type_attribute(
            "CsvMetadata.MatchScope",
            "#[derive(serde::Deserialize, serde::Serialize)]",
        )
        .type_attribute(
            "ImportAnkiPackageUpdateCondition",
            "#[derive(serde::Deserialize, serde::Serialize)]",
        )
        .compile_protos(paths.as_slice(), &[proto_dir])
        .context("prost build")?;

    let descriptors = read_file(&tmp_descriptors)?;
    create_dir_all(
        descriptors_path
            .parent()
            .context("missing parent of descriptor")?,
    )?;
    write_file_if_changed(descriptors_path, &descriptors)?;

    let pool = DescriptorPool::decode(descriptors.as_ref())?;
    add_must_use_annotations(
        &out_dir,
        |path| path.file_name().unwrap().starts_with("anki."),
        |path, name| determine_if_message_is_empty(&pool, path, name),
    )?;
    Ok(pool)
}

fn gather_proto_paths(proto_dir: &Path) -> Result<Vec<PathBuf>> {
    let subfolders = &["anki"];
    let mut paths = vec![];
    for subfolder in subfolders {
        for entry in proto_dir.join(subfolder).read_dir().unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(".proto")
            {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                paths.push(path);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// Set PROTOC to the custom path provided by PROTOC_BINARY, or add .exe to
/// the standard path if on Windows.
fn set_protoc_path() {
    match env::var("PROTOC") {
        Ok(value) => println!("cargo:warning=existing PROTOC env value: {value}"),
        Err(_) => println!("cargo:warning=no existing PROTOC env value"),
    }
    if let Ok(custom_protoc) = env::var("PROTOC_BINARY") {
        let path = Path::new(&custom_protoc);
        if path.exists() && path.is_file() {
            #[cfg(unix)]
            if let Ok(metadata) = path.metadata() {
                use std::os::unix::fs::PermissionsExt;
                if metadata.permissions().mode() & 0o111 != 0 {
                    env::set_var("PROTOC", custom_protoc);
                    return;
                } else {
                    println!(
                        "cargo:warning=PROTOC_BINARY path exists but is not executable: {}",
                        custom_protoc
                    );
                }
            } else {
                println!(
                    "cargo:warning=Could not get metadata for PROTOC_BINARY: {}",
                    custom_protoc
                );
            }
            #[cfg(not(unix))]
            {
                env::set_var("PROTOC", custom_protoc);
                return;
            }
        } else {
            println!(
                "cargo:warning=PROTOC_BINARY path does not exist or is not a file: {}",
                custom_protoc
            );
        }
    }

    if let Ok(bundled_protoc) = env::var("PROTOC") {
        let mut path = PathBuf::from(&bundled_protoc);
        if cfg!(windows) && path.extension().is_none() {
            path.set_extension("exe");
        }
        if path.exists() {
            env::set_var("PROTOC", path);
            return;
        }
        println!(
            "cargo:warning=bundled protoc missing at {}, falling back to vendored binary",
            path.display()
        );
    }

    match protoc_bin_vendored::protoc_bin_path() {
        Ok(vendored_path) => {
            println!(
                "cargo:warning=using vendored protoc binary at {}",
                vendored_path.display()
            );
            env::set_var("PROTOC", vendored_path.to_string_lossy().into_owned());
        }
        Err(error) => {
            eprintln!("Error: Failed to locate vendored protoc binary: {}", error);
            std::process::exit(1);
        }
    }
}
