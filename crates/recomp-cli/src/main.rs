use clap::{Parser, Subcommand, ValueEnum};
mod automation;
use recomp_pipeline::bundle::{package_bundle, PackageOptions};
use recomp_pipeline::homebrew::{
    intake_homebrew, lift_homebrew, IntakeOptions, LiftMode, LiftOptions,
};
use recomp_pipeline::xci::{
    check_intake_manifest, intake_nca, intake_xci, NcaIntakeOptions, XciIntakeOptions,
    XciToolPreference,
};
use recomp_pipeline::{run_pipeline, PipelineOptions};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Static recompilation pipeline driver", version)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Run(RunArgs),
    Package(PackageArgs),
    Automate(AutomateArgs),
    HomebrewIntake(HomebrewIntakeArgs),
    HomebrewLift(HomebrewLiftArgs),
    XciIntake(XciIntakeArgs),
    NcaIntake(NcaIntakeArgs),
    XciValidate(XciValidateArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    provenance: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value = "../crates/recomp-runtime")]
    runtime_path: PathBuf,
}

#[derive(Parser, Debug)]
struct PackageArgs {
    #[arg(long)]
    project_dir: PathBuf,
    #[arg(long)]
    provenance: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    assets_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct AutomateArgs {
    #[arg(long)]
    config: PathBuf,
}

#[derive(Parser, Debug)]
struct HomebrewIntakeArgs {
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    nso: Vec<PathBuf>,
    #[arg(long)]
    provenance: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    apply_relocations: bool,
    /// Resolve relocations only for analysis; keep segments as loaded so the
    /// title's rtld links them at boot.
    #[arg(long, default_value_t = false)]
    link_at_runtime: bool,
}

#[derive(Parser, Debug)]
struct HomebrewLiftArgs {
    #[arg(long)]
    module_json: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value = "entry")]
    entry: String,
    /// Hex entry address (memory-relative) past the NSO MOD0 header.
    #[arg(long)]
    entry_address: Option<String>,
    /// Lift only segments from this module name (skips placeholder modules).
    #[arg(long)]
    module_name: Option<String>,
    /// Extra function entry point (hex, absolute); repeatable.
    #[arg(long = "extra-entry")]
    extra_entries: Vec<String>,
    #[arg(long, value_enum, default_value = "decode")]
    mode: HomebrewLiftMode,
}

#[derive(Parser, Debug)]
struct XciIntakeArgs {
    #[arg(long)]
    xci: PathBuf,
    #[arg(long)]
    keys: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    provenance: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    assets_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    xci_tool: XciToolPreferenceArg,
    #[arg(long)]
    xci_tool_path: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct NcaIntakeArgs {
    #[arg(long)]
    nca: PathBuf,
    #[arg(long)]
    keys: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    provenance: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    assets_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    xci_tool: XciToolPreferenceArg,
    #[arg(long)]
    xci_tool_path: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct XciValidateArgs {
    #[arg(long)]
    manifest: PathBuf,
}

#[derive(ValueEnum, Debug, Clone)]
enum XciToolPreferenceArg {
    Auto,
    Hactool,
    Hactoolnet,
    Mock,
}

#[derive(ValueEnum, Debug, Clone)]
enum HomebrewLiftMode {
    Stub,
    Decode,
}

impl From<HomebrewLiftMode> for LiftMode {
    fn from(value: HomebrewLiftMode) -> Self {
        match value {
            HomebrewLiftMode::Stub => LiftMode::Stub,
            HomebrewLiftMode::Decode => LiftMode::Decode,
        }
    }
}

impl From<XciToolPreferenceArg> for XciToolPreference {
    fn from(value: XciToolPreferenceArg) -> Self {
        match value {
            XciToolPreferenceArg::Auto => XciToolPreference::Auto,
            XciToolPreferenceArg::Hactool => XciToolPreference::Hactool,
            XciToolPreferenceArg::Hactoolnet => XciToolPreference::Hactoolnet,
            XciToolPreferenceArg::Mock => XciToolPreference::Mock,
        }
    }
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Run(run) => {
            let options = PipelineOptions {
                module_path: run.module,
                config_path: run.config,
                provenance_path: run.provenance,
                out_dir: run.out_dir,
                runtime_path: run.runtime_path,
            };

            match run_pipeline(options) {
                Ok(report) => {
                    println!(
                        "Wrote {} files to {}",
                        report.files_written.len(),
                        report.out_dir.display()
                    );
                    for input in report.detected_inputs {
                        println!(
                            "Input {} format={} sha256={} size={}",
                            input.path.display(),
                            input.format.as_str(),
                            input.sha256,
                            input.size
                        );
                    }
                }
                Err(err) => {
                    eprintln!("Pipeline error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Package(package) => {
            let options = PackageOptions {
                project_dir: package.project_dir,
                provenance_path: package.provenance,
                out_dir: package.out_dir,
                assets_dir: package.assets_dir,
            };
            match package_bundle(options) {
                Ok(report) => {
                    println!(
                        "Packaged bundle at {} ({} files)",
                        report.out_dir.display(),
                        report.files_written.len()
                    );
                    println!("Bundle manifest: {}", report.manifest_path.display());
                }
                Err(err) => {
                    eprintln!("Packaging error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Automate(automate) => match automation::run_automation(&automate.config) {
            Ok(report) => {
                let cache_hits = report.steps.iter().filter(|step| step.cache_hit).count();
                println!(
                    "Automation run {} completed ({}/{} cache hits)",
                    report.run_id,
                    cache_hits,
                    report.steps.len()
                );
                println!("Execution mode: {}", report.execution_mode);
                println!("Input fingerprint: {}", report.input_fingerprint);
                if let Some(path) = report.validation_report {
                    println!("Validation report: {}", path);
                }
            }
            Err(err) => {
                eprintln!("Automation error: {err}");
                std::process::exit(1);
            }
        },
        Command::HomebrewIntake(intake) => {
            let options = IntakeOptions {
                module_path: intake.module,
                nso_paths: intake.nso,
                provenance_path: intake.provenance,
                out_dir: intake.out_dir,
                apply_relocations: intake.apply_relocations,
                link_at_runtime: intake.link_at_runtime,
            };
            match intake_homebrew(options) {
                Ok(report) => {
                    println!(
                        "Homebrew intake wrote {} files to {}",
                        report.files_written.len(),
                        report.out_dir.display()
                    );
                    println!("module.json: {}", report.module_json_path.display());
                    println!("manifest.json: {}", report.manifest_path.display());
                    for (name, rel_report) in &report.relocations_applied {
                        println!(
                            "relocations {name}: relative={} symbol={} unresolved={} skipped={}",
                            rel_report.relative,
                            rel_report.symbol,
                            rel_report.unresolved,
                            rel_report.skipped
                        );
                        if !rel_report.unresolved_names.is_empty() {
                            println!(
                                "  unresolved symbols: {}",
                                rel_report.unresolved_names.join(", ")
                            );
                        }
                    }
                    for warning in &report.relocation_warnings {
                        println!("relocation warning: {warning}");
                    }
                }
                Err(err) => {
                    eprintln!("Homebrew intake error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::HomebrewLift(lift) => {
            let entry_address = match &lift.entry_address {
                Some(raw) => match u64::from_str_radix(raw.trim().trim_start_matches("0x"), 16) {
                    Ok(value) => Some(value),
                    Err(err) => {
                        eprintln!("invalid entry address {raw}: {err}");
                        std::process::exit(1);
                    }
                },
                None => None,
            };
            let mut extra_entries = Vec::new();
            for raw in &lift.extra_entries {
                let raw = raw.trim();
                match u64::from_str_radix(raw.trim_start_matches("0x"), 16) {
                    Ok(value) => extra_entries.push(value),
                    Err(err) => {
                        eprintln!("invalid extra entry {raw}: {err}");
                        std::process::exit(1);
                    }
                }
            }
            let options = LiftOptions {
                module_json_path: lift.module_json,
                out_dir: lift.out_dir,
                entry_name: lift.entry,
                entry_address,
                module_filter: lift.module_name,
                extra_entries,
                mode: lift.mode.into(),
            };
            match lift_homebrew(options) {
                Ok(report) => {
                    println!(
                        "Homebrew lift wrote {} functions to {}",
                        report.functions_emitted,
                        report.module_json_path.display()
                    );
                    if !report.warnings.is_empty() {
                        println!("Warnings:");
                        for warning in report.warnings {
                            println!("- {}", warning);
                        }
                    }
                }
                Err(err) => {
                    eprintln!("Homebrew lift error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::XciIntake(intake) => {
            let assets_dir = intake
                .assets_dir
                .unwrap_or_else(|| intake.out_dir.join("assets"));
            let options = XciIntakeOptions {
                xci_path: intake.xci,
                keys_path: intake.keys,
                config_path: intake.config,
                provenance_path: intake.provenance,
                out_dir: intake.out_dir,
                assets_dir,
                tool_preference: intake.xci_tool.into(),
                tool_path: intake.xci_tool_path,
            };
            match intake_xci(options) {
                Ok(report) => {
                    println!(
                        "XCI intake wrote {} files to {}",
                        report.files_written.len(),
                        report.out_dir.display()
                    );
                    println!("module.json: {}", report.module_json_path.display());
                    println!("manifest.json: {}", report.manifest_path.display());
                }
                Err(err) => {
                    eprintln!("XCI intake error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::NcaIntake(intake) => {
            let assets_dir = intake
                .assets_dir
                .unwrap_or_else(|| intake.out_dir.join("assets"));
            let options = NcaIntakeOptions {
                nca_path: intake.nca,
                keys_path: intake.keys,
                config_path: intake.config,
                provenance_path: intake.provenance,
                out_dir: intake.out_dir,
                assets_dir,
                tool_preference: intake.xci_tool.into(),
                tool_path: intake.xci_tool_path,
            };
            match intake_nca(options) {
                Ok(report) => {
                    println!(
                        "NCA intake wrote {} files to {}",
                        report.files_written.len(),
                        report.out_dir.display()
                    );
                    println!("module.json: {}", report.module_json_path.display());
                    println!("manifest.json: {}", report.manifest_path.display());
                }
                Err(err) => {
                    eprintln!("NCA intake error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::XciValidate(validate) => match check_intake_manifest(&validate.manifest) {
            Ok(check) => {
                if !check.missing_files.is_empty() {
                    eprintln!("XCI intake manifest has missing files:");
                    for missing in &check.missing_files {
                        eprintln!("- {missing}");
                    }
                    std::process::exit(1);
                }
                let program = &check.manifest.program;
                println!(
                    "XCI intake manifest ok: title_id={} name={} version={}",
                    program.title_id,
                    program.name.as_deref().unwrap_or("unknown"),
                    program.version
                );
                println!(
                    "assets: {} generated_files: {}",
                    check.manifest.assets.len(),
                    check.manifest.generated_files.len()
                );
            }
            Err(err) => {
                eprintln!("XCI intake manifest error: {err}");
                std::process::exit(1);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_includes_automate_subcommand() {
        let command = Args::command();
        assert!(command
            .get_subcommands()
            .any(|subcommand| subcommand.get_name() == "automate"));
    }
}
