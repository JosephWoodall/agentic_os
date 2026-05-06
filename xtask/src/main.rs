use std::path::Path;
use clap::{Parser, Subcommand};
use xshell::{cmd, Shell};

#[derive(Parser)]
#[command(author, version, about = "Agentic OS build runner")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build the kernel and create a bootable disk image
    Build {
        /// Build in release mode
        #[arg(long)]
        release: bool,
    },
    /// Build and run the kernel in QEMU
    Run {
        /// Build in release mode
        #[arg(long)]
        release: bool,
        /// Enable graphics mode (GOP framebuffer in QEMU window)
        #[arg(long)]
        graphics: bool,
    },
    /// Clean all build artifacts
    Clean,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match cli.command {
        Commands::Build { release } => {
            build(&sh, release)?;
        }
        Commands::Run { release, graphics } => {
            build(&sh, release)?;
            run(&sh, release, graphics)?;
        }
        Commands::Clean => {
            clean(&sh)?;
        }
    }

    Ok(())
}

fn build(sh: &Shell, release: bool) -> anyhow::Result<()> {
    println!("Building kernel...");

    if release {
        cmd!(sh, "cargo build --package kernel --target x86_64-unknown-uefi --release").run()?;
    } else {
        cmd!(sh, "cargo build --package kernel --target x86_64-unknown-uefi").run()?;
    }

    let profile = if release { "release" } else { "debug" };
    let kernel_efi = format!("target/x86_64-unknown-uefi/{}/kernel.efi", profile);
    let img_path = "target/esp.img";

    println!("Creating boot disk image...");
    sh.create_dir("target")?;

    cmd!(sh, "dd if=/dev/zero of={img_path} bs=1M count=64").run()?;
    cmd!(sh, "mformat -i {img_path} ::").run()?;
    cmd!(sh, "mmd -i {img_path} ::/EFI").run()?;
    cmd!(sh, "mmd -i {img_path} ::/EFI/BOOT").run()?;
    cmd!(sh, "mcopy -i {img_path} {kernel_efi} ::/EFI/BOOT/BOOTX64.EFI").run()?;

    println!("Creating weights disk image...");
    let weights_path = "target/weights.img";
    cmd!(sh, "dd if=/dev/zero of={weights_path} bs=1M count=10").run()?;

    // Create a valid GGUF file header with mock vocabulary metadata
    let mut mock_gguf = vec![0u8; 4096];
    let mut pos = 0;

    // Magic
    mock_gguf[pos..pos + 4].copy_from_slice(b"GGUF");
    pos += 4;
    // Version 3
    mock_gguf[pos..pos + 4].copy_from_slice(&3u32.to_le_bytes());
    pos += 4;
    // 0 tensors
    mock_gguf[pos..pos + 8].copy_from_slice(&0u64.to_le_bytes());
    pos += 8;
    // 1 metadata KV (model architecture)
    mock_gguf[pos..pos + 8].copy_from_slice(&1u64.to_le_bytes());
    pos += 8;

    // KV: "general.architecture" = "gemma"
    let key = b"general.architecture";
    mock_gguf[pos..pos + 8].copy_from_slice(&(key.len() as u64).to_le_bytes());
    pos += 8;
    mock_gguf[pos..pos + key.len()].copy_from_slice(key);
    pos += key.len();
    // Type: STRING (8)
    mock_gguf[pos..pos + 4].copy_from_slice(&8u32.to_le_bytes());
    pos += 4;
    let val = b"gemma";
    mock_gguf[pos..pos + 8].copy_from_slice(&(val.len() as u64).to_le_bytes());
    pos += 8;
    mock_gguf[pos..pos + val.len()].copy_from_slice(val);

    sh.write_file("target/mock_weights.bin", &mock_gguf)?;
    cmd!(sh, "dd if=target/mock_weights.bin of={weights_path} conv=notrunc").run()?;

    println!("Build complete. Images created in target/");
    Ok(())
}

fn find_ovmf() -> anyhow::Result<String> {
    // Check env var first
    if let Ok(path) = std::env::var("OVMF_PATH") {
        if Path::new(&path).exists() {
            return Ok(path);
        }
    }

    // Search common paths
    let candidates = [
        "/usr/share/edk2/x64/OVMF.4m.fd",
        "/usr/share/edk2/ovmf/OVMF.fd",
        "/usr/share/OVMF/OVMF.fd",
        "/usr/share/ovmf/OVMF.fd",
        "/usr/share/qemu/OVMF.fd",
    ];

    for path in &candidates {
        if Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }

    Err(anyhow::anyhow!(
        "OVMF firmware not found. Set OVMF_PATH env var or install edk2-ovmf."
    ))
}

fn run(sh: &Shell, _release: bool, graphics: bool) -> anyhow::Result<()> {
    println!("Running in QEMU...");

    let img_path = "target/esp.img";
    let weights_path = "target/weights.img";
    let ovmf_path = find_ovmf()?;

    if graphics {
        cmd!(
            sh,
            "qemu-system-x86_64
                -enable-kvm
                -bios {ovmf_path}
                -drive format=raw,file={img_path},index=0,media=disk
                -drive format=raw,file={weights_path},index=1,media=disk
                -m 16G
                -net none
                -serial stdio
                -usb
                -device usb-kbd
                -device usb-tablet
                -monitor tcp:127.0.0.1:5556,server,nowait"
        )
        .run()?;
    } else {
        cmd!(
            sh,
            "qemu-system-x86_64
                -enable-kvm
                -bios {ovmf_path}
                -drive format=raw,file={img_path},index=0,media=disk
                -drive format=raw,file={weights_path},index=1,media=disk
                -m 16G
                -net none
                -nographic
                -serial stdio
                -monitor none"
        )
        .run()?;
    }

    Ok(())
}

fn clean(sh: &Shell) -> anyhow::Result<()> {
    println!("Cleaning build artifacts...");
    cmd!(sh, "cargo clean").run()?;
    println!("Clean complete.");
    Ok(())
}
