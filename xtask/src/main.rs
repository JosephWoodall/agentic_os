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
        /// LLM provider (mock, ollama, openai, anthropic)
        #[arg(long, default_value = "ollama")]
        llm_provider: String,
        /// LLM model name
        #[arg(long, default_value = "qwen2.5:32b")]
        llm_model: String,
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
        Commands::Run { release, graphics, llm_provider, llm_model } => {
            build(&sh, release)?;
            run(&sh, release, graphics, &llm_provider, &llm_model)?;
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

    sh.create_dir("target")?;

    if !Path::new(img_path).exists() {
        println!("Creating boot disk image...");
        cmd!(sh, "dd if=/dev/zero of={img_path} bs=1M count=64").run()?;
        cmd!(sh, "mformat -i {img_path} ::").run()?;
        cmd!(sh, "mmd -i {img_path} ::/EFI").run()?;
        cmd!(sh, "mmd -i {img_path} ::/EFI/BOOT").run()?;
    } else {
        println!("Updating existing boot disk image...");
        // Remove the old EFI file if it exists so mcopy doesn't prompt for overwrite
        let _ = cmd!(sh, "mdel -i {img_path} ::/EFI/BOOT/BOOTX64.EFI").run();
    }
    cmd!(sh, "mcopy -i {img_path} {kernel_efi} ::/EFI/BOOT/BOOTX64.EFI").run()?;

    let weights_path = "target/weights.img";
    if !Path::new(weights_path).exists() {
        println!("Creating weights disk image...");
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
    } else {
        println!("Reusing existing weights disk image...");
    }

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
        "/usr/share/ovmf/x64/OVMF.fd",
        "/usr/share/edk2-ovmf/x64/OVMF.fd",
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

fn run(sh: &Shell, _release: bool, graphics: bool, provider: &str, model: &str) -> anyhow::Result<()> {
    if provider == "ollama" {
        println!("Checking for Ollama...");
        let client = std::net::TcpStream::connect_timeout(
            &"127.0.0.1:11434".parse().unwrap(),
            std::time::Duration::from_millis(500)
        );

        if client.is_err() {
            println!("Ollama not found. Starting 'ollama serve' in background...");
            std::process::Command::new("ollama")
                .arg("serve")
                .spawn()?;
            
            // Wait for Ollama to wake up
            println!("Waiting for Ollama to be ready...");
            for _ in 0..15 {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if std::net::TcpStream::connect("127.0.0.1:11434").is_ok() {
                    println!("Ollama is ready!");
                    break;
                }
            }
        } else {
            println!("Ollama is already running.");
        }
    }

    println!("Starting LLM bridge (Provider: {}, Model: {})...", provider, model);
    
    // Check if llm_bridge.py exists
    if !Path::new("llm_bridge.py").exists() {
        println!("Warning: llm_bridge.py not found. Running without LLM bridge.");
    } else {
        // Spawn the bridge as a background process using std::process::Command
        // Inject the provider and model directly into the process environment.
        std::process::Command::new("python3")
            .arg("-u")
            .arg("llm_bridge.py")
            .env("LLM_PROVIDER", provider)
            .env("LLM_MODEL", model)
            .spawn()?;
        println!("LLM bridge started in background.");
    }

    println!("Running in QEMU...");

    let img_path = "target/esp.img";
    let weights_path = "target/weights.img";
    let ovmf_path = find_ovmf()?;
    let kvm = if Path::new("/dev/kvm").exists() {
        vec!["-enable-kvm"]
    } else {
        vec![]
    };

    if graphics {
        cmd!(
            sh,
            "qemu-system-x86_64
                {kvm...}
                -bios {ovmf_path}
                -drive format=raw,file={img_path},index=0,media=disk
                -drive format=raw,file={weights_path},index=1,media=disk
                -m 4G
                -net none
                -serial tcp:127.0.0.1:5557,server,nowait
                -usb
                -device usb-mouse
                -monitor tcp:127.0.0.1:5556,server,nowait"
        )
        .run()?;
    } else {
        cmd!(
            sh,
            "qemu-system-x86_64
                {kvm...}
                -bios {ovmf_path}
                -drive format=raw,file={img_path},index=0,media=disk
                -drive format=raw,file={weights_path},index=1,media=disk
                -m 4G
                -net none
                -nographic
                -serial tcp:127.0.0.1:5557,server,nowait
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
