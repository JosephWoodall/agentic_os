//! Agentic OS — Probabilistic State-Space Operating System
//!
//! A bare-metal UEFI kernel where an LLM replaces the traditional scheduler.
//! The dividing line between hardware orchestration and cognitive reasoning vanishes.
//!
//! Architecture Pillars:
//! 1. Compute as Inference (Attention as Scheduling)
//! 2. Memory as a Continuous Latent Space
//! 3. Semantic Intent over POSIX APIs
//! 4. Emergent Security via Policy-as-Reasoning

#![no_std]
#![no_main]

extern crate alloc;

// Phase 1: Foundation
mod allocator;
mod framebuffer;
mod storage;

// Phase 1-7: C support
mod libc_stub;

// Phase 2: Inference
mod tensor;
mod tokenizer;
mod transformer;
mod gguf;
mod inference;

// Phase 3: Tool Call Substrate
mod syscall;
mod grammar;

// Phase 4: State & Context
mod state;
mod context;
mod compressor;

// Phase 5: Executive Loop
mod scheduler;

// Phase 6: PID 1 & Shell
mod keyboard;
mod shell;

// Phase 7: UI & Desktop
mod mouse;
mod compositor;
mod ui;
mod desktop;

use uefi::prelude::*;
use log::info;
use uefi::proto::console::gop::GraphicsOutput;
use alloc::string::String;
use crate::allocator::ALLOCATOR;
use crate::framebuffer::Framebuffer;
use crate::storage::BlockDevice;
use crate::inference::InferenceEngine;
use crate::scheduler::ExecutiveLoop;
use crate::keyboard::Keyboard;
use crate::shell::Shell;
use crate::mouse::Mouse;
use crate::compositor::Compositor;
use crate::desktop::DesktopAgent;

/// Heap size: 2GB for model weights + working memory.
const HEAP_SIZE: usize = 2 * 1024 * 1024 * 1024;

#[entry]
fn main(_handle: Handle, mut system_table: SystemTable<Boot>) -> Status {
    // ═══════════════════════════════════════════════════════════════════
    // PHASE 0: Initialize UEFI services and logging
    // ═══════════════════════════════════════════════════════════════════
    uefi::helpers::init().unwrap();
    
    // Disable watchdog timer (Code >= 0x10000 for OS use)
    let _ = system_table.boot_services().set_watchdog_timer(0, 0x10000, None);

    info!("════════════════════════════════════════════");
    info!("  AGENTIC OS v0.1 — Probabilistic Kernel   ");
    info!("════════════════════════════════════════════");

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Memory Initialization
    // ═══════════════════════════════════════════════════════════════════
    let mut heap_size = 0;
    let mut heap_start = u64::MAX;

    // Allocate contiguous physical memory for the kernel heap
    // Note: We try 2GB first, then fall back.
    for &size in &[2 * 1024 * 1024 * 1024, 512 * 1024 * 1024, 128 * 1024 * 1024] {
        let pages = size / 4096;
        if let Ok(ptr) = system_table.boot_services().allocate_pages(
            uefi::table::boot::AllocateType::AnyPages,
            uefi::table::boot::MemoryType::LOADER_DATA,
            pages,
        ) {
            heap_size = size;
            heap_start = ptr;
            break;
        }
    }

    if heap_start == u64::MAX {
        panic!("Failed to allocate contiguous heap memory!");
    }

    // If UEFI returns physical address 0, adjust it to avoid Rust null pointer bugs
    if heap_start == 0 {
        heap_start += 4096;
        heap_size -= 4096;
    }

    unsafe {
        ALLOCATOR.init(heap_start as usize, heap_size);
    }
    info!("[Phase 1] Allocator ready. {} MB heap.", heap_size / 1024 / 1024);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Framebuffer Initialization (GOP)
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 1] Initializing GOP framebuffer...");
    let mut fb = {
        let bt = system_table.boot_services();
        let gop_handle = bt
            .get_handle_for_protocol::<GraphicsOutput>()
            .expect("GOP not found");
        let mut gop = bt
            .open_protocol_exclusive::<GraphicsOutput>(gop_handle)
            .expect("Failed to open GOP");
        let mode_info = gop.current_mode_info();
        let mut raw_fb = gop.frame_buffer();
        let width = mode_info.resolution().0;
        let height = mode_info.resolution().1;
        let stride = mode_info.stride();

        info!("[Phase 1] Framebuffer: {}x{} stride={}", width, height, stride);

        unsafe {
            Framebuffer::new(raw_fb.as_mut_ptr(), raw_fb.size(), stride, width, height)
        }
    };

    fb.clear(framebuffer::colors::BLACK);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Storage — Load weights from secondary drive
    // ═══════════════════════════════════════════════════════════════════
    let weights_data = {
        let bt = system_table.boot_services();
        match BlockDevice::find_weights_disk(bt) {
            Ok(device) => Some(device),
            Err(e) => {
                log::warn!("[Phase 1] No weights disk found: {:?}. Using mock inference.", e);
                None
            }
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Initialize Inference Engine
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 2] Initializing inference engine...");
    let inference = match &weights_data {
        Some(device) => InferenceEngine::init(device.as_bytes()),
        None => InferenceEngine::mock(),
    };
    info!("[Phase 2] Inference engine created!");
    info!("[Phase 2] Inference mode: {:?}", inference.mode);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: Initialize Executive Loop
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 5] Initializing executive loop...");
    let mut executive = ExecutiveLoop::with_inference(inference);
    info!("[Phase 5] Executive loop ready. PID 1 spawned.");

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 6: Initialize Keyboard & Shell
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 6] Initializing keyboard driver...");
    let mut keyboard = Keyboard::new();

    info!("[Phase 6] Initializing Natural Language Shell...");
    let shell_margin = 10;
    let mut shell = Shell::new(
        shell_margin,
        shell_margin,
        fb.width - shell_margin * 2,
        fb.height - shell_margin * 2,
    );

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 7: Initialize Mouse, Compositor, Desktop Agent
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 7] Initializing compositor and desktop agent...");
    let mut _mouse = Mouse::new(fb.width, fb.height);
    let mut _compositor = Compositor::new();
    let mut _desktop_agent = DesktopAgent::new();

    // ═══════════════════════════════════════════════════════════════════
    // BOOT COMPLETE — Enter main event loop
    // ═══════════════════════════════════════════════════════════════════
    info!("════════════════════════════════════════════");
    info!("  BOOT COMPLETE — All phases initialized   ");
    info!("════════════════════════════════════════════");

    shell.print_system("Boot complete. All 8 phases initialized.");
    shell.print_system(&alloc::format!(
        "Inference mode: {:?} | Heap: {} MB | Display: {}x{}",
        executive.inference.mode,
        ALLOCATOR.remaining() / 1024 / 1024,
        fb.width, fb.height,
    ));
    shell.print(&alloc::format!(""));

    // Initial render
    shell.render(&mut fb);

    // Demonstration: run a few ticks with simulated input
    let demo_commands = ["hello", "spawn browser", "status"];
    for cmd in &demo_commands {
        shell.print_colored(
            &alloc::format!("agentic> {}", cmd),
            framebuffer::colors::WHITE,
        );

        executive.submit_request(String::from(*cmd));
        let result = executive.tick();

        // Color based on result type
        if result.starts_with("[ERROR]") {
            shell.print_error(&result);
        } else {
            shell.print_system(&result);
        }

        shell.render(&mut fb);

        // Brief delay for visual effect
        for _ in 0..10_000_000 {
            core::hint::spin_loop();
        }
    }

    shell.print_system("Demo complete. Entering interactive mode...");
    shell.render(&mut fb);

    loop {
        // Poll keyboard
        keyboard.poll(&mut system_table);
        while let Some(event) = keyboard.next_event() {
            if let Some(command) = shell.handle_key(event) {
                shell.print_colored(
                    &alloc::format!("agentic> {}", command),
                    framebuffer::colors::WHITE,
                );

                executive.submit_request(command);
                let result = executive.tick();

                if result.starts_with("[ERROR]") {
                    shell.print_error(&result);
                } else {
                    shell.print_system(&result);
                }
            }
        }

        // Periodic render
        shell.render(&mut fb);

        // Don't burn 100% CPU
        for _ in 0..10_000 {
            core::hint::spin_loop();
        }
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    log::error!("\r\n══ KERNEL PANIC ══\r\n{}", info);
    
    use core::fmt::Write;
    struct SerialWriter;
    impl Write for SerialWriter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for b in s.as_bytes() {
                unsafe {
                    core::arch::asm!(
                        "out dx, al",
                        in("dx") 0x3F8u16,
                        in("al") *b,
                    );
                }
            }
            Ok(())
        }
    }

    let mut writer = SerialWriter;
    let _ = writeln!(writer, "\r\n══ KERNEL PANIC ══");
    let _ = writeln!(writer, "{}", info);
    
    loop {
        core::hint::spin_loop();
    }
}
