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
mod serial;

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
    crate::serial::init();
    
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
    let inference = InferenceEngine::serial();
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
    let shell_width = (fb.width / 2) - shell_margin * 2;
    let mut shell = Shell::new(
        fb.width / 2 + shell_margin, // X offset starts at halfway point
        shell_margin,
        shell_width,
        fb.height - shell_margin * 2,
    );

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 7: Initialize Mouse, Compositor, Desktop Agent
    // ═══════════════════════════════════════════════════════════════════
    info!("[Phase 7] Initializing compositor and desktop agent...");
    let mut mouse = Mouse::new(fb.width, fb.height);
    mouse.initialize(&mut system_table); // Initialize hardware
    let mut compositor = Compositor::new();
    let mut desktop_agent = DesktopAgent::new();

    // Initialize desktop environment (creates shell window, etc.)
    desktop_agent.initialize(&mut compositor, fb.width, fb.height);

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
    fb.clear(framebuffer::colors::BG_DEEP); // Use the new deep navy background
    compositor.full_redraw = true;
    compositor.render(&mut fb);
    compositor.render_taskbar(&mut fb);
    shell.render(&mut fb);
    mouse.render_cursor(&mut fb);

    let mut tick_timer = 0;

    loop {
        let mut ui_dirty = false;
        let mut should_tick = false;

        // Poll keyboard
        keyboard.poll(&mut system_table);
        while let Some(event) = keyboard.next_event() {
            // Check if it's a mouse control key (arrows/space)
            if let Some(mouse_event) = mouse.handle_key_fallback(event) {
                if let Some(action) = desktop_agent.handle_mouse_event(mouse_event, &mut compositor) {
                    log::info!("Desktop action: {}", action);
                }
            } else if let Some(command) = shell.handle_key(event) {
                executive.submit_request(command);
                should_tick = true;
            }
            ui_dirty = true;
        }

        // Poll mouse
        if let Some(mouse_event) = mouse.poll(&mut system_table) {
            if let Some(action) = desktop_agent.handle_mouse_event(mouse_event, &mut compositor) {
                log::info!("Desktop action: {}", action);
            }
            ui_dirty = true;
        }

        tick_timer += 1;
        // Trigger a background tick every ~5 million spin loops if idle
        if tick_timer > 5_000_000 {
            should_tick = true;
            tick_timer = 0;
        }

        if should_tick {
            let result = executive.tick(&mut compositor);
            if result.starts_with("[ERROR]") {
                shell.print_error(&result);
            } else if result.starts_with("[OK] ") {
                shell.print_system(&result[5..]);
            } else if result.starts_with("Process yielded") {
                // Ignore
            } else {
                shell.print_system(&result);
            }
            ui_dirty = true;
        }

        // Handle shell or compositor dirty state
        if shell.dirty || compositor.full_redraw || ui_dirty {
            // For high-tech Cyberpunk UI, we prefer full redraws for consistent effects
            compositor.full_redraw = true;
            compositor.render(&mut fb);
            compositor.render_taskbar(&mut fb);
            shell.render(&mut fb);
        }

        // Mouse cursor is rendered independently at the end to keep it on top
        mouse.render_cursor(&mut fb);

        // Don't burn 100% CPU
        for _ in 0..10_000 {
            core::hint::spin_loop();
        }
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
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
