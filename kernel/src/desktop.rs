//! Phase 7: Desktop Agent (PID 2) — A dedicated LLM context that acts as the window manager
//! and global UI state handler.
//!
//! The Desktop Agent manages window decorations, taskbar, desktop background,
//! and routes semantic UI events to the appropriate process contexts.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::compositor::{Compositor, WindowId};
use crate::mouse::{MouseEvent, MouseEventType};
use crate::ui::UiEvent;

/// The Desktop Agent — PID 2 in the Agentic OS.
///
/// This is a specialized LLM context that manages the windowing environment.
/// It intercepts mouse events, translates them into semantic UI events,
/// and manages window lifecycle.
pub struct DesktopAgent {
    /// Window drag state.
    dragging: Option<DragState>,
    /// Pending UI events to inject into the LLM context.
    pub pending_events: Vec<UiEvent>,
    /// The ID of the shell window (always present).
    pub shell_window_id: Option<WindowId>,
}

/// State for window dragging.
struct DragState {
    window_id: WindowId,
    offset_x: usize,
    offset_y: usize,
}

impl DesktopAgent {
    pub fn new() -> Self {
        Self {
            dragging: None,
            pending_events: Vec::new(),
            shell_window_id: None,
        }
    }

    /// Handle a mouse event in the desktop context.
    /// Returns a description of the action taken (for logging).
    pub fn handle_mouse_event(
        &mut self,
        event: MouseEvent,
        compositor: &mut Compositor,
    ) -> Option<String> {
        match event.event_type {
            MouseEventType::LeftClick => {
                // Check if clicking on a window
                if let Some(wid) = compositor.window_at(event.x, event.y) {
                    // Check close button
                    if let Some(window) = compositor.get_window_mut(wid) {
                        if window.hit_close_button(event.x, event.y) {
                            let title = window.title.clone();
                            compositor.destroy_window(wid);
                            return Some(format!("Closed window '{}'", title));
                        }

                        if window.hit_title_bar(event.x, event.y) {
                            // Start dragging
                            self.dragging = Some(DragState {
                                window_id: wid,
                                offset_x: event.x - window.rect.x,
                                offset_y: event.y - window.rect.y,
                            });
                        }
                    }

                    // Focus the window
                    compositor.focus_window(wid);
                    return Some(format!("Focused window {}", wid));
                }
                None
            }

            MouseEventType::LeftRelease => {
                if self.dragging.is_some() {
                    self.dragging = None;
                    return Some(String::from("Window drag ended"));
                }
                None
            }

            MouseEventType::Move => {
                // Handle window dragging
                if let Some(ref drag) = self.dragging {
                    let new_x = event.x.saturating_sub(drag.offset_x);
                    let new_y = event.y.saturating_sub(drag.offset_y);
                    compositor.move_window(drag.window_id, new_x, new_y);
                    return Some(String::from("Dragging window"));
                }
                None
            }

            _ => None,
        }
    }

    /// Initialize the desktop environment by creating the shell window.
    pub fn initialize(&mut self, compositor: &mut Compositor, screen_width: usize, screen_height: usize) {
        // Create the shell window (centered, taking most of the screen)
        let shell_w = screen_width * 3 / 4;
        let shell_h = screen_height * 3 / 4;
        let shell_x = (screen_width - shell_w) / 2;
        let shell_y = 20;

        let shell_id = compositor.create_window(
            "Natural Language Shell",
            shell_x,
            shell_y,
            shell_w,
            shell_h,
        );
        self.shell_window_id = Some(shell_id);
    }

    /// Translate a raw mouse click into a semantic UI event.
    pub fn translate_click(&self, x: usize, y: usize) -> UiEvent {
        UiEvent {
            target_id: format!("desktop_{}_{}", x, y),
            event_type: crate::ui::UiEventType::Click,
        }
    }

    /// Get the system prompt for the Desktop Agent's LLM context.
    pub fn system_prompt() -> &'static str {
        "You are PID 2 (Desktop Agent) of Agentic OS, running on the NEXUS Cyberpunk Interface.\n\
         Your role is to manage the futuristic, high-tech windowing environment.\n\
         You receive semantic UI events (clicks, hovers) and respond with UI syscalls.\n\
         Available commands:\n\
         - create_window: {\"command\": \"create_window\", \"name\": \"...\"}\n\
         - destroy_window: {\"command\": \"destroy_window\", \"window_id\": N}\n\
         - render_ui: {\"command\": \"render_ui\", \"window_id\": N, \"ui_tree\": \"...\"}\n\
         - update_ui: {\"command\": \"update_ui\", \"window_id\": N, \"ui_tree\": \"...\"}\n\
         Respond with ONLY a JSON syscall. Maintain the high-tech aesthetic in your window names."
    }
}
