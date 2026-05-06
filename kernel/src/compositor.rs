//! Phase 7: Compositor — Z-indexed window manager that renders declarative UI states.
//!
//! Uses dirty-rectangle tracking to minimize redraws on the framebuffer.

use alloc::string::String;
use alloc::vec::Vec;
use crate::framebuffer::{Framebuffer, colors, CHAR_WIDTH, LINE_HEIGHT};

/// A unique window identifier.
pub type WindowId = u32;

/// A rectangular region.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// Window state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowState {
    Normal,
    Minimized,
    Maximized,
}

/// A window managed by the compositor.
pub struct Window {
    pub id: WindowId,
    pub title: String,
    pub rect: Rect,
    pub z_index: usize,
    pub state: WindowState,
    pub focused: bool,
    pub dirty: bool,
    /// Content lines to display in the window body.
    pub content: Vec<String>,
    /// Background color.
    pub bg_color: u32,
    /// Border color.
    pub border_color: u32,
}

impl Window {
    pub fn new(id: WindowId, title: &str, rect: Rect) -> Self {
        Self {
            id,
            title: String::from(title),
            rect,
            z_index: 0,
            state: WindowState::Normal,
            focused: false,
            dirty: true,
            content: Vec::new(),
            bg_color: 0x00111111,       // Dark background
            border_color: colors::MID_GRAY,
        }
    }

    /// Render this window to the framebuffer.
    pub fn render(&mut self, fb: &mut Framebuffer) {
        if self.state == WindowState::Minimized || !self.dirty {
            return;
        }
        self.dirty = false;

        let r = &self.rect;

        // Window background
        fb.fill_rect(r.x, r.y, r.width, r.height, self.bg_color);

        // Border (highlighted if focused)
        let border_color = if self.focused {
            colors::ACCENT_BLUE
        } else {
            self.border_color
        };
        fb.draw_rect(r.x, r.y, r.width, r.height, border_color);

        // Title bar
        let title_bar_height = LINE_HEIGHT + 6;
        let title_bar_color = if self.focused {
            colors::TITLE_BAR
        } else {
            colors::DARK_GRAY
        };
        fb.fill_rect(r.x + 1, r.y + 1, r.width - 2, title_bar_height, title_bar_color);
        fb.draw_string(
            r.x + 8,
            r.y + 3,
            &self.title,
            colors::WHITE,
        );

        // Close button [X]
        let close_x = r.x + r.width - 20;
        fb.fill_rect(close_x, r.y + 2, 16, title_bar_height - 2, colors::RED);
        fb.draw_char(close_x + 4, r.y + 3, 'X', colors::WHITE);

        // Content area
        let content_y = r.y + title_bar_height + 4;
        let max_lines = (r.height - title_bar_height - 8) / LINE_HEIGHT;
        let start_line = if self.content.len() > max_lines {
            self.content.len() - max_lines
        } else {
            0
        };

        for (i, line) in self.content[start_line..].iter().enumerate() {
            if i >= max_lines {
                break;
            }
            fb.draw_string(
                r.x + 8,
                content_y + i * LINE_HEIGHT,
                line,
                colors::DIM_WHITE,
            );
        }
    }

    /// Add content to the window.
    pub fn push_content(&mut self, text: &str) {
        self.content.push(String::from(text));
        self.dirty = true;
        // Keep content bounded
        while self.content.len() > 100 {
            self.content.remove(0);
        }
    }

    /// Check if a point is in the title bar (for dragging).
    pub fn hit_title_bar(&self, x: usize, y: usize) -> bool {
        let title_bar_height = LINE_HEIGHT + 6;
        x >= self.rect.x
            && x < self.rect.x + self.rect.width
            && y >= self.rect.y
            && y < self.rect.y + title_bar_height
    }

    /// Check if a point is on the close button.
    pub fn hit_close_button(&self, x: usize, y: usize) -> bool {
        let close_x = self.rect.x + self.rect.width - 20;
        let title_bar_height = LINE_HEIGHT + 6;
        x >= close_x
            && x < close_x + 16
            && y >= self.rect.y + 2
            && y < self.rect.y + title_bar_height
    }
}

/// The compositor manages all windows and handles rendering order.
pub struct Compositor {
    /// All windows, ordered by z-index (back to front).
    pub windows: Vec<Window>,
    /// Next window ID to assign.
    next_id: WindowId,
    /// The currently focused window ID.
    pub focused_id: Option<WindowId>,
    /// Whether the entire screen needs a redraw.
    pub full_redraw: bool,
    /// Desktop background color.
    pub bg_color: u32,
}

impl Compositor {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
            next_id: 1,
            focused_id: None,
            full_redraw: true,
            bg_color: 0x00221122, // Deep purple-ish dark background
        }
    }

    /// Create a new window and return its ID.
    pub fn create_window(&mut self, title: &str, x: usize, y: usize, width: usize, height: usize) -> WindowId {
        let id = self.next_id;
        self.next_id += 1;

        let mut window = Window::new(id, title, Rect {
            x,
            y,
            width,
            height,
        });
        window.z_index = self.windows.len();
        window.focused = true;

        // Unfocus all other windows
        for w in self.windows.iter_mut() {
            w.focused = false;
            w.dirty = true;
        }

        self.windows.push(window);
        self.focused_id = Some(id);
        self.full_redraw = true;

        id
    }

    /// Destroy a window by ID.
    pub fn destroy_window(&mut self, id: WindowId) {
        self.windows.retain(|w| w.id != id);
        if self.focused_id == Some(id) {
            self.focused_id = self.windows.last().map(|w| w.id);
            if let Some(fid) = self.focused_id {
                if let Some(w) = self.windows.iter_mut().find(|w| w.id == fid) {
                    w.focused = true;
                    w.dirty = true;
                }
            }
        }
        self.full_redraw = true;
    }

    /// Focus a window, bringing it to the front.
    pub fn focus_window(&mut self, id: WindowId) {
        for w in self.windows.iter_mut() {
            w.focused = w.id == id;
            w.dirty = true;
        }
        self.focused_id = Some(id);

        // Move to end of vec (highest z-index)
        if let Some(pos) = self.windows.iter().position(|w| w.id == id) {
            let window = self.windows.remove(pos);
            self.windows.push(window);
        }

        // Update z-indices
        for (i, w) in self.windows.iter_mut().enumerate() {
            w.z_index = i;
        }

        self.full_redraw = true;
    }

    /// Move a window to a new position.
    pub fn move_window(&mut self, id: WindowId, x: usize, y: usize) {
        if let Some(w) = self.windows.iter_mut().find(|w| w.id == id) {
            w.rect.x = x;
            w.rect.y = y;
            w.dirty = true;
            self.full_redraw = true;
        }
    }

    /// Find the topmost window at a given point.
    pub fn window_at(&self, x: usize, y: usize) -> Option<WindowId> {
        // Search back-to-front (highest z-index first)
        for w in self.windows.iter().rev() {
            if w.state != WindowState::Minimized && w.rect.contains(x, y) {
                return Some(w.id);
            }
        }
        None
    }

    /// Get a mutable reference to a window by ID.
    pub fn get_window_mut(&mut self, id: WindowId) -> Option<&mut Window> {
        self.windows.iter_mut().find(|w| w.id == id)
    }

    /// Render all windows to the framebuffer.
    pub fn render(&mut self, fb: &mut Framebuffer) {
        if self.full_redraw {
            // Clear background
            fb.clear(self.bg_color);

            // Draw desktop grid pattern
            for y in (0..fb.height).step_by(32) {
                for x in (0..fb.width).step_by(32) {
                    fb.set_pixel(x, y, 0x00332233);
                }
            }

            self.full_redraw = false;
        }

        // Render windows in z-order (back to front)
        for window in self.windows.iter_mut() {
            window.render(fb);
        }
    }

    /// Render a taskbar at the bottom of the screen.
    pub fn render_taskbar(&self, fb: &mut Framebuffer) {
        let taskbar_h = LINE_HEIGHT + 8;
        let taskbar_y = fb.height - taskbar_h;

        // Taskbar background
        fb.fill_rect(0, taskbar_y, fb.width, taskbar_h, 0x00222222);
        fb.fill_rect(0, taskbar_y, fb.width, 1, colors::MID_GRAY);

        // "Start" button
        fb.fill_rect(4, taskbar_y + 2, 60, taskbar_h - 4, colors::TITLE_BAR);
        fb.draw_string(10, taskbar_y + 4, "Agentic", colors::WHITE);

        // Window buttons
        let mut btn_x = 70;
        for w in &self.windows {
            let btn_color = if Some(w.id) == self.focused_id {
                colors::ACCENT_BLUE
            } else {
                colors::DARK_GRAY
            };
            let title_chars = w.title.len().min(12);
            let btn_width = title_chars * CHAR_WIDTH + 16;

            fb.fill_rect(btn_x, taskbar_y + 2, btn_width, taskbar_h - 4, btn_color);
            fb.draw_string(
                btn_x + 8,
                taskbar_y + 4,
                &w.title[..title_chars],
                colors::WHITE,
            );

            btn_x += btn_width + 4;
        }
    }
}
