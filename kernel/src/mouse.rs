//! Phase 7: Mouse Driver — UEFI SimplePointer protocol for cursor input.
//!
//! Tracks absolute X/Y position and renders a hardware cursor independently of LLM ticks.

use crate::framebuffer::Framebuffer;

/// Mouse button state.
#[derive(Debug, Clone, Copy)]
pub struct MouseButtons {
    pub left: bool,
    pub right: bool,
}

/// A mouse event.
#[derive(Debug, Clone, Copy)]
pub struct MouseEvent {
    pub x: usize,
    pub y: usize,
    pub buttons: MouseButtons,
    pub event_type: MouseEventType,
}

/// Type of mouse event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseEventType {
    Move,
    LeftClick,
    LeftRelease,
    RightClick,
    RightRelease,
}

/// Mouse driver with absolute position tracking and cursor rendering.
pub struct Mouse {
    /// Current X position.
    pub x: usize,
    /// Current Y position.
    pub y: usize,
    /// Previous X position (for erasing old cursor).
    prev_x: usize,
    /// Previous Y position.
    prev_y: usize,
    /// Button state.
    pub buttons: MouseButtons,
    /// Previous button state (for detecting clicks).
    prev_buttons: MouseButtons,
    /// Screen width bound.
    max_x: usize,
    /// Screen height bound.
    max_y: usize,
    /// Whether the cursor has moved and needs redraw.
    pub cursor_dirty: bool,
    /// Sensitivity multiplier.
    sensitivity: f32,
}

impl Mouse {
    pub fn new(screen_width: usize, screen_height: usize) -> Self {
        Self {
            x: screen_width / 2,
            y: screen_height / 2,
            prev_x: screen_width / 2,
            prev_y: screen_height / 2,
            buttons: MouseButtons {
                left: false,
                right: false,
            },
            prev_buttons: MouseButtons {
                left: false,
                right: false,
            },
            max_x: screen_width.saturating_sub(1),
            max_y: screen_height.saturating_sub(1),
            cursor_dirty: true,
            sensitivity: 1.0,
        }
    }

    /// Update mouse state from UEFI SimplePointer data.
    /// `dx`, `dy` are relative movement values from the pointer protocol.
    pub fn update(&mut self, dx: i32, dy: i32, left: bool, right: bool) -> Option<MouseEvent> {
        self.prev_x = self.x;
        self.prev_y = self.y;
        self.prev_buttons = self.buttons;

        // Apply movement with sensitivity
        let new_x = self.x as i32 + (dx as f32 * self.sensitivity) as i32;
        let new_y = self.y as i32 + (dy as f32 * self.sensitivity) as i32;

        self.x = new_x.max(0).min(self.max_x as i32) as usize;
        self.y = new_y.max(0).min(self.max_y as i32) as usize;

        self.buttons = MouseButtons { left, right };

        if self.x != self.prev_x || self.y != self.prev_y {
            self.cursor_dirty = true;
        }

        // Determine event type
        let event_type = if left && !self.prev_buttons.left {
            MouseEventType::LeftClick
        } else if !left && self.prev_buttons.left {
            MouseEventType::LeftRelease
        } else if right && !self.prev_buttons.right {
            MouseEventType::RightClick
        } else if !right && self.prev_buttons.right {
            MouseEventType::RightRelease
        } else if self.x != self.prev_x || self.y != self.prev_y {
            MouseEventType::Move
        } else {
            return None;
        };

        Some(MouseEvent {
            x: self.x,
            y: self.y,
            buttons: self.buttons,
            event_type,
        })
    }

    /// Render the mouse cursor using XOR overlay (flicker-free, independent of LLM).
    ///
    /// Call this AFTER all other rendering to overlay the cursor.
    pub fn render_cursor(&mut self, fb: &mut Framebuffer) {
        if !self.cursor_dirty {
            return;
        }
        self.cursor_dirty = false;

        // Erase old cursor (XOR again to restore)
        Self::draw_cursor_shape(fb, self.prev_x, self.prev_y);

        // Draw new cursor
        Self::draw_cursor_shape(fb, self.x, self.y);
    }

    /// Draw an arrow cursor shape using XOR.
    fn draw_cursor_shape(fb: &mut Framebuffer, x: usize, y: usize) {
        // Simple arrow cursor (12x16 pixels)
        let cursor_data: &[&[u8]] = &[
            &[1],
            &[1, 1],
            &[1, 1, 1],
            &[1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 0, 1, 1],
            &[1, 1, 0, 0, 0, 1, 1],
            &[1, 0, 0, 0, 0, 1, 1],
            &[0, 0, 0, 0, 0, 0, 1, 1],
            &[0, 0, 0, 0, 0, 0, 1],
        ];

        for (dy, row) in cursor_data.iter().enumerate() {
            for (dx, &pixel) in row.iter().enumerate() {
                if pixel == 1 {
                    fb.xor_rect(x + dx, y + dy, 1, 1);
                }
            }
        }
    }
}
