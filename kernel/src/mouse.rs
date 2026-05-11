//! Phase 7: Mouse Driver — UEFI SimplePointer support with Keyboard Fallback.
//!
//! Tracks absolute X/Y position and renders a hardware cursor independently of LLM ticks.

use crate::framebuffer::Framebuffer;
use uefi::proto::console::pointer::Pointer;

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
    pub x: usize,
    pub y: usize,
    pub buttons: MouseButtons,
    prev_buttons: MouseButtons,
    max_x: usize,
    max_y: usize,
    sensitivity: f32,
}

impl Mouse {
    pub fn new(screen_width: usize, screen_height: usize) -> Self {
        Self {
            x: screen_width / 2,
            y: screen_height / 2,
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
            sensitivity: 6.0, // Extra high sensitivity
        }
    }

    pub fn initialize(&self, system_table: &mut uefi::prelude::SystemTable<uefi::prelude::Boot>) {
        let bt = system_table.boot_services();
        if let Ok(handle) = bt.get_handle_for_protocol::<Pointer>() {
            if let Ok(mut pointer) = bt.open_protocol_exclusive::<Pointer>(handle) {
                let _ = pointer.reset(false);
                log::info!("[Mouse] SimplePointer protocol reset.");
            }
        }
    }

    pub fn update(&mut self, dx: i32, dy: i32, left: bool, right: bool) -> Option<MouseEvent> {
        if dx == 0 && dy == 0 && left == self.buttons.left && right == self.buttons.right {
            return None;
        }

        self.prev_buttons = self.buttons;

        let scaled_dx = (dx as f32 * self.sensitivity) as i32;
        let scaled_dy = (dy as f32 * self.sensitivity) as i32;

        let new_x = self.x as i32 + scaled_dx;
        let new_y = self.y as i32 + scaled_dy;

        self.x = new_x.max(0).min(self.max_x as i32) as usize;
        self.y = new_y.max(0).min(self.max_y as i32) as usize;

        self.buttons = MouseButtons { left, right };

        let event_type = if left && !self.prev_buttons.left {
            MouseEventType::LeftClick
        } else if !left && self.prev_buttons.left {
            MouseEventType::LeftRelease
        } else if right && !self.prev_buttons.right {
            MouseEventType::RightClick
        } else if !right && self.prev_buttons.right {
            MouseEventType::RightRelease
        } else {
            MouseEventType::Move
        };

        Some(MouseEvent {
            x: self.x,
            y: self.y,
            buttons: self.buttons,
            event_type,
        })
    }

    pub fn poll(&mut self, system_table: &mut uefi::prelude::SystemTable<uefi::prelude::Boot>) -> Option<MouseEvent> {
        let bt = system_table.boot_services();

        if let Ok(handle) = bt.get_handle_for_protocol::<Pointer>() {
            if let Ok(mut pointer) = bt.open_protocol_exclusive::<Pointer>(handle) {
                if let Ok(Some(state)) = pointer.read_state() {
                    return self.update(
                        state.relative_movement[0],
                        state.relative_movement[1],
                        state.button[0],
                        state.button[1],
                    );
                }
            }
        }
        None
    }

    /// Handle keyboard as mouse movement fallback.
    pub fn handle_key_fallback(&mut self, event: crate::keyboard::KeyEvent) -> Option<MouseEvent> {
        use crate::keyboard::KeyEvent;
        let mut dx = 0;
        let mut dy = 0;
        let mut left = self.buttons.left;

        match event {
            KeyEvent::Up => dy = -20,
            KeyEvent::Down => dy = 20,
            KeyEvent::Left => dx = -20,
            KeyEvent::Right => dx = 20,
            KeyEvent::Char(' ') => left = !left,
            _ => return None,
        }

        self.update(dx, dy, left, self.buttons.right)
    }

    pub fn render_cursor(&mut self, fb: &mut Framebuffer) {
        Self::draw_cursor_shape(fb, self.x, self.y);
    }

    fn draw_cursor_shape(fb: &mut Framebuffer, x: usize, y: usize) {
        let cursor_data: &[&[u8]] = &[
            &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1], &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1, 1], &[1, 1, 1, 0, 1, 1], &[1, 1, 0, 0, 0, 1, 1],
            &[1, 0, 0, 0, 0, 1, 1], &[0, 0, 0, 0, 0, 0, 1, 1], &[0, 0, 0, 0, 0, 0, 1],
        ];

        use crate::framebuffer::colors;
        for (dy, row) in cursor_data.iter().enumerate() {
            for (dx, &pixel) in row.iter().enumerate() {
                if pixel == 1 {
                    fb.set_pixel_alpha(x + dx + 1, y + dy + 1, colors::NEON_MAGENTA, 180);
                    fb.set_pixel(x + dx, y + dy, colors::NEON_CYAN);
                }
            }
        }
    }
}
