//! Phase 6: Keyboard Driver — PS/2 keyboard input via UEFI SimpleTextInput protocol.
//!
//! Provides a non-blocking key event queue for the shell.

use alloc::collections::VecDeque;
use uefi::prelude::*;
use uefi::proto::console::text::Key;

/// A processed key event.
#[derive(Debug, Clone, Copy)]
pub enum KeyEvent {
    /// A printable character was pressed.
    Char(char),
    /// Enter/Return was pressed.
    Enter,
    /// Backspace was pressed.
    Backspace,
    /// Escape was pressed.
    Escape,
    /// Up arrow.
    Up,
    /// Down arrow.
    Down,
    /// Left arrow.
    Left,
    /// Right arrow.
    Right,
    /// Tab key.
    Tab,
}

/// Keyboard driver wrapping UEFI SimpleTextInput.
pub struct Keyboard {
    /// Queue of pending key events.
    event_queue: VecDeque<KeyEvent>,
}

impl Keyboard {
    pub fn new() -> Self {
        Self {
            event_queue: VecDeque::with_capacity(32),
        }
    }

    /// Poll the UEFI console for key events. Non-blocking.
    ///
    /// Call this in the main loop to accumulate key events.
    pub fn poll(&mut self, system_table: &mut SystemTable<Boot>) {
        // Try to read up to 8 keys per poll to avoid missing fast typing
        for _ in 0..8 {
            match system_table.stdin().read_key() {
                Ok(Some(key)) => {
                    if let Some(event) = Self::translate_key(key) {
                        self.event_queue.push_back(event);
                    }
                }
                _ => break, // No more keys available
            }
        }
    }

    /// Translate a UEFI Key into our KeyEvent.
    fn translate_key(key: Key) -> Option<KeyEvent> {
        match key {
            Key::Printable(c) => {
                let ch: char = c.into();
                match ch {
                    '\r' | '\n' => Some(KeyEvent::Enter),
                    '\x08' => Some(KeyEvent::Backspace),
                    '\t' => Some(KeyEvent::Tab),
                    c if c as u32 >= 32 => Some(KeyEvent::Char(c)),
                    _ => None,
                }
            }
            Key::Special(scan) => {
                match scan {
                    uefi::proto::console::text::ScanCode::ESCAPE => Some(KeyEvent::Escape),
                    uefi::proto::console::text::ScanCode::UP => Some(KeyEvent::Up),
                    uefi::proto::console::text::ScanCode::DOWN => Some(KeyEvent::Down),
                    uefi::proto::console::text::ScanCode::LEFT => Some(KeyEvent::Left),
                    uefi::proto::console::text::ScanCode::RIGHT => Some(KeyEvent::Right),
                    _ => None,
                }
            }
        }
    }

    /// Get the next key event from the queue.
    pub fn next_event(&mut self) -> Option<KeyEvent> {
        self.event_queue.pop_front()
    }

    /// Check if there are pending events.
    pub fn has_events(&self) -> bool {
        !self.event_queue.is_empty()
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> alloc::vec::Vec<KeyEvent> {
        self.event_queue.drain(..).collect()
    }
}
