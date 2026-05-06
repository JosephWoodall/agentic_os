//! Phase 6: Natural Language Shell — PID 1 user interface rendered on the GOP framebuffer.
//!
//! Captures keyboard input, displays responses, and routes user text to the Executive Loop.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::framebuffer::{Framebuffer, colors, CHAR_WIDTH, LINE_HEIGHT};
use crate::keyboard::KeyEvent;

/// Maximum line length for input.
const MAX_INPUT_LEN: usize = 120;
/// Maximum scrollback lines.
const MAX_SCROLLBACK: usize = 200;
/// Shell prompt string.
const PROMPT: &str = "agentic> ";

/// A line in the shell output with its color.
#[derive(Debug, Clone)]
struct OutputLine {
    text: String,
    color: u32,
}

/// The interactive Natural Language Shell.
pub struct Shell {
    /// Current input buffer.
    input_buffer: String,
    /// Cursor position within the input buffer.
    cursor_pos: usize,
    /// Scrollback buffer of output lines.
    output_lines: Vec<OutputLine>,
    /// First visible line index (for scrolling).
    scroll_offset: usize,
    /// Number of visible lines in the shell area.
    visible_lines: usize,
    /// Command history.
    history: Vec<String>,
    /// Current position in command history (for up/down navigation).
    history_pos: usize,
    /// X position where the shell content starts (pixels).
    pub x_offset: usize,
    /// Y position where the shell content starts (pixels).
    pub y_offset: usize,
    /// Width of the shell area in pixels.
    pub width: usize,
    /// Height of the shell area in pixels.
    pub height: usize,
    /// Whether the shell needs a redraw.
    pub dirty: bool,
}

impl Shell {
    /// Create a new shell that fills the given region.
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        let visible_lines = (height / LINE_HEIGHT).saturating_sub(3); // Reserve space for header + input

        let mut shell = Self {
            input_buffer: String::new(),
            cursor_pos: 0,
            output_lines: Vec::with_capacity(MAX_SCROLLBACK),
            scroll_offset: 0,
            visible_lines,
            history: Vec::new(),
            history_pos: 0,
            x_offset: x,
            y_offset: y,
            width,
            height,
            dirty: true,
        };

        // Welcome message
        shell.print_colored("╔══════════════════════════════════════════════════════════╗", colors::ACCENT_BLUE);
        shell.print_colored("║     AGENTIC OS — Probabilistic State-Space Kernel       ║", colors::ACCENT_BLUE);
        shell.print_colored("║     Natural Language Shell v0.1                         ║", colors::ACCENT_BLUE);
        shell.print_colored("╚══════════════════════════════════════════════════════════╝", colors::ACCENT_BLUE);
        shell.print_colored("", colors::WHITE);
        shell.print_colored("Type natural language commands. The LLM executive will translate", colors::DIM_WHITE);
        shell.print_colored("your intent into system calls. Try: 'spawn a new task'", colors::DIM_WHITE);
        shell.print_colored("", colors::WHITE);

        shell
    }

    /// Process a key event. Returns Some(command) if Enter was pressed.
    pub fn handle_key(&mut self, event: KeyEvent) -> Option<String> {
        self.dirty = true;

        match event {
            KeyEvent::Char(c) => {
                if self.input_buffer.len() < MAX_INPUT_LEN {
                    self.input_buffer.insert(self.cursor_pos, c);
                    self.cursor_pos += 1;
                }
                None
            }
            KeyEvent::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input_buffer.remove(self.cursor_pos);
                }
                None
            }
            KeyEvent::Enter => {
                let command = self.input_buffer.clone();
                if !command.is_empty() {
                    // Echo the command to output
                    self.print_colored(
                        &format!("{}{}", PROMPT, command),
                        colors::WHITE,
                    );
                    // Add to history
                    self.history.push(command.clone());
                    self.history_pos = self.history.len();
                    // Clear input
                    self.input_buffer.clear();
                    self.cursor_pos = 0;
                    Some(command)
                } else {
                    None
                }
            }
            KeyEvent::Up => {
                if self.history_pos > 0 {
                    self.history_pos -= 1;
                    self.input_buffer = self.history[self.history_pos].clone();
                    self.cursor_pos = self.input_buffer.len();
                }
                None
            }
            KeyEvent::Down => {
                if self.history_pos < self.history.len() {
                    self.history_pos += 1;
                    if self.history_pos < self.history.len() {
                        self.input_buffer = self.history[self.history_pos].clone();
                    } else {
                        self.input_buffer.clear();
                    }
                    self.cursor_pos = self.input_buffer.len();
                }
                None
            }
            KeyEvent::Tab => {
                // Simple tab completion hint
                self.print_colored(
                    "Commands: spawn, kill, read, write, status, compress, hydrate, help",
                    colors::DIM_WHITE,
                );
                None
            }
            KeyEvent::Escape => {
                self.input_buffer.clear();
                self.cursor_pos = 0;
                None
            }
        }
    }

    /// Print a line to the shell output.
    pub fn print(&mut self, text: &str) {
        self.print_colored(text, colors::GREEN);
    }

    /// Print a line with a specific color.
    pub fn print_colored(&mut self, text: &str, color: u32) {
        self.dirty = true;

        // Handle long lines by wrapping
        let chars_per_line = self.width / CHAR_WIDTH;
        if chars_per_line == 0 {
            return;
        }

        if text.len() <= chars_per_line {
            self.output_lines.push(OutputLine {
                text: String::from(text),
                color,
            });
        } else {
            // Word wrap
            let mut remaining = text;
            while !remaining.is_empty() {
                let mut split_at = remaining.len().min(chars_per_line);
                while split_at > 0 && !remaining.is_char_boundary(split_at) {
                    split_at -= 1;
                }
                if split_at == 0 {
                    // If a single character is too wide, just take the first char
                    split_at = remaining.chars().next().unwrap().len_utf8();
                }
                self.output_lines.push(OutputLine {
                    text: String::from(&remaining[..split_at]),
                    color,
                });
                remaining = &remaining[split_at..];
            }
        }

        // Trim scrollback
        while self.output_lines.len() > MAX_SCROLLBACK {
            self.output_lines.remove(0);
        }

        // Auto-scroll to bottom
        if self.output_lines.len() > self.visible_lines {
            self.scroll_offset = self.output_lines.len() - self.visible_lines;
        }
    }

    /// Print a system response (cyan).
    pub fn print_system(&mut self, text: &str) {
        self.print_colored(text, colors::CYAN);
    }

    /// Print an error (red).
    pub fn print_error(&mut self, text: &str) {
        self.print_colored(text, colors::RED);
    }

    /// Render the shell to the framebuffer.
    pub fn render(&mut self, fb: &mut Framebuffer) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        let x = self.x_offset;
        let y = self.y_offset;

        // Clear shell area
        fb.fill_rect(x, y, self.width, self.height, colors::BLACK);

        // Draw border
        fb.draw_rect(x, y, self.width, self.height, colors::DARK_GRAY);

        // Draw title bar
        fb.fill_rect(x + 1, y + 1, self.width - 2, LINE_HEIGHT + 4, colors::TITLE_BAR);
        fb.draw_string(
            x + 8, y + 3,
            "Agentic OS - Natural Language Shell",
            colors::WHITE,
        );

        // Draw output lines
        let content_y = y + LINE_HEIGHT + 8;
        let end_idx = self.output_lines.len().min(self.scroll_offset + self.visible_lines);

        for (i, line_idx) in (self.scroll_offset..end_idx).enumerate() {
            let line = &self.output_lines[line_idx];
            fb.draw_string(
                x + 8,
                content_y + i * LINE_HEIGHT,
                &line.text,
                line.color,
            );
        }

        // Draw input line
        let input_y = y + self.height - LINE_HEIGHT - 8;
        fb.fill_rect(x + 1, input_y - 2, self.width - 2, LINE_HEIGHT + 4, colors::DARK_GRAY);
        fb.draw_string(x + 8, input_y, PROMPT, colors::GREEN);

        let prompt_width = PROMPT.len() * CHAR_WIDTH;
        fb.draw_string(
            x + 8 + prompt_width,
            input_y,
            &self.input_buffer,
            colors::WHITE,
        );

        // Draw cursor
        let cursor_x = x + 8 + prompt_width + self.cursor_pos * CHAR_WIDTH;
        fb.fill_rect(cursor_x, input_y, 2, LINE_HEIGHT - 2, colors::WHITE);

        // Draw scroll indicator if needed
        if self.output_lines.len() > self.visible_lines {
            let indicator_height = (self.visible_lines as f32 / self.output_lines.len() as f32
                * (self.height - LINE_HEIGHT - 20) as f32) as usize;
            let indicator_y = content_y
                + (self.scroll_offset as f32 / self.output_lines.len() as f32
                    * (self.height - LINE_HEIGHT - 20) as f32) as usize;
            fb.fill_rect(
                x + self.width - 4,
                indicator_y,
                3,
                indicator_height.max(8),
                colors::MID_GRAY,
            );
        }
    }

    /// Get the current input for display purposes.
    pub fn current_input(&self) -> &str {
        &self.input_buffer
    }
}
