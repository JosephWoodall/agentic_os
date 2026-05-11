//! Phase 1: Framebuffer — GOP-based rendering with text, rectangles, and cursor support.

use font8x8::UnicodeFonts;

/// Color constants for the Agentic OS visual identity (Format: 0x00BBGGRR).
pub mod colors {
    pub const BLACK: u32 = 0x00000000;
    pub const WHITE: u32 = 0x00FFFFFF;
    
    // Futuristic UI Color Scheme (0x00BBGGRR)
    pub const BG_DEEP: u32 = 0x001F0A0A;       // Deep Navy (#0A0A1F)
    pub const PANEL_BG: u32 = 0x002B1212;      // Secondary BG (#12122B)
    pub const GLASS_BG: u32 = 0x00402020;      // Slightly lighter for glass effect
    
    pub const NEON_CYAN: u32 = 0x00FFF500;     // Primary Cyan (#00F5FF)
    pub const NEON_MAGENTA: u32 = 0x00AA00FF;  // Accent Magenta (#FF00AA)
    pub const NEON_GREEN: u32 = 0x004DFF7B;    // Highlight Neon Green (#7BFF4D)
    
    pub const TITLE_BAR: u32 = 0x003A1A1A;     // Slightly lighter navy
    pub const BORDER: u32 = 0x004DFF7B;        // Default border is Neon Green
    pub const BORDER_FOCUS: u32 = 0x00FFF500;  // Cyan for focused windows
    
    pub const TEXT_BRIGHT: u32 = 0x00FFF8E0;   // Text (#E0F8FF)
    pub const TEXT_MUTED: u32 = 0x00AA9999;
    
    // Legacy support aliases
    pub const RED: u32 = 0x002222EE;
    pub const GREEN: u32 = 0x004DFF7B;
    pub const CYAN: u32 = 0x00FFF500;
    pub const YELLOW: u32 = 0x0000DDFF;
    pub const DIM_WHITE: u32 = 0x00FFF8E0;
    pub const DARK_GRAY: u32 = 0x00333333;
    pub const MID_GRAY: u32 = 0x00666666;
    pub const ACCENT_BLUE: u32 = 0x00FFF500;
}

/// Character dimensions for font8x8 glyphs.
pub const CHAR_WIDTH: usize = 8;
pub const CHAR_HEIGHT: usize = 8;
pub const LINE_SPACING: usize = 2;
pub const LINE_HEIGHT: usize = CHAR_HEIGHT + LINE_SPACING;

/// Represents a pixel-addressable framebuffer backed by GOP.
pub struct Framebuffer {
    /// Raw pointer to the framebuffer memory (owned by UEFI, not us).
    base: *mut u8,
    /// Total size of the framebuffer in bytes.
    size: usize,
    /// Horizontal stride in pixels.
    pub stride: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

// Safety: The framebuffer pointer is only accessed from the main boot thread.
unsafe impl Send for Framebuffer {}
unsafe impl Sync for Framebuffer {}

impl Framebuffer {
    /// Create a new Framebuffer wrapper.
    ///
    /// # Safety
    /// `base` must point to valid framebuffer memory of at least `size` bytes.
    pub unsafe fn new(base: *mut u8, size: usize, stride: usize, width: usize, height: usize) -> Self {
        Self {
            base,
            size,
            stride,
            width,
            height,
        }
    }

    /// Get a mutable slice over the entire framebuffer.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.base, self.size) }
    }

    /// Clear the entire screen to a solid color.
    pub fn clear(&mut self, color: u32) {
        let fb = self.as_mut_slice();
        let r = (color & 0xFF) as u8;
        let g = ((color >> 8) & 0xFF) as u8;
        let b = ((color >> 16) & 0xFF) as u8;

        for i in 0..(fb.len() / 4) {
            fb[i * 4] = r;
            fb[i * 4 + 1] = g;
            fb[i * 4 + 2] = b;
            fb[i * 4 + 3] = 0;
        }
    }

    /// Set a pixel with alpha blending (0-255).
    pub fn set_pixel_alpha(&mut self, x: usize, y: usize, color: u32, alpha: u8) {
        if x >= self.width || y >= self.height || alpha == 0 {
            return;
        }
        if alpha == 255 {
            self.set_pixel(x, y, color);
            return;
        }

        let offset = (y * self.stride + x) * 4;
        let fb = self.as_mut_slice();
        if offset + 3 < fb.len() {
            let src_r = (color & 0xFF) as u16;
            let src_g = ((color >> 8) & 0xFF) as u16;
            let src_b = ((color >> 16) & 0xFF) as u16;

            let dst_r = fb[offset] as u16;
            let dst_g = fb[offset + 1] as u16;
            let dst_b = fb[offset + 2] as u16;

            let a = alpha as u16;
            let inv_a = 255 - a;

            fb[offset] = ((src_r * a + dst_r * inv_a) / 255) as u8;
            fb[offset + 1] = ((src_g * a + dst_g * inv_a) / 255) as u8;
            fb[offset + 2] = ((src_b * a + dst_b * inv_a) / 255) as u8;
        }
    }

    /// Draw a filled rectangle with alpha blending.
    pub fn fill_rect_alpha(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32, alpha: u8) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel_alpha(x + dx, y + dy, color, alpha);
            }
        }
    }

    /// Draw a vertical gradient.
    pub fn fill_gradient_v(&mut self, x: usize, y: usize, w: usize, h: usize, color_top: u32, color_bottom: u32) {
        if h == 0 { return; }
        let tr = (color_top & 0xFF) as i32;
        let tg = ((color_top >> 8) & 0xFF) as i32;
        let tb = ((color_top >> 16) & 0xFF) as i32;

        let br = (color_bottom & 0xFF) as i32;
        let bg = ((color_bottom >> 8) & 0xFF) as i32;
        let bb = ((color_bottom >> 16) & 0xFF) as i32;

        for dy in 0..h {
            let r = tr + (br - tr) * dy as i32 / h as i32;
            let g = tg + (bg - tg) * dy as i32 / h as i32;
            let b = tb + (bb - tb) * dy as i32 / h as i32;
            let color = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16);
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    /// Draw a rounded rectangle.
    pub fn fill_rounded_rect(&mut self, x: usize, y: usize, w: usize, h: usize, r: usize, color: u32) {
        if r == 0 {
            self.fill_rect(x, y, w, h, color);
            return;
        }

        for dy in 0..h {
            for dx in 0..w {
                let mut draw = true;
                if dx < r && dy < r { // Top-left
                    if (r - dx) * (r - dx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx >= w - r && dy < r { // Top-right
                    let idx = dx - (w - r - 1);
                    if (idx) * (idx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx < r && dy >= h - r { // Bottom-left
                    let idy = dy - (h - r - 1);
                    if (r - dx) * (r - dx) + (idy) * (idy) > r * r { draw = false; }
                } else if dx >= w - r && dy >= h - r { // Bottom-right
                    let idx = dx - (w - r - 1);
                    let idy = dy - (h - r - 1);
                    if (idx) * (idx) + (idy) * (idy) > r * r { draw = false; }
                }

                if draw {
                    self.set_pixel(x + dx, y + dy, color);
                }
            }
        }
    }

    /// Draw a rounded rectangle with alpha blending.
    pub fn fill_rounded_rect_alpha(&mut self, x: usize, y: usize, w: usize, h: usize, r: usize, color: u32, alpha: u8) {
        if r == 0 {
            self.fill_rect_alpha(x, y, w, h, color, alpha);
            return;
        }

        for dy in 0..h {
            for dx in 0..w {
                let mut draw = true;
                if dx < r && dy < r {
                    if (r - dx) * (r - dx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx >= w - r && dy < r {
                    let idx = dx - (w - r - 1);
                    if (idx) * (idx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx < r && dy >= h - r {
                    let idy = dy - (h - r - 1);
                    if (r - dx) * (r - dx) + (idy) * (idy) > r * r { draw = false; }
                } else if dx >= w - r && dy >= h - r {
                    let idx = dx - (w - r - 1);
                    let idy = dy - (h - r - 1);
                    if (idx) * (idx) + (idy) * (idy) > r * r { draw = false; }
                }

                if draw {
                    self.set_pixel_alpha(x + dx, y + dy, color, alpha);
                }
            }
        }
    }

    /// Draw a neon glowing rectangle (multi-layered border).
    pub fn draw_neon_rect(&mut self, x: usize, y: usize, w: usize, h: usize, r: usize, color: u32) {
        // Outer glow layers
        self.draw_rounded_rect_alpha(x.saturating_sub(2), y.saturating_sub(2), w + 4, h + 4, r + 2, color, 40);
        self.draw_rounded_rect_alpha(x.saturating_sub(1), y.saturating_sub(1), w + 2, h + 2, r + 1, color, 80);
        // Main border
        self.draw_rounded_rect(x, y, w, h, r, color);
    }

    pub fn draw_rounded_rect(&mut self, x: usize, y: usize, w: usize, h: usize, r: usize, color: u32) {
        self.draw_rounded_rect_alpha(x, y, w, h, r, color, 255);
    }

    pub fn draw_rounded_rect_alpha(&mut self, x: usize, y: usize, w: usize, h: usize, r: usize, color: u32, alpha: u8) {
        for dy in 0..h {
            for dx in 0..w {
                let is_border = dx == 0 || dx == w - 1 || dy == 0 || dy == h - 1;
                if !is_border { continue; }

                let mut draw = true;
                if dx < r && dy < r {
                    if (r - dx) * (r - dx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx >= w - r && dy < r {
                    let idx = dx - (w - r - 1);
                    if (idx) * (idx) + (r - dy) * (r - dy) > r * r { draw = false; }
                } else if dx < r && dy >= h - r {
                    let idy = dy - (h - r - 1);
                    if (r - dx) * (r - dx) + (idy) * (idy) > r * r { draw = false; }
                } else if dx >= w - r && dy >= h - r {
                    let idx = dx - (w - r - 1);
                    let idy = dy - (h - r - 1);
                    if (idx) * (idx) + (idy) * (idy) > r * r { draw = false; }
                }

                if draw {
                    self.set_pixel_alpha(x + dx, y + dy, color, alpha);
                }
            }
        }
    }

    /// Set a single pixel.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = (y * self.stride + x) * 4;
        let fb = self.as_mut_slice();
        if offset + 3 < fb.len() {
            fb[offset] = (color & 0xFF) as u8;
            fb[offset + 1] = ((color >> 8) & 0xFF) as u8;
            fb[offset + 2] = ((color >> 16) & 0xFF) as u8;
            fb[offset + 3] = 0;
        }
    }

    /// Draw a filled rectangle.
    pub fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    /// Draw a rectangle outline.
    pub fn draw_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        // Top & bottom
        for dx in 0..w {
            self.set_pixel(x + dx, y, color);
            self.set_pixel(x + dx, y + h.saturating_sub(1), color);
        }
        // Left & right
        for dy in 0..h {
            self.set_pixel(x, y + dy, color);
            self.set_pixel(x + w.saturating_sub(1), y + dy, color);
        }
    }

    /// Draw a single 8x8 character at pixel position (x, y).
    pub fn draw_char(&mut self, x: usize, y: usize, c: char, color: u32) {
        if let Some(glyph) = font8x8::BASIC_FONTS.get(c) {
            for (row, byte) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (byte & (1 << col)) != 0 {
                        self.set_pixel(x + col, y + row, color);
                    }
                }
            }
        }
    }

    /// Draw a string at pixel position (x, y). No wrapping.
    pub fn draw_string(&mut self, x: usize, y: usize, s: &str, color: u32) {
        for (i, c) in s.chars().enumerate() {
            self.draw_char(x + i * CHAR_WIDTH, y, c, color);
        }
    }

    /// Draw a string with word wrapping within a bounding box.
    pub fn draw_string_wrapped(
        &mut self,
        x: usize,
        y: usize,
        max_width: usize,
        s: &str,
        color: u32,
    ) -> usize {
        let chars_per_line = max_width / CHAR_WIDTH;
        if chars_per_line == 0 {
            return 0;
        }

        let mut cx = 0usize;
        let mut cy = 0usize;

        for c in s.chars() {
            if c == '\n' || cx >= chars_per_line {
                cx = 0;
                cy += 1;
            }
            if c == '\n' {
                continue;
            }
            self.draw_char(x + cx * CHAR_WIDTH, y + cy * LINE_HEIGHT, c, color);
            cx += 1;
        }
        cy + 1 // Return number of lines used
    }

    /// Scroll the framebuffer up by `lines` text lines (each LINE_HEIGHT pixels).
    pub fn scroll_up(&mut self, lines: usize) {
        let pixel_shift = lines * LINE_HEIGHT;
        let stride = self.stride;
        let height = self.height;
        let fb = self.as_mut_slice();
        let row_bytes = stride * 4;

        // Copy rows upward
        for y in pixel_shift..height {
            let dst_offset = (y - pixel_shift) * row_bytes;
            let src_offset = y * row_bytes;
            if src_offset + row_bytes <= fb.len() && dst_offset + row_bytes <= fb.len() {
                unsafe {
                    core::ptr::copy(
                        fb.as_ptr().add(src_offset),
                        fb.as_mut_ptr().add(dst_offset),
                        row_bytes,
                    );
                }
            }
        }

        // Clear the bottom region
        let clear_start = (height - pixel_shift) * row_bytes;
        if clear_start < fb.len() {
            for byte in &mut fb[clear_start..] {
                *byte = 0;
            }
        }
    }

    /// Draw a blinking cursor block at the given character grid position.
    pub fn draw_cursor(&mut self, col: usize, row: usize, color: u32) {
        let x = col * CHAR_WIDTH;
        let y = row * LINE_HEIGHT;
        self.fill_rect(x, y, CHAR_WIDTH, CHAR_HEIGHT, color);
    }

    /// XOR a region (used for hardware cursor overlay).
    pub fn xor_rect(&mut self, x: usize, y: usize, w: usize, h: usize) {
        let width = self.width;
        let height = self.height;
        let stride = self.stride;
        let fb = self.as_mut_slice();
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;
                if px < width && py < height {
                    let offset = (py * stride + px) * 4;
                    if offset + 2 < fb.len() {
                        fb[offset] ^= 0xFF;
                        fb[offset + 1] ^= 0xFF;
                        fb[offset + 2] ^= 0xFF;
                    }
                }
            }
        }
    }
}
