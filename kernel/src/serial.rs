use core::arch::asm;
use alloc::string::String;

const COM1: u16 = 0x3F8;

pub fn init() {
    unsafe {
        outb(COM1 + 1, 0x00);    // Disable all interrupts
        outb(COM1 + 3, 0x80);    // Enable DLAB (set baud rate divisor)
        outb(COM1 + 0, 0x03);    // Set divisor to 3 (lo byte) 38400 baud
        outb(COM1 + 1, 0x00);    //                  (hi byte)
        outb(COM1 + 3, 0x03);    // 8 bits, no parity, one stop bit
        outb(COM1 + 2, 0xC7);    // Enable FIFO, clear them, with 14-byte threshold
        outb(COM1 + 4, 0x0B);    // IRQs enabled, RTS/DSR set
    }
}

unsafe fn outb(port: u16, val: u8) {
    asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
}

unsafe fn inb(port: u16) -> u8 {
    let ret: u8;
    asm!("in al, dx", out("al") ret, in("dx") port, options(nomem, nostack, preserves_flags));
    ret
}

pub fn is_transmit_empty() -> bool {
    unsafe { inb(COM1 + 5) & 0x20 != 0 }
}

pub fn write_byte(b: u8) {
    while !is_transmit_empty() {
        core::hint::spin_loop();
    }
    unsafe { outb(COM1, b); }
}

pub fn write_str(s: &str) {
    for b in s.bytes() {
        write_byte(b);
    }
}

pub fn is_data_ready() -> bool {
    unsafe { inb(COM1 + 5) & 1 != 0 }
}

pub fn read_byte() -> u8 {
    while !is_data_ready() {
        core::hint::spin_loop();
    }
    unsafe { inb(COM1) }
}

pub fn read_line() -> String {
    let mut s = String::new();
    loop {
        let b = read_byte();
        let c = b as char;
        if c == '\n' {
            break;
        }
        if c != '\r' {
            s.push(c);
        }
    }
    s
}
