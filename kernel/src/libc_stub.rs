//! Phase 1–7: libc stubs for `no_std` UEFI environment.
//!
//! Provides minimal C runtime functions needed by any FFI code.

use core::ffi::{c_void, c_char};
use core::ptr;

#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    let mut d = dest as *mut u8;
    let mut s = src as *const u8;
    let mut len = n;

    // Fast path if both pointers share the same alignment offset
    if (d as usize) % core::mem::size_of::<usize>() == (s as usize) % core::mem::size_of::<usize>() {
        while (d as usize) % core::mem::size_of::<usize>() != 0 && len > 0 {
            core::ptr::write_volatile(d, core::ptr::read_volatile(s));
            d = d.add(1);
            s = s.add(1);
            len -= 1;
        }
        while len >= core::mem::size_of::<usize>() {
            core::ptr::write_volatile(d as *mut usize, core::ptr::read_volatile(s as *const usize));
            d = d.add(core::mem::size_of::<usize>());
            s = s.add(core::mem::size_of::<usize>());
            len -= core::mem::size_of::<usize>();
        }
    }

    // Remainder or unaligned byte-by-byte copy
    for i in 0..len {
        core::ptr::write_volatile(d.add(i), core::ptr::read_volatile(s.add(i)));
    }
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memset(s: *mut c_void, c: i32, n: usize) -> *mut c_void {
    let mut p = s as *mut u8;
    let mut len = n;
    let value = c as u8;
    
    // Create a word-sized fill value
    let mut word_value: usize = 0;
    for i in 0..core::mem::size_of::<usize>() {
        word_value |= (value as usize) << (i * 8);
    }

    // Align the pointer to avoid unaligned volatile writes
    while (p as usize) % core::mem::size_of::<usize>() != 0 && len > 0 {
        core::ptr::write_volatile(p, value);
        p = p.add(1);
        len -= 1;
    }

    // Word-aligned fill
    while len >= core::mem::size_of::<usize>() {
        core::ptr::write_volatile(p as *mut usize, word_value);
        p = p.add(core::mem::size_of::<usize>());
        len -= core::mem::size_of::<usize>();
    }

    // Remainder
    for i in 0..len {
        core::ptr::write_volatile(p.add(i), value);
    }
    s
}

#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    let d = dest as *mut u8;
    let s = src as *const u8;
    if d == s as *mut u8 || n == 0 {
        return dest;
    }
    if (d as usize) < (s as usize) {
        // Forward copy
        for i in 0..n {
            core::ptr::write_volatile(d.add(i), core::ptr::read_volatile(s.add(i)));
        }
    } else {
        // Backward copy
        for i in (0..n).rev() {
            core::ptr::write_volatile(d.add(i), core::ptr::read_volatile(s.add(i)));
        }
    }
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memcmp(s1: *const c_void, s2: *const c_void, n: usize) -> i32 {
    let a = core::slice::from_raw_parts(s1 as *const u8, n);
    let b = core::slice::from_raw_parts(s2 as *const u8, n);
    for i in 0..n {
        if a[i] != b[i] {
            return a[i] as i32 - b[i] as i32;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn strlen(s: *const c_char) -> usize {
    let mut len = 0;
    while *s.add(len) != 0 {
        len += 1;
    }
    len
}

#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        return ptr::null_mut();
    }
    let layout = match core::alloc::Layout::from_size_align(size, 16) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(),
    };
    let ptr = alloc::alloc::alloc(layout);
    ptr as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    let total = nmemb.saturating_mul(size);
    let ptr = malloc(total);
    if !ptr.is_null() {
        memset(ptr, 0, total);
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn free(_ptr: *mut c_void) {
    // Slab allocator handles small frees; bump can't free.
}

#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    if ptr.is_null() {
        return malloc(size);
    }
    let new_ptr = malloc(size);
    if !new_ptr.is_null() {
        // Copy conservatively (we don't know old size)
        memcpy(new_ptr, ptr, size);
    }
    new_ptr
}

#[no_mangle]
pub unsafe extern "C" fn abort() -> ! {
    panic!("C code called abort()");
}

#[no_mangle]
pub unsafe extern "C" fn __assert_fail(
    _assertion: *const c_char,
    _file: *const c_char,
    _line: u32,
    _function: *const c_char,
) -> ! {
    panic!("C assertion failed");
}

#[no_mangle]
pub unsafe extern "C" fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[no_mangle]
pub unsafe extern "C" fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[no_mangle]
pub unsafe extern "C" fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[no_mangle]
pub unsafe extern "C" fn expf(x: f32) -> f32 {
    libm::expf(x)
}

#[no_mangle]
pub unsafe extern "C" fn logf(x: f32) -> f32 {
    libm::logf(x)
}

#[no_mangle]
pub unsafe extern "C" fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

#[no_mangle]
pub unsafe extern "C" fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

#[no_mangle]
pub unsafe extern "C" fn tanhf(x: f32) -> f32 {
    libm::tanhf(x)
}

#[no_mangle]
pub unsafe extern "C" fn fabs(x: f64) -> f64 {
    libm::fabs(x)
}

#[no_mangle]
pub unsafe extern "C" fn fabsf(x: f32) -> f32 {
    libm::fabsf(x)
}
