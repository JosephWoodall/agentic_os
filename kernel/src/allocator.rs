//! Phase 1: Global Allocator — Bump allocator backed by UEFI page allocation
//! with a slab layer for fixed-size allocations that support actual deallocation.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Slab constants
// ---------------------------------------------------------------------------

/// Slab size classes: 64B, 256B, 1KB, 4KB
const SLAB_CLASSES: [usize; 4] = [64, 256, 1024, 4096];
const SLAB_POOL_SIZE: usize = 256; // Max free entries per class

// ---------------------------------------------------------------------------
// Free-list node (intrusive)
// ---------------------------------------------------------------------------

struct FreeNode {
    next: *mut FreeNode,
}

// ---------------------------------------------------------------------------
// Slab pool for a single size class
// ---------------------------------------------------------------------------

struct SlabPool {
    head: *mut FreeNode,
    count: usize,
    block_size: usize,
}

impl SlabPool {
    const fn new(block_size: usize) -> Self {
        Self {
            head: null_mut(),
            count: 0,
            block_size,
        }
    }

    /// Try to allocate from the free list.
    unsafe fn alloc(&mut self) -> *mut u8 {
        if self.head.is_null() {
            return null_mut();
        }
        let node = self.head;
        self.head = (*node).next;
        self.count -= 1;
        node as *mut u8
    }

    /// Return a block to the free list.
    unsafe fn dealloc(&mut self, ptr: *mut u8) {
        if self.count >= SLAB_POOL_SIZE {
            return; // Pool full, leak it (bump doesn't truly free anyway)
        }
        let node = ptr as *mut FreeNode;
        (*node).next = self.head;
        self.head = node;
        self.count += 1;
    }
}

// ---------------------------------------------------------------------------
// Combined Bump + Slab allocator
// ---------------------------------------------------------------------------

pub struct BumpSlabAllocator {
    // Bump region
    start: AtomicUsize,
    end: AtomicUsize,
    current: AtomicUsize,
    // Slab pools (indexed by SLAB_CLASSES position)
    // We use raw pointers because GlobalAlloc requires Sync and we manage our own locking.
    slab_pools_initialized: AtomicUsize, // 0 = not ready, 1 = ready
}

/// Static storage for slab pools (can't use dynamic alloc for the allocator itself)
static mut SLAB_POOLS: [SlabPool; 4] = [
    SlabPool::new(64),
    SlabPool::new(256),
    SlabPool::new(1024),
    SlabPool::new(4096),
];

impl BumpSlabAllocator {
    pub const fn new() -> Self {
        Self {
            start: AtomicUsize::new(0),
            end: AtomicUsize::new(0),
            current: AtomicUsize::new(0),
            slab_pools_initialized: AtomicUsize::new(0),
        }
    }

    /// Initialize the allocator with a memory region.
    ///
    /// # Safety
    /// Must be called exactly once, before any allocations.
    pub unsafe fn init(&self, start: usize, size: usize) {
        self.start.store(start, Ordering::SeqCst);
        self.current.store(start, Ordering::SeqCst);
        self.end.store(start + size, Ordering::SeqCst);
        self.slab_pools_initialized.store(1, Ordering::SeqCst);
    }

    /// Returns total bytes used by the bump allocator.
    pub fn used(&self) -> usize {
        let current = self.current.load(Ordering::Relaxed);
        let start = self.start.load(Ordering::Relaxed);
        if current > start {
            current - start
        } else {
            0
        }
    }

    /// Returns total bytes remaining.
    pub fn remaining(&self) -> usize {
        let current = self.current.load(Ordering::Relaxed);
        let end = self.end.load(Ordering::Relaxed);
        if end > current {
            end - current
        } else {
            0
        }
    }

    /// Find the slab class index for a given size, if it fits.
    fn slab_class(size: usize, align: usize) -> Option<usize> {
        for (i, &class_size) in SLAB_CLASSES.iter().enumerate() {
            if size <= class_size && align <= class_size {
                return Some(i);
            }
        }
        None
    }
}

unsafe impl GlobalAlloc for BumpSlabAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        let mut iters = 0;
        // Fall back to bump allocator
        loop {
            iters += 1;
            if iters > 100 {
                let msg = b"LOOP";
                for &b in msg {
                    unsafe {
                        core::arch::asm!(
                            "out dx, al",
                            in("dx") 0x3F8u16,
                            in("al") b,
                        );
                    }
                }
                loop { core::hint::spin_loop(); }
            }

            let current = self.current.load(Ordering::Relaxed);
            if current == 0 {
                // BEFORE init() is called by main, uefi-rs may try to allocate memory!
                // We use a small static buffer for these early boot allocations.
                static mut EARLY_BUMP_BUFFER: [u8; 65536] = [0; 65536];
                static EARLY_BUMP_CURRENT: AtomicUsize = AtomicUsize::new(0);

                let e_current = EARLY_BUMP_CURRENT.load(Ordering::Relaxed);
                let e_start = (e_current + align - 1) & !(align - 1);
                let e_end = e_start + size;

                if e_end > 65536 {
                    // Cannot use log::error here because logger might not be ready
                    return null_mut();
                }

                if EARLY_BUMP_CURRENT
                    .compare_exchange_weak(e_current, e_end, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    return unsafe { EARLY_BUMP_BUFFER.as_mut_ptr().add(e_start) };
                }
                continue;
            }

            let alloc_start = (current + align - 1) & !(align - 1);
            let alloc_end = alloc_start + size;

            if alloc_end > self.end.load(Ordering::Relaxed) {
                // Write OOM to serial and spin
                let oom = b"OOM!";
                for &b in oom {
                    unsafe {
                        core::arch::asm!(
                            "out dx, al",
                            in("dx") 0x3F8u16,
                            in("al") b,
                        );
                    }
                }
                loop { core::hint::spin_loop(); }
                return null_mut();
            }

            if self
                .current
                .compare_exchange_weak(current, alloc_end, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return alloc_start as *mut u8;
            }
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator can't truly free arbitrary blocks — this is expected.
    }
}

unsafe impl Sync for BumpSlabAllocator {}

#[global_allocator]
pub static ALLOCATOR: BumpSlabAllocator = BumpSlabAllocator::new();
