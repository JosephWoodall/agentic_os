//! Phase 1: Storage — Block device abstraction for reading GGUF weights from a secondary drive.

use alloc::vec::Vec;
use log::{info, warn};
use uefi::prelude::*;
use uefi::proto::media::block::BlockIO;
use uefi::Identify;

/// Errors that can occur during storage operations.
#[derive(Debug)]
pub enum StorageError {
    NoDiskFound,
    ProtocolNotFound,
    ReadError,
    InvalidMagic,
    OutOfBounds,
}

/// A handle to a block device containing model weights.
pub struct BlockDevice {
    data: Vec<u8>,
    pub block_size: usize,
    /// Total number of blocks.
    pub block_count: u64,
}

impl BlockDevice {
    /// Scan all BlockIO handles and find the one containing GGUF weights.
    /// Reads the entire device into memory.
    pub fn find_weights_disk(bt: &BootServices) -> Result<Self, StorageError> {
        info!("Scanning for weights storage...");

        let handles = bt
            .locate_handle_buffer(uefi::table::boot::SearchType::ByProtocol(&BlockIO::GUID))
            .map_err(|_| StorageError::NoDiskFound)?;

        for handle in handles.iter() {
            if let Ok(block_io) = bt.open_protocol_exclusive::<BlockIO>(*handle) {
                let media = block_io.media();

                // Skip removable media and empty disks
                if media.is_removable_media() || media.last_block() == 0 {
                    continue;
                }

                let block_size = media.block_size() as usize;
                let total_blocks = media.last_block() + 1;

                // Read first block to check for GGUF magic
                let mut header_buf = alloc::vec![0u8; block_size];
                if block_io
                    .read_blocks(media.media_id(), 0, &mut header_buf)
                    .is_err()
                {
                    continue;
                }

                if header_buf.len() >= 4 && &header_buf[..4] == b"GGUF" {
                    info!("GGUF magic detected! Loading weights ({} blocks)...", total_blocks);

                    // Read the entire disk into memory
                    let buffer_size = total_blocks as usize * block_size;
                    let mut buffer = alloc::vec![0u8; buffer_size];

                    let result = Self::read_with_retry(&block_io, media.media_id(), &mut buffer, block_size);

                    if result.is_ok() {
                        info!("Weights disk loaded successfully: {} bytes", buffer.len());
                        return Ok(Self {
                            data: buffer,
                            block_size,
                            block_count: total_blocks,
                        });
                    } else {
                        warn!("Failed to read full weights disk.");
                    }
                }
            }
        }

        Err(StorageError::NoDiskFound)
    }

    /// Read with retry logic (up to 3 attempts).
    fn read_with_retry(
        block_io: &BlockIO,
        media_id: u32,
        buffer: &mut [u8],
        _block_size: usize,
    ) -> Result<(), StorageError> {
        for _ in 0..3 {
            if block_io.read_blocks(media_id, 0, buffer).is_ok() {
                return Ok(());
            }
        }
        Err(StorageError::ReadError)
    }

    /// Access the raw bytes of the disk.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Read a range of bytes from the loaded disk.
    pub fn read_range(&self, offset: usize, len: usize) -> Result<&[u8], StorageError> {
        if offset + len > self.data.len() {
            return Err(StorageError::OutOfBounds);
        }
        Ok(&self.data[offset..offset + len])
    }

    /// Total size in bytes.
    pub fn total_size(&self) -> usize {
        self.data.len()
    }
}
