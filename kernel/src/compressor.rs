//! Phase 4: State Compressor — Compresses process state into a fixed-size continuous
//! latent representation and decompresses (hydrates) it back.
//!
//! Implements Pillar 2: "Memory as a Continuous Latent Space."

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use alloc::format;

/// Dimension of the compressed latent state vector.
pub const LATENT_DIM: usize = 64;

/// Compresses and decompresses process state to/from a fixed-size latent vector.
pub struct StateCompressor;

impl StateCompressor {
    /// Compress a process's text history into a fixed-size latent vector.
    ///
    /// Uses a hash-based feature extraction approach:
    /// - Running character statistics (mean, variance of byte values)
    /// - N-gram hash features for content-addressable summary
    /// - Length and structure features
    pub fn compress(history: &str) -> Vec<f32> {
        let mut latent = vec![0.0f32; LATENT_DIM];

        if history.is_empty() {
            return latent;
        }

        let bytes = history.as_bytes();
        let len = bytes.len() as f32;

        // Feature 0: Normalized length
        latent[0] = (len / 1024.0).min(1.0);

        // Features 1-4: Byte statistics
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut min_b = 255.0f32;
        let mut max_b = 0.0f32;

        for &b in bytes {
            let v = b as f32;
            sum += v;
            sum_sq += v * v;
            if v < min_b { min_b = v; }
            if v > max_b { max_b = v; }
        }

        latent[1] = sum / len / 255.0; // Mean
        let mean_val = sum / len;
        latent[2] = libm::sqrtf((sum_sq / len - mean_val * mean_val).max(0.0)) / 255.0; // Std dev
        latent[3] = min_b / 255.0;
        latent[4] = max_b / 255.0;

        // Features 5-8: Character class ratios
        let mut alpha = 0u32;
        let mut digit = 0u32;
        let mut space = 0u32;
        let mut punct = 0u32;

        for &b in bytes {
            if b.is_ascii_alphabetic() { alpha += 1; }
            else if b.is_ascii_digit() { digit += 1; }
            else if b.is_ascii_whitespace() { space += 1; }
            else { punct += 1; }
        }

        let total = bytes.len() as f32;
        latent[5] = alpha as f32 / total;
        latent[6] = digit as f32 / total;
        latent[7] = space as f32 / total;
        latent[8] = punct as f32 / total;

        // Features 9-12: Line/structure features
        let lines = bytes.iter().filter(|&&b| b == b'\n').count();
        latent[9] = (lines as f32 / 100.0).min(1.0);

        // Features 10-LATENT_DIM: N-gram hash features
        // Use rolling hash over character trigrams, project into latent dims
        let ngram_start = 10;
        let ngram_count = LATENT_DIM - ngram_start;

        if bytes.len() >= 3 {
            for window in bytes.windows(3) {
                let hash = Self::trigram_hash(window);
                let idx = ngram_start + (hash as usize % ngram_count);
                latent[idx] += 1.0;
            }

            // Normalize n-gram features
            let max_ngram = latent[ngram_start..]
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);
            if max_ngram > 0.0 {
                for v in &mut latent[ngram_start..] {
                    *v /= max_ngram;
                }
            }
        }

        latent
    }

    /// Decompress (hydrate) a latent vector back into an approximate text summary.
    ///
    /// Since the compression is lossy (by design — this is the continuous latent space),
    /// we reconstruct a summary description of what the process was doing.
    pub fn decompress(latent: &[f32]) -> String {
        if latent.len() < LATENT_DIM || latent.iter().all(|&v| v == 0.0) {
            return String::from("[Empty process state]");
        }

        let mut summary = String::from("[Hydrated from latent space] ");

        // Reconstruct rough characteristics
        let length_est = (latent[0] * 1024.0) as usize;
        let _ = core::fmt::Write::write_fmt(
            &mut summary,
            format_args!("History ~{} chars. ", length_est),
        );

        // Character profile
        let alpha_ratio = latent[5];
        let digit_ratio = latent[6];

        if alpha_ratio > 0.7 {
            summary.push_str("Primarily text content. ");
        } else if digit_ratio > 0.3 {
            summary.push_str("Contains significant numeric data. ");
        } else {
            summary.push_str("Mixed content. ");
        }

        // Complexity estimate from n-gram diversity
        let ngram_start = 10;
        let nonzero_ngrams = latent[ngram_start..].iter().filter(|&&v| v > 0.1).count();
        let complexity = nonzero_ngrams as f32 / (LATENT_DIM - ngram_start) as f32;

        if complexity > 0.5 {
            summary.push_str("High complexity/diversity.");
        } else if complexity > 0.2 {
            summary.push_str("Moderate complexity.");
        } else {
            summary.push_str("Low complexity/repetitive.");
        }

        summary
    }

    /// Simple trigram hash function.
    fn trigram_hash(window: &[u8]) -> u32 {
        let mut h = 0x811c9dc5u32; // FNV-1a offset basis
        for &b in window {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193); // FNV-1a prime
        }
        h
    }
}
