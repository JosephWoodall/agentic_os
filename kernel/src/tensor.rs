//! Phase 2: Tensor operations — Core math primitives for transformer inference.
//!
//! Provides RMSNorm, softmax, matmul, RoPE, GeLU, argmax, and top-p sampling.

use alloc::vec;
use alloc::vec::Vec;

pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![0.0; total],
            shape,
        }
    }

    /// Last dimension size.
    pub fn last_dim(&self) -> usize {
        *self.shape.last().unwrap_or(&0)
    }

    /// RMS Layer Normalization.
    pub fn rms_norm(&self, weight: &[f32], eps: f32) -> Self {
        let size = self.last_dim();
        let mut output = vec![0.0; self.data.len()];

        for i in (0..self.data.len()).step_by(size) {
            let chunk = &self.data[i..i + size];
            let mut ss = 0.0;
            for &x in chunk {
                ss += x * x;
            }
            ss /= size as f32;
            let inv_sqrt = 1.0 / libm::sqrtf(ss + eps);

            for j in 0..size {
                output[i + j] = weight[j] * (chunk[j] * inv_sqrt);
            }
        }

        Self::new(output, self.shape.clone())
    }

    /// Standard Layer Normalization.
    pub fn layer_norm(&self, weight: &[f32], bias: &[f32], eps: f32) -> Self {
        let size = self.last_dim();
        let mut output = vec![0.0; self.data.len()];

        for i in (0..self.data.len()).step_by(size) {
            let chunk = &self.data[i..i + size];

            // Mean
            let mut mean = 0.0;
            for &x in chunk {
                mean += x;
            }
            mean /= size as f32;

            // Variance
            let mut var = 0.0;
            for &x in chunk {
                let d = x - mean;
                var += d * d;
            }
            var /= size as f32;

            let inv_std = 1.0 / libm::sqrtf(var + eps);
            for j in 0..size {
                output[i + j] = weight[j] * ((chunk[j] - mean) * inv_std) + bias[j];
            }
        }

        Self::new(output, self.shape.clone())
    }

    /// In-place softmax over the last dimension.
    pub fn softmax(&mut self) {
        let size = self.last_dim();
        for i in (0..self.data.len()).step_by(size) {
            let chunk = &mut self.data[i..i + size];

            let mut max_val = chunk[0];
            for &x in chunk.iter() {
                if x > max_val {
                    max_val = x;
                }
            }

            let mut sum = 0.0;
            for x in chunk.iter_mut() {
                *x = libm::expf(*x - max_val);
                sum += *x;
            }

            for x in chunk.iter_mut() {
                *x /= sum;
            }
        }
    }

    /// GeLU activation (Gaussian Error Linear Unit) — approximate version.
    pub fn gelu(&mut self) {
        for x in self.data.iter_mut() {
            // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let x3 = *x * *x * *x;
            let inner = 0.7978845608 * (*x + 0.044715 * x3); // sqrt(2/pi) ≈ 0.7978845608
            *x = 0.5 * *x * (1.0 + libm::tanhf(inner));
        }
    }

    /// Argmax — returns the index of the maximum value.
    pub fn argmax(&self) -> usize {
        let mut max_idx = 0;
        let mut max_val = self.data[0];
        for (i, &v) in self.data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        max_idx
    }

    /// Top-p (nucleus) sampling.
    /// Returns a token index sampled from the top-p probability mass.
    pub fn sample_top_p(&self, top_p: f32, temperature: f32, seed: u64) -> usize {
        let size = self.data.len();
        let mut logits = self.data.clone();

        // Apply temperature
        if temperature > 0.0 && temperature != 1.0 {
            for x in logits.iter_mut() {
                *x /= temperature;
            }
        }

        // Softmax
        let mut max_val = logits[0];
        for &x in logits.iter() {
            if x > max_val {
                max_val = x;
            }
        }
        let mut sum = 0.0f32;
        for x in logits.iter_mut() {
            *x = libm::expf(*x - max_val);
            sum += *x;
        }
        for x in logits.iter_mut() {
            *x /= sum;
        }

        // Sort indices by probability (descending)
        let mut indices: Vec<usize> = (0..size).collect();
        // Simple insertion sort (good enough for vocab sizes in no_std)
        for i in 1..indices.len() {
            let mut j = i;
            while j > 0 && logits[indices[j]] > logits[indices[j - 1]] {
                indices.swap(j, j - 1);
                j -= 1;
            }
        }

        // Accumulate probability mass until we hit top_p
        let mut cumulative = 0.0f32;
        let mut cutoff = size;
        for (k, &idx) in indices.iter().enumerate() {
            cumulative += logits[idx];
            if cumulative >= top_p {
                cutoff = k + 1;
                break;
            }
        }

        // Re-normalize the top-p set
        let top_indices = &indices[..cutoff];
        let mut top_sum = 0.0f32;
        for &idx in top_indices {
            top_sum += logits[idx];
        }

        // Simple PRNG sampling
        let r = pseudo_random(seed) as f32 / u64::MAX as f32;
        let mut accum = 0.0f32;
        for &idx in top_indices {
            accum += logits[idx] / top_sum;
            if accum >= r {
                return idx;
            }
        }

        // Fallback
        indices[0]
    }
}

/// Simple xorshift64 PRNG for sampling.
pub fn pseudo_random(mut seed: u64) -> u64 {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    seed
}

/// Matrix multiplication: out[m×n] = a[m×k] × b[k×n]
pub fn matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Matrix-vector multiplication: out[n] = mat[n×d] × vec[d]
pub fn matvec(out: &mut [f32], mat: &[f32], vec_in: &[f32], n: usize, d: usize) {
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..d {
            sum += mat[i * d + j] * vec_in[j];
        }
        out[i] = sum;
    }
}

/// Apply Rotary Position Embeddings (RoPE) in-place.
///
/// `q` and `k` are attention vectors of dimension `head_dim`.
/// `pos` is the absolute position index.
/// `head_dim` is the dimension per head.
pub fn rope_embedding(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / libm::powf(theta, (2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let cos_val = libm::cosf(angle);
        let sin_val = libm::sinf(angle);

        // Apply rotation to q
        let q0 = q[i];
        let q1 = q[i + half];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + half] = q0 * sin_val + q1 * cos_val;

        // Apply rotation to k
        let k0 = k[i];
        let k1 = k[i + half];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + half] = k0 * sin_val + k1 * cos_val;
    }
}

/// Element-wise addition: a += b
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

/// Element-wise multiplication: a *= b (for gating in FFN)
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= *y;
    }
}
