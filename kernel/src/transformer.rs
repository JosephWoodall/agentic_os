//! Phase 2: Transformer — Gemma architecture forward pass in pure Rust `no_std`.
//!
//! Implements the full transformer pipeline:
//! embedding → N×(RMSNorm → MHA → RMSNorm → FFN) → RMSNorm → logits
//! with KV-cache for autoregressive generation.

use alloc::vec;
use alloc::vec::Vec;
use crate::tensor::{self, Tensor, rope_embedding, matvec, add_inplace, mul_inplace};

/// Transformer model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub dim: usize,          // Model dimension (hidden size)
    pub hidden_dim: usize,   // FFN intermediate dimension
    pub n_layers: usize,     // Number of transformer layers
    pub n_heads: usize,      // Number of attention heads
    pub n_kv_heads: usize,   // Number of KV heads (for GQA)
    pub vocab_size: usize,   // Vocabulary size
    pub max_seq_len: usize,  // Maximum sequence length
    pub head_dim: usize,     // Dimension per head (dim / n_heads)
    pub rope_theta: f32,     // RoPE base frequency
    pub rms_norm_eps: f32,   // RMS norm epsilon
}

impl TransformerConfig {
    /// Default Gemma-2B-like configuration.
    pub fn gemma_2b() -> Self {
        Self {
            dim: 2048,
            hidden_dim: 16384,
            n_layers: 18,
            n_heads: 8,
            n_kv_heads: 1,
            vocab_size: 256000,
            max_seq_len: 8192,
            head_dim: 256,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// Small config for mock/testing.
    pub fn mock() -> Self {
        Self {
            dim: 64,
            hidden_dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 1,
            vocab_size: 128,
            max_seq_len: 512,
            head_dim: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Weights for a single transformer layer.
pub struct LayerWeights {
    /// Attention input norm weights [dim]
    pub attn_norm: Vec<f32>,
    /// Q projection [n_heads * head_dim, dim]
    pub wq: Vec<f32>,
    /// K projection [n_kv_heads * head_dim, dim]
    pub wk: Vec<f32>,
    /// V projection [n_kv_heads * head_dim, dim]
    pub wv: Vec<f32>,
    /// Output projection [dim, n_heads * head_dim]
    pub wo: Vec<f32>,
    /// FFN input norm weights [dim]
    pub ffn_norm: Vec<f32>,
    /// FFN gate projection [hidden_dim, dim]
    pub w_gate: Vec<f32>,
    /// FFN up projection [hidden_dim, dim]
    pub w_up: Vec<f32>,
    /// FFN down projection [dim, hidden_dim]
    pub w_down: Vec<f32>,
}

/// All weights for the transformer model.
pub struct TransformerWeights {
    /// Token embedding table [vocab_size, dim]
    pub token_embedding: Vec<f32>,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// Final RMS norm [dim]
    pub final_norm: Vec<f32>,
    /// Output projection (often tied to embedding) [vocab_size, dim]
    pub output: Vec<f32>,
}

/// Mutable state for inference (KV cache + scratch buffers).
pub struct TransformerState {
    /// KV cache: key_cache[layer][pos * kv_dim .. (pos+1) * kv_dim]
    pub key_cache: Vec<Vec<f32>>,
    /// KV cache: val_cache[layer][pos * kv_dim .. (pos+1) * kv_dim]
    pub val_cache: Vec<Vec<f32>>,
    /// Current sequence position.
    pub pos: usize,
}

impl TransformerState {
    pub fn new(config: &TransformerConfig) -> Self {
        let kv_dim = config.n_kv_heads * config.head_dim;
        let cache_size = config.max_seq_len * kv_dim;

        let mut key_cache = Vec::with_capacity(config.n_layers);
        let mut val_cache = Vec::with_capacity(config.n_layers);

        for _ in 0..config.n_layers {
            key_cache.push(vec![0.0; cache_size]);
            val_cache.push(vec![0.0; cache_size]);
        }

        Self {
            key_cache,
            val_cache,
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        for cache in self.key_cache.iter_mut() {
            for v in cache.iter_mut() {
                *v = 0.0;
            }
        }
        for cache in self.val_cache.iter_mut() {
            for v in cache.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

/// The core transformer model.
pub struct Transformer {
    pub config: TransformerConfig,
    pub weights: TransformerWeights,
    pub state: TransformerState,
}

impl Transformer {
    /// Create a transformer with random/zero weights (for mock mode).
    pub fn mock(config: TransformerConfig) -> Self {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;

        // Initialize weights with small random values for testing
        let mut seed = 42u64;
        let init_weight = |size: usize, s: &mut u64| -> Vec<f32> {
            let mut w = vec![0.0f32; size];
            for v in w.iter_mut() {
                *s = tensor::pseudo_random(*s);
                *v = (*s as f32 / u64::MAX as f32 - 0.5) * 0.02;
            }
            w
        };

        let token_embedding = init_weight(vocab_size * dim, &mut seed);
        let output = init_weight(vocab_size * dim, &mut seed);
        let final_norm = vec![1.0; dim]; // Norm weights initialized to 1.0

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(LayerWeights {
                attn_norm: vec![1.0; dim],
                wq: init_weight(n_heads * head_dim * dim, &mut seed),
                wk: init_weight(n_kv_heads * head_dim * dim, &mut seed),
                wv: init_weight(n_kv_heads * head_dim * dim, &mut seed),
                wo: init_weight(dim * n_heads * head_dim, &mut seed),
                ffn_norm: vec![1.0; dim],
                w_gate: init_weight(hidden_dim * dim, &mut seed),
                w_up: init_weight(hidden_dim * dim, &mut seed),
                w_down: init_weight(dim * hidden_dim, &mut seed),
            });
        }

        let weights = TransformerWeights {
            token_embedding,
            layers,
            final_norm,
            output,
        };

        let state = TransformerState::new(&config);

        Self {
            config,
            weights,
            state,
        }
    }

    /// Run a single forward pass for one token at the current position.
    /// Returns logits of shape [vocab_size].
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let pos = self.state.pos;

        // Embedding lookup
        let mut x = vec![0.0f32; dim];
        let emb_offset = (token as usize) * dim;
        if emb_offset + dim <= self.weights.token_embedding.len() {
            x.copy_from_slice(&self.weights.token_embedding[emb_offset..emb_offset + dim]);
        }

        // Gemma: scale embedding by sqrt(dim)
        let scale = libm::sqrtf(dim as f32);
        for v in x.iter_mut() {
            *v *= scale;
        }

        // Scratch buffers
        let mut xb = vec![0.0f32; dim];       // After norm
        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut xb2 = vec![0.0f32; dim];      // After attention output
        let mut hb = vec![0.0f32; cfg.hidden_dim];  // FFN hidden
        let mut hb2 = vec![0.0f32; cfg.hidden_dim]; // FFN gate

        for layer in 0..cfg.n_layers {
            let w = &self.weights.layers[layer];

            // ---- Attention ----
            // RMSNorm
            {
                let t = Tensor::new(x.clone(), vec![dim]);
                let normed = t.rms_norm(&w.attn_norm, cfg.rms_norm_eps);
                xb.copy_from_slice(&normed.data);
            }

            // Q, K, V projections
            matvec(&mut q, &w.wq, &xb, n_heads * head_dim, dim);
            matvec(&mut k, &w.wk, &xb, kv_dim, dim);
            matvec(&mut v, &w.wv, &xb, kv_dim, dim);

            // RoPE on each head's Q and K
            let kv_mul = n_heads / n_kv_heads;
            for h in 0..n_heads {
                let kv_h = h / kv_mul;
                let q_offset = h * head_dim;
                let k_offset = kv_h * head_dim;
                rope_embedding(
                    &mut q[q_offset..q_offset + head_dim],
                    &mut k[k_offset..k_offset + head_dim],
                    pos,
                    head_dim,
                    cfg.rope_theta,
                );
            }

            // Store K,V in cache
            let cache_offset = pos * kv_dim;
            if cache_offset + kv_dim <= self.state.key_cache[layer].len() {
                self.state.key_cache[layer][cache_offset..cache_offset + kv_dim]
                    .copy_from_slice(&k);
                self.state.val_cache[layer][cache_offset..cache_offset + kv_dim]
                    .copy_from_slice(&v);
            }

            // Multi-head attention with KV cache
            let mut attn_out = vec![0.0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let kv_h = h / kv_mul;
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                // Compute attention scores for all cached positions
                let mut scores = vec![0.0f32; pos + 1];
                let att_scale = 1.0 / libm::sqrtf(head_dim as f32);

                for t in 0..=pos {
                    let k_offset = t * kv_dim + kv_h * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * self.state.key_cache[layer][k_offset + d];
                    }
                    scores[t] = dot * att_scale;
                }

                // Softmax over scores
                let mut max_score = scores[0];
                for &s in scores.iter() {
                    if s > max_score {
                        max_score = s;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = libm::expf(*s - max_score);
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }

                // Weighted sum of values
                let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for t in 0..=pos {
                    let v_offset = t * kv_dim + kv_h * head_dim;
                    let score = scores[t];
                    for d in 0..head_dim {
                        out_head[d] += score * self.state.val_cache[layer][v_offset + d];
                    }
                }
            }

            // Output projection
            matvec(&mut xb2, &w.wo, &attn_out, dim, n_heads * head_dim);

            // Residual connection
            add_inplace(&mut x, &xb2);

            // ---- FFN ----
            // RMSNorm
            {
                let t = Tensor::new(x.clone(), vec![dim]);
                let normed = t.rms_norm(&w.ffn_norm, cfg.rms_norm_eps);
                xb.copy_from_slice(&normed.data);
            }

            // Gate and Up projections
            matvec(&mut hb, &w.w_gate, &xb, cfg.hidden_dim, dim);
            matvec(&mut hb2, &w.w_up, &xb, cfg.hidden_dim, dim);

            // GeLU on gate, then multiply with up
            let mut gate_tensor = Tensor::new(hb.clone(), vec![cfg.hidden_dim]);
            gate_tensor.gelu();
            hb.copy_from_slice(&gate_tensor.data);
            mul_inplace(&mut hb, &hb2);

            // Down projection
            matvec(&mut xb2, &w.w_down, &hb, dim, cfg.hidden_dim);

            // Residual connection
            add_inplace(&mut x, &xb2);
        }

        // Final RMSNorm
        {
            let t = Tensor::new(x.clone(), vec![dim]);
            let normed = t.rms_norm(&self.weights.final_norm, cfg.rms_norm_eps);
            x.copy_from_slice(&normed.data);
        }

        // Logits: output projection
        let mut logits = vec![0.0f32; cfg.vocab_size];
        matvec(&mut logits, &self.weights.output, &x, cfg.vocab_size, dim);

        self.state.pos += 1;
        logits
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, prompt_tokens: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let mut output_tokens = Vec::new();

        // Process prompt (prefill)
        for &token in prompt_tokens {
            let _logits = self.forward(token);
        }

        // Generate new tokens
        let mut next_token = {
            let logits = if prompt_tokens.is_empty() {
                self.forward(0) // BOS
            } else {
                // Last forward already computed logits
                let last = *prompt_tokens.last().unwrap();
                self.forward(last)
            };
            let t = Tensor::new(logits, vec![self.config.vocab_size]);
            t.argmax() as u32
        };

        for _ in 0..max_new_tokens {
            output_tokens.push(next_token);

            if next_token == 1 {
                // EOS
                break;
            }

            let logits = self.forward(next_token);
            let t = Tensor::new(logits, vec![self.config.vocab_size]);
            next_token = t.argmax() as u32;
        }

        output_tokens
    }
}
