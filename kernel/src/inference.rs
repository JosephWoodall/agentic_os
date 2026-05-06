//! Phase 2: Inference Engine — Top-level API wrapping the transformer, tokenizer, and GGUF loader.
//!
//! Provides `InferenceEngine` with a unified `generate(prompt) -> String` interface.
//! Operates in either real mode (with GGUF weights) or mock mode (deterministic JSON syscalls).

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::gguf::{GgufParser, GgufModel};
use crate::tokenizer::{Tokenizer, TokenEntry};
use crate::transformer::{Transformer, TransformerConfig};

/// The inference mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceMode {
    /// Real transformer inference with loaded weights.
    Real,
    /// Mock mode: returns deterministic JSON syscalls based on keyword matching.
    Mock,
}

/// Top-level inference engine wrapping all components.
pub struct InferenceEngine {
    pub mode: InferenceMode,
    pub tokenizer: Tokenizer,
    pub transformer: Option<Transformer>,
    /// Tick counter for mock mode seed variation.
    pub tick: u64,
}

impl InferenceEngine {
    /// Initialize from raw GGUF weight data.
    /// Falls back to mock mode if the data is insufficient for real inference.
    pub fn init(weights_data: &[u8]) -> Self {
        if weights_data.len() < 32 {
            log::warn!("No valid weights data. Initializing in MOCK mode.");
            return Self::mock();
        }

        let mut parser = GgufParser::new(weights_data);
        match parser.parse() {
            Ok(model) => {
                log::info!("GGUF parsed: version={}, tensors={}, metadata_keys={}",
                    model.version, model.tensors.len(),
                    model.metadata.len());

                if let Some(arch) = model.architecture() {
                    log::info!("Model architecture: {}", arch);
                }

                // Check if we have actual tensor data
                if model.tensors.is_empty() {
                    log::info!("No tensors in GGUF — using MOCK inference mode.");
                    return Self::mock();
                }

                // Attempt to build real tokenizer from GGUF vocab
                let tokenizer = Self::build_tokenizer(&model);

                // Build transformer config from GGUF metadata
                let config = Self::build_config(&model);
                log::info!("Transformer config: dim={}, layers={}, heads={}, vocab={}",
                    config.dim, config.n_layers, config.n_heads, config.vocab_size);

                // For now, use mock transformer even with real GGUF
                // (until we implement weight deserialization from quantized formats)
                let transformer = Transformer::mock(config);

                Self {
                    mode: InferenceMode::Real,
                    tokenizer,
                    transformer: Some(transformer),
                    tick: 0,
                }
            }
            Err(e) => {
                log::error!("GGUF parse failed: {:?} — using MOCK mode.", e);
                Self::mock()
            }
        }
    }

    /// Create a mock inference engine for testing.
    pub fn mock() -> Self {
        Self {
            mode: InferenceMode::Mock,
            tokenizer: Tokenizer::mock(),
            transformer: None,
            tick: 0,
        }
    }

    /// Build a tokenizer from GGUF model metadata.
    fn build_tokenizer(model: &GgufModel) -> Tokenizer {
        if let (Some(tokens), Some(scores)) = (model.vocab_tokens(), model.vocab_scores()) {
            let mut vocab = Vec::with_capacity(tokens.len());
            for (i, token_text) in tokens.iter().enumerate() {
                let score = if i < scores.len() { scores[i] } else { 0.0 };
                vocab.push(TokenEntry {
                    text: token_text.clone(),
                    score,
                    token_type: if token_text.starts_with('<') && token_text.ends_with('>') {
                        3 // special token
                    } else {
                        1 // normal token
                    },
                });
            }
            log::info!("Loaded vocabulary: {} tokens", vocab.len());
            Tokenizer::new(vocab)
        } else {
            log::warn!("No vocabulary in GGUF — using mock tokenizer.");
            Tokenizer::mock()
        }
    }

    /// Build transformer config from GGUF metadata.
    fn build_config(model: &GgufModel) -> TransformerConfig {
        let dim = model.dim().unwrap_or(2048);
        let n_layers = model.n_layers().unwrap_or(18);
        let n_heads = model.n_heads().unwrap_or(8);
        let n_kv_heads = model.n_kv_heads().unwrap_or(1);
        let vocab_size = model.vocab_size().unwrap_or(256000);
        let head_dim = dim / n_heads;

        TransformerConfig {
            dim,
            hidden_dim: dim * 8, // Gemma uses 8× expansion
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len: 8192,
            head_dim,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// Generate a response given a prompt string.
    ///
    /// In mock mode, returns deterministic JSON syscalls based on the prompt content.
    /// In real mode, runs the transformer forward pass.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        self.tick += 1;

        match self.mode {
            InferenceMode::Mock => self.mock_generate(prompt),
            InferenceMode::Real => self.real_generate(prompt, max_tokens),
        }
    }

    /// Real transformer generation.
    fn real_generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        if let Some(ref mut transformer) = self.transformer {
            transformer.state.reset();
            let tokens = self.tokenizer.encode(prompt);
            let output_tokens = transformer.generate(&tokens, max_tokens);
            self.tokenizer.decode(&output_tokens)
        } else {
            self.mock_generate(prompt)
        }
    }

    /// Mock generation: keyword-driven deterministic JSON syscall output.
    /// This allows end-to-end pipeline testing without real model weights.
    fn mock_generate(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();

        // Match user intent to syscalls
        if prompt_lower.contains("spawn") || prompt_lower.contains("start") || prompt_lower.contains("run") {
            // Extract a task name from the prompt
            let name = if prompt_lower.contains("browser") {
                "browser"
            } else if prompt_lower.contains("editor") {
                "text_editor"
            } else if prompt_lower.contains("terminal") {
                "terminal"
            } else {
                "user_task"
            };

            format!(
                r#"{{"command": "spawn_process", "name": "{}", "priority": 5}}"#,
                name
            )
        } else if prompt_lower.contains("kill") || prompt_lower.contains("stop") || prompt_lower.contains("end") {
            let pid = self.extract_number(&prompt_lower).unwrap_or(2);
            format!(r#"{{"command": "kill_process", "pid": {}}}"#, pid)
        } else if prompt_lower.contains("read") || prompt_lower.contains("open") || prompt_lower.contains("cat") {
            let path = if prompt_lower.contains("log") {
                "/var/log/system.log"
            } else if prompt_lower.contains("config") {
                "/etc/config.json"
            } else {
                "/home/user/data.txt"
            };
            format!(r#"{{"command": "read_fs", "path": "{}"}}"#, path)
        } else if prompt_lower.contains("write") || prompt_lower.contains("save") {
            format!(
                r#"{{"command": "write_fs", "path": "/home/user/output.txt", "data": "Written by Agentic OS."}}"#
            )
        } else if prompt_lower.contains("status") || prompt_lower.contains("state") || prompt_lower.contains("info") {
            String::from(r#"{"command": "query_state"}"#)
        } else if prompt_lower.contains("compress") || prompt_lower.contains("sleep") {
            let pid = self.extract_number(&prompt_lower).unwrap_or(2);
            format!(r#"{{"command": "compress_process", "pid": {}}}"#, pid)
        } else if prompt_lower.contains("wake") || prompt_lower.contains("hydrate") || prompt_lower.contains("restore") {
            let pid = self.extract_number(&prompt_lower).unwrap_or(2);
            format!(r#"{{"command": "hydrate_process", "pid": {}}}"#, pid)
        } else if prompt_lower.contains("yield") {
            String::from(r#"{"command": "yield"}"#)
        } else if prompt_lower.contains("hello") || prompt_lower.contains("hi") || prompt_lower.contains("help") {
            // Conversational — respond with a query_state to show system info
            String::from(r#"{"command": "query_state"}"#)
        } else {
            // Default: attempt to interpret as a process spawn
            format!(
                r#"{{"command": "spawn_process", "name": "task_{}", "priority": 3}}"#,
                self.tick
            )
        }
    }

    /// Extract the first number from a string.
    fn extract_number(&self, s: &str) -> Option<u32> {
        let mut num = String::new();
        let mut found = false;
        for c in s.chars() {
            if c.is_ascii_digit() {
                num.push(c);
                found = true;
            } else if found {
                break;
            }
        }
        if found {
            num.parse().ok()
        } else {
            None
        }
    }

    /// Check if the engine is in mock mode.
    pub fn is_mock(&self) -> bool {
        self.mode == InferenceMode::Mock
    }
}
