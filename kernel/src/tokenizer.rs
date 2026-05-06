//! Phase 2: Tokenizer — BPE/SentencePiece tokenizer for Gemma models.
//!
//! Loads vocabulary from GGUF metadata and provides encode/decode.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A token entry from the vocabulary.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub text: String,
    pub score: f32,
    pub token_type: u32,
}

/// BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token ID → text mapping.
    vocab: Vec<TokenEntry>,
    /// Text → token ID mapping for single-token lookups.
    token_to_id: BTreeMap<String, u32>,
    /// Special token IDs.
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
}

impl Tokenizer {
    /// Create a new tokenizer with the given vocabulary.
    pub fn new(vocab: Vec<TokenEntry>) -> Self {
        let mut token_to_id = BTreeMap::new();
        for (i, entry) in vocab.iter().enumerate() {
            token_to_id.insert(entry.text.clone(), i as u32);
        }

        // Default special tokens (Gemma conventions)
        let bos_token = token_to_id.get("<bos>").copied().unwrap_or(2);
        let eos_token = token_to_id.get("<eos>").copied().unwrap_or(1);
        let pad_token = token_to_id.get("<pad>").copied().unwrap_or(0);

        Self {
            vocab,
            token_to_id,
            bos_token,
            eos_token,
            pad_token,
        }
    }

    /// Create a minimal mock tokenizer for testing without real weights.
    pub fn mock() -> Self {
        let mut vocab = Vec::new();
        // Build a basic character-level + common word vocabulary
        let entries = [
            ("<pad>", 0.0, 3),   // 0
            ("<eos>", 0.0, 3),   // 1
            ("<bos>", 0.0, 3),   // 2
            ("<unk>", 0.0, 3),   // 3
            (" ", -1.0, 1),      // 4
            ("\n", -1.0, 1),     // 5
            ("{", -1.0, 1),      // 6
            ("}", -1.0, 1),      // 7
            ("\"", -1.0, 1),     // 8
            (":", -1.0, 1),      // 9
            (",", -1.0, 1),      // 10
            ("command", -2.0, 1), // 11
            ("spawn", -2.0, 1),  // 12
            ("_", -1.0, 1),      // 13
            ("process", -2.0, 1), // 14
            ("name", -2.0, 1),   // 15
            ("priority", -2.0, 1), // 16
            ("hello", -2.0, 1),  // 17
            ("yield", -2.0, 1),  // 18
            ("kill", -2.0, 1),   // 19
            ("read", -2.0, 1),   // 20
            ("write", -2.0, 1),  // 21
            ("query", -2.0, 1),  // 22
            ("state", -2.0, 1),  // 23
        ];

        // Add ASCII printable characters (32-126)
        for (text, score, ttype) in &entries {
            vocab.push(TokenEntry {
                text: String::from(*text),
                score: *score,
                token_type: *ttype,
            });
        }

        // Pad to at least 128 tokens with single chars
        for c in b'a'..=b'z' {
            let s = String::from(c as char);
            if !entries.iter().any(|(t, _, _)| *t == s.as_str()) {
                vocab.push(TokenEntry {
                    text: s,
                    score: -3.0,
                    token_type: 1,
                });
            }
        }
        for c in b'0'..=b'9' {
            vocab.push(TokenEntry {
                text: String::from(c as char),
                score: -3.0,
                token_type: 1,
            });
        }

        Self::new(vocab)
    }

    /// Encode text into token IDs using greedy longest-match.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id = 3; // <unk>

            // Try longest match first
            for end in (i + 1..=chars.len()).rev() {
                let substr: String = chars[i..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&substr) {
                    best_len = end - i;
                    best_id = id;
                    break;
                }
            }

            if best_len == 0 {
                // Single character fallback
                tokens.push(3); // <unk>
                i += 1;
            } else {
                tokens.push(best_id);
                i += best_len;
            }
        }

        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &id in tokens {
            if (id as usize) < self.vocab.len() {
                let entry = &self.vocab[id as usize];
                // Skip special tokens in output
                if entry.token_type != 3 {
                    text.push_str(&entry.text);
                }
            }
        }
        text
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
