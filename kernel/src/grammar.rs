//! Phase 3: Grammar Engine — GBNF-inspired constrained decoding for JSON syscall output.
//!
//! Constrains LLM logit outputs so that only valid JSON matching the syscall DSL
//! can be generated. Implements a state machine that tracks JSON structure.

use alloc::string::String;
use alloc::vec::Vec;
use crate::syscall::VALID_COMMANDS;

/// Grammar state machine states for tracking JSON output structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrammarState {
    /// Expecting the opening `{`
    ExpectObjectStart,
    /// Inside the object, expecting a key string or `}`
    ExpectKeyOrEnd,
    /// Expecting `"` to start a key string
    InKey,
    /// Expecting `:` after a key
    ExpectColon,
    /// Expecting a value (string, number, null, object, array)
    ExpectValue,
    /// Inside a string value
    InStringValue,
    /// Inside a number value
    InNumberValue,
    /// Expecting `,` or `}` after a value
    ExpectCommaOrEnd,
    /// Grammar complete (closed JSON object)
    Complete,
    /// Grammar encountered an error
    Error,
}

/// Tracks which key we're currently populating.
#[derive(Debug, Clone, PartialEq)]
enum CurrentKey {
    None,
    Command,
    StringField(String),
    NumberField(String),
}

/// The grammar constraint engine.
pub struct GrammarConstraint {
    /// Current state of the grammar state machine.
    pub state: GrammarState,
    /// Depth counter for nested objects/arrays.
    depth: usize,
    /// The current key being parsed.
    current_key: CurrentKey,
    /// The "command" value seen so far, for validating against VALID_COMMANDS.
    command_value: String,
    /// Characters accumulated for the current token.
    buffer: String,
    /// Whether the "command" key has been seen.
    has_command: bool,
    /// Number of key-value pairs seen.
    pair_count: usize,
}

impl GrammarConstraint {
    pub fn new() -> Self {
        Self {
            state: GrammarState::ExpectObjectStart,
            depth: 0,
            current_key: CurrentKey::None,
            command_value: String::new(),
            buffer: String::new(),
            has_command: false,
            pair_count: 0,
        }
    }

    /// Reset the grammar for a new generation.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feed a character and advance the state machine.
    /// Returns true if the character is valid in the current state.
    pub fn feed(&mut self, c: char) -> bool {
        match self.state {
            GrammarState::ExpectObjectStart => {
                if c == '{' {
                    self.state = GrammarState::ExpectKeyOrEnd;
                    self.depth = 1;
                    true
                } else if c.is_whitespace() {
                    true // skip whitespace
                } else {
                    self.state = GrammarState::Error;
                    false
                }
            }

            GrammarState::ExpectKeyOrEnd => {
                if c == '"' {
                    self.state = GrammarState::InKey;
                    self.buffer.clear();
                    true
                } else if c == '}' {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.state = GrammarState::Complete;
                    }
                    true
                } else if c.is_whitespace() {
                    true
                } else {
                    self.state = GrammarState::Error;
                    false
                }
            }

            GrammarState::InKey => {
                if c == '"' {
                    // Key is complete
                    let key = self.buffer.clone();
                    if key == "command" {
                        self.current_key = CurrentKey::Command;
                    } else if matches!(
                        key.as_str(),
                        "name" | "path" | "data" | "target" | "message" | "ui_tree"
                    ) {
                        self.current_key = CurrentKey::StringField(key);
                    } else if matches!(
                        key.as_str(),
                        "priority" | "pid" | "context_size" | "window_id" | "dest_pid"
                    ) {
                        self.current_key = CurrentKey::NumberField(key);
                    } else {
                        self.current_key = CurrentKey::StringField(key);
                    }
                    self.state = GrammarState::ExpectColon;
                    true
                } else {
                    self.buffer.push(c);
                    true
                }
            }

            GrammarState::ExpectColon => {
                if c == ':' {
                    self.state = GrammarState::ExpectValue;
                    true
                } else if c.is_whitespace() {
                    true
                } else {
                    self.state = GrammarState::Error;
                    false
                }
            }

            GrammarState::ExpectValue => {
                if c == '"' {
                    self.state = GrammarState::InStringValue;
                    self.buffer.clear();
                    true
                } else if c.is_ascii_digit() || c == '-' {
                    self.state = GrammarState::InNumberValue;
                    self.buffer.clear();
                    self.buffer.push(c);
                    true
                } else if c == 'n' {
                    // Possible "null"
                    self.buffer.clear();
                    self.buffer.push(c);
                    self.state = GrammarState::InNumberValue; // reuse for literal
                    true
                } else if c.is_whitespace() {
                    true
                } else {
                    self.state = GrammarState::Error;
                    false
                }
            }

            GrammarState::InStringValue => {
                if c == '"' {
                    // Value complete
                    if self.current_key == CurrentKey::Command {
                        self.command_value = self.buffer.clone();
                        self.has_command = true;
                    }
                    self.pair_count += 1;
                    self.current_key = CurrentKey::None;
                    self.state = GrammarState::ExpectCommaOrEnd;
                    true
                } else if c == '\\' {
                    // Escape — accept next char unconditionally
                    self.buffer.push(c);
                    true
                } else {
                    self.buffer.push(c);
                    true
                }
            }

            GrammarState::InNumberValue => {
                if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' {
                    self.buffer.push(c);
                    true
                } else if c == 'u' || c == 'l' {
                    // Part of "null"
                    self.buffer.push(c);
                    true
                } else {
                    // Number/literal is complete, process the delimiter
                    self.pair_count += 1;
                    self.current_key = CurrentKey::None;

                    if c == ',' {
                        self.state = GrammarState::ExpectKeyOrEnd;
                        true
                    } else if c == '}' {
                        self.depth -= 1;
                        self.state = if self.depth == 0 {
                            GrammarState::Complete
                        } else {
                            GrammarState::ExpectCommaOrEnd
                        };
                        true
                    } else if c.is_whitespace() {
                        self.state = GrammarState::ExpectCommaOrEnd;
                        true
                    } else {
                        self.state = GrammarState::Error;
                        false
                    }
                }
            }

            GrammarState::ExpectCommaOrEnd => {
                if c == ',' {
                    self.state = GrammarState::ExpectKeyOrEnd;
                    true
                } else if c == '}' {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.state = GrammarState::Complete;
                    }
                    true
                } else if c.is_whitespace() {
                    true
                } else {
                    self.state = GrammarState::Error;
                    false
                }
            }

            GrammarState::Complete | GrammarState::Error => false,
        }
    }

    /// Validate a complete JSON string against the grammar.
    pub fn validate(json: &str) -> bool {
        let mut grammar = Self::new();
        for c in json.chars() {
            if !grammar.feed(c) {
                return false;
            }
        }
        grammar.state == GrammarState::Complete && grammar.has_command
    }

    /// Check if the parsed command is a valid syscall.
    pub fn is_valid_command(&self) -> bool {
        VALID_COMMANDS
            .iter()
            .any(|&cmd| cmd == self.command_value.as_str())
    }

    /// Get the set of valid next characters given the current state.
    /// Used for constrained decoding: mask logits for invalid continuations.
    pub fn valid_next_chars(&self) -> Vec<char> {
        let mut chars = Vec::new();

        match self.state {
            GrammarState::ExpectObjectStart => {
                chars.push('{');
                chars.push(' ');
                chars.push('\n');
            }
            GrammarState::ExpectKeyOrEnd => {
                chars.push('"');
                chars.push('}');
                chars.push(' ');
            }
            GrammarState::InKey => {
                chars.push('"');
                for c in b'a'..=b'z' {
                    chars.push(c as char);
                }
                chars.push('_');
            }
            GrammarState::ExpectColon => {
                chars.push(':');
                chars.push(' ');
            }
            GrammarState::ExpectValue => {
                chars.push('"');
                for c in b'0'..=b'9' {
                    chars.push(c as char);
                }
                chars.push('-');
                chars.push('n'); // null
                chars.push(' ');
            }
            GrammarState::InStringValue => {
                // Almost any char is valid inside a string
                for c in 32u8..=126u8 {
                    chars.push(c as char);
                }
            }
            GrammarState::InNumberValue => {
                for c in b'0'..=b'9' {
                    chars.push(c as char);
                }
                chars.push('.');
                chars.push(',');
                chars.push('}');
                chars.push(' ');
            }
            GrammarState::ExpectCommaOrEnd => {
                chars.push(',');
                chars.push('}');
                chars.push(' ');
            }
            _ => {}
        }

        chars
    }

    /// Whether the grammar has reached a complete, valid state.
    pub fn is_complete(&self) -> bool {
        self.state == GrammarState::Complete
    }

    /// Whether the grammar is in an error state.
    pub fn is_error(&self) -> bool {
        self.state == GrammarState::Error
    }
}
