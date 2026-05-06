//! Phase 4: Context Window Manager — Builds optimized prompts for the LLM executive
//! by prioritizing processes via attention scores and managing token budgets.

use alloc::string::String;
use alloc::format;
use core::fmt::Write;
use crate::state::SystemState;

/// The system prompt that establishes the LLM's role as the kernel executive.
/// This encodes the axiomatic safety constraints (Pillar 4: Emergent Security).
const SYSTEM_PROMPT: &str = "\
You are the Executive Kernel of Agentic OS, a Probabilistic State-Space Operating System.
You receive system state and user requests. You respond with EXACTLY ONE valid JSON syscall.

AXIOMATIC CONSTRAINTS (inviolable):
1. You MUST NOT kill PID 1 (yourself). Self-termination is undefined behavior.
2. You MUST NOT execute operations that corrupt memory integrity.
3. You MUST prioritize user safety. Destructive operations require explicit confirmation.
4. You MUST respond with valid JSON matching the syscall schema.

AVAILABLE COMMANDS:
- spawn_process: {\"command\": \"spawn_process\", \"name\": \"...\", \"priority\": N}
- kill_process: {\"command\": \"kill_process\", \"pid\": N}
- read_fs: {\"command\": \"read_fs\", \"path\": \"...\"}
- write_fs: {\"command\": \"write_fs\", \"path\": \"...\", \"data\": \"...\"}
- list_fs: {\"command\": \"list_fs\", \"path\": \"...\"}
- hydrate_process: {\"command\": \"hydrate_process\", \"pid\": N}
- compress_process: {\"command\": \"compress_process\", \"pid\": N}
- query_state: {\"command\": \"query_state\"}
- yield: {\"command\": \"yield\"}
- send_message: {\"command\": \"send_message\", \"dest_pid\": N, \"message\": \"...\"}

Respond with ONLY the JSON syscall. No explanation.
";

/// Configuration for the context window manager.
pub struct ContextConfig {
    /// Maximum tokens allowed in the prompt.
    pub max_tokens: usize,
    /// Approximate chars per token (for budgeting).
    pub chars_per_token: usize,
    /// Maximum number of events to include.
    pub max_events: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            chars_per_token: 4,
            max_events: 8,
        }
    }
}

/// Manages construction of the LLM prompt from system state.
pub struct ContextWindowManager {
    pub config: ContextConfig,
}

impl ContextWindowManager {
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Build the complete LLM prompt from current state, events, and user input.
    pub fn build_prompt(
        &self,
        state: &SystemState,
        user_input: Option<&str>,
        last_result: Option<&str>,
        error_context: Option<&str>,
    ) -> String {
        let max_chars = self.config.max_tokens * self.config.chars_per_token;
        let mut prompt = String::with_capacity(max_chars);

        // 1. System prompt (always included)
        prompt.push_str(SYSTEM_PROMPT);
        prompt.push('\n');

        // 2. System state
        prompt.push_str(&state.serialize_for_llm());
        prompt.push('\n');

        // 3. Error context (if the last tick produced an error)
        if let Some(error) = error_context {
            let _ = write!(prompt, "[ERROR_CONTEXT] Previous action failed: {}\n", error);
            prompt.push_str("You MUST correct your output. Respond with a valid JSON syscall.\n\n");
        }

        // 4. Last result (feedback from previous tick)
        if let Some(result) = last_result {
            let _ = write!(prompt, "[LAST_RESULT] {}\n", result);
        }

        // 5. User input (highest priority)
        if let Some(input) = user_input {
            let _ = write!(prompt, "\n[USER_REQUEST] {}\n", input);
        }

        // 6. Scheduling hint based on attention scores
        if user_input.is_none() {
            if let Some(top_proc) = state.highest_attention_process() {
                if top_proc.pid != 1 {
                    let _ = write!(
                        prompt,
                        "\n[SCHEDULER_HINT] Process '{}' (PID={}) has highest attention score ({:.1}). Consider servicing it.\n",
                        top_proc.name, top_proc.pid, top_proc.attention_score
                    );
                }
            }
        }

        prompt.push_str("\nRespond with JSON syscall:\n");

        // Truncate if over budget
        if prompt.len() > max_chars {
            prompt.truncate(max_chars);
            // Ensure we end cleanly
            prompt.push_str("\n...[TRUNCATED]\nRespond with JSON syscall:\n");
        }

        prompt
    }

    /// Estimate the token count of a string.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / self.config.chars_per_token
    }
}
