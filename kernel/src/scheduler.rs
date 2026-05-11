//! Phase 5: Executive Loop — The kernel scheduler that connects the LLM inference engine
//! to hardware state in a continuous tick-based loop.
//!
//! Implements the full gather → serialize → infer → parse → dispatch → update pipeline.

use alloc::string::String;
use alloc::format;
use crate::state::{SystemState, EventType, ProcessState};
use crate::syscall::{Syscall, Dispatcher, SyscallResult, SyscallError};
use crate::inference::InferenceEngine;
use crate::context::{ContextWindowManager, ContextConfig};
use crate::grammar::GrammarConstraint;
use crate::compressor::StateCompressor;

/// The executive loop — the heart of the Probabilistic State-Space OS.
pub struct ExecutiveLoop {
    /// The global system state (all processes, events).
    pub state: SystemState,
    /// The LLM inference engine.
    pub inference: InferenceEngine,
    /// The context window manager.
    pub context_mgr: ContextWindowManager,
    /// Tick counter.
    pub tick_count: u64,
    /// Pending user requests (from the shell).
    pub pending_requests: alloc::vec::Vec<String>,
    /// The result string from the last tick (for feedback).
    last_result: Option<String>,
    /// Error context from the last tick (for self-correction).
    error_context: Option<String>,
    /// Maximum tokens for LLM generation.
    pub max_gen_tokens: usize,
    /// Consecutive error count (for circuit breaking).
    consecutive_errors: u32,
}

impl ExecutiveLoop {
    /// Create a new executive loop.
    pub fn new() -> Self {
        let mut state = SystemState::new();
        // PID 1: The Executive itself
        state.spawn_with_priority("Executive (PID 1)", 10);

        let inference = InferenceEngine::mock();
        let context_mgr = ContextWindowManager::new(ContextConfig::default());

        Self {
            state,
            inference,
            context_mgr,
            tick_count: 0,
            pending_requests: alloc::vec::Vec::new(),
            last_result: None,
            error_context: None,
            max_gen_tokens: 256,
            consecutive_errors: 0,
        }
    }

    /// Create an executive loop with a real or mock inference engine.
    pub fn with_inference(inference: InferenceEngine) -> Self {
        let mut s = Self::new();
        s.inference = inference;
        s
    }

    /// Submit a user request to be processed on the next tick.
    pub fn submit_request(&mut self, request: String) {
        self.state.log_event(
            EventType::UserInput,
            format!("User: {}", &request[..request.len().min(80)]),
        );
        self.pending_requests.push(request);
    }

    /// Execute one tick of the executive loop.
    ///
    /// Returns the result string for display/logging.
    pub fn tick(&mut self, compositor: &mut crate::compositor::Compositor) -> String {
        self.tick_count += 1;
        self.state.current_tick = self.tick_count;

        log::info!("=== EXECUTIVE TICK {} ===", self.tick_count);

        // ---- 1. GATHER: Collect pending requests and state ----
        let user_input = if !self.pending_requests.is_empty() {
            Some(self.pending_requests.remove(0))
        } else {
            None
        };

        // ---- 2. SERIALIZE: Build LLM prompt from system state ----
        let prompt = self.context_mgr.build_prompt(
            &self.state,
            user_input.as_deref(),
            self.last_result.as_deref(),
            self.error_context.as_deref(),
        );

        log::info!("CONTEXT: {} chars (~{} tokens)",
            prompt.len(), self.context_mgr.estimate_tokens(&prompt));

        // ---- 3. INFER: Run the inference engine ----
        let raw_output = self.inference.generate(&prompt, self.max_gen_tokens);
        log::info!("LLM OUTPUT: {}", &raw_output[..raw_output.len().min(200)]);

        self.state.log_event(
            EventType::LlmOutput,
            format!("LLM: {}", &raw_output[..raw_output.len().min(80)]),
        );

        // ---- 4. VALIDATE: Check grammar constraints ----
        let json_str = Self::extract_json(&raw_output);

        if !GrammarConstraint::validate(&json_str) {
            log::warn!("Grammar validation failed for: {}", json_str);
            // Don't hard-fail; the JSON parser may still succeed
        }

        // ---- 5. PARSE & DISPATCH ----
        let result = self.execute_syscall(&json_str, user_input.as_deref(), compositor);

        // ---- 6. UPDATE: Feed results back and update state ----
        let result_str = result.to_context_string();
        log::info!("RESULT: {}", result_str);

        // Update attention scores
        let focused_pid = match &result {
            SyscallResult::ProcessSpawned(_) | SyscallResult::ProcessKilled(_) => None,
            _ => user_input.as_ref().and_then(|_| Some(1u32)),
        };
        self.state.update_all_attention(focused_pid);

        // Track errors for self-correction
        match &result {
            SyscallResult::Error(_) => {
                self.consecutive_errors += 1;
                self.error_context = Some(result_str.clone());
                if self.consecutive_errors >= 3 {
                    log::error!("Circuit breaker: {} consecutive errors. Resetting error context.", self.consecutive_errors);
                    self.error_context = None;
                    self.consecutive_errors = 0;
                }
            }
            _ => {
                self.consecutive_errors = 0;
                self.error_context = None;
            }
        }

        self.last_result = Some(result_str.clone());

        // Periodic GC of terminated processes
        if self.tick_count % 10 == 0 {
            self.state.gc_terminated();
        }

        result_str
    }

    /// Execute a syscall from JSON, handling side effects on SystemState.
    fn execute_syscall(&mut self, json: &str, _user_input: Option<&str>, compositor: &mut crate::compositor::Compositor) -> SyscallResult {
        let result = Dispatcher::parse_and_dispatch(json);

        // Apply side effects to SystemState based on the result
        match &result {
            SyscallResult::ProcessSpawned(_) => {
                // Parse the syscall again to get the name/priority
                if let Ok((syscall, _)) = serde_json_core::from_str::<Syscall>(json) {
                    let name = syscall.name.unwrap_or("unnamed");
                    let priority = syscall.priority.unwrap_or(0);
                    let pid = self.state.spawn_with_priority(name, priority);
                    
                    // Create a physical window for this process
                    let offset = (pid as usize * 30) % 200;
                    compositor.create_window(
                        name,
                        20 + offset, // X offset (left side of screen)
                        20 + offset, // Y offset
                        400,         // Width
                        300,         // Height
                    );
                    
                    return SyscallResult::ProcessSpawned(pid);
                }
            }
            SyscallResult::UiUpdated(_) => {
                 if let Ok((syscall, _)) = serde_json_core::from_str::<Syscall>(json) {
                    if syscall.command == "create_window" {
                        let name = syscall.name.unwrap_or("Window");
                        let wid = compositor.create_window(name, 50, 50, 400, 300);
                        return SyscallResult::UiUpdated(wid);
                    } else if syscall.command == "destroy_window" {
                        if let Some(wid) = syscall.window_id {
                            compositor.destroy_window(wid);
                            return SyscallResult::UiUpdated(wid);
                        }
                    }
                 }
            }
            SyscallResult::ProcessKilled(pid) => {
                let pid = *pid;
                if !self.state.kill(pid) {
                    return SyscallResult::Error(SyscallError::ProcessNotFound(pid));
                }
            }
            SyscallResult::ProcessCompressed(pid) => {
                let pid = *pid;
                if let Some(proc) = self.state.find_process(pid) {
                    let latent = StateCompressor::compress(&proc.history);
                    if !self.state.compress(pid, latent) {
                        return SyscallResult::Error(SyscallError::ProcessNotFound(pid));
                    }
                } else {
                    return SyscallResult::Error(SyscallError::ProcessNotFound(pid));
                }
            }
            SyscallResult::ProcessHydrated(pid) => {
                let pid = *pid;
                if !self.state.hydrate(pid) {
                    return SyscallResult::Error(SyscallError::ProcessNotFound(pid));
                }
            }
            SyscallResult::StateInfo(_) => {
                // Return the actual serialized state
                return SyscallResult::StateInfo(self.state.serialize_for_llm());
            }
            SyscallResult::FileData(_) => {
                if let Ok((syscall, _)) = serde_json_core::from_str::<Syscall>(json) {
                    if syscall.command == "read_fs" {
                        let path = syscall.path.unwrap_or("/");
                        if let Some(data) = self.state.vfs.get(path) {
                            return SyscallResult::FileData(data.clone());
                        } else {
                            return SyscallResult::Error(SyscallError::ImpossibleOperation(format!("File not found: {}", path)));
                        }
                    } else if syscall.command == "write_fs" {
                        let path = syscall.path.unwrap_or("/tmp/output");
                        let data = syscall.data.unwrap_or("");
                        self.state.vfs.insert(alloc::string::String::from(path), alloc::string::String::from(data));
                        return SyscallResult::Ok(format!("Wrote {} bytes to {}", data.len(), path));
                    } else if syscall.command == "list_fs" {
                        let mut files = alloc::string::String::new();
                        for key in self.state.vfs.keys() {
                            files.push_str(key);
                            files.push('\n');
                        }
                        if files.is_empty() {
                            return SyscallResult::FileData(alloc::string::String::from("No files found."));
                        }
                        return SyscallResult::FileData(files);
                    }
                }
            }
            _ => {}
        }

        result
    }

    /// Extract JSON from LLM output (handles cases where the model wraps JSON in text).
    fn extract_json(output: &str) -> String {
        let trimmed = output.trim();

        // If it starts with {, find the matching }
        if let Some(start) = trimmed.find('{') {
            let mut depth = 0;
            let mut in_string = false;
            let mut escape = false;

            for (i, c) in trimmed[start..].char_indices() {
                if escape {
                    escape = false;
                    continue;
                }
                if c == '\\' && in_string {
                    escape = true;
                    continue;
                }
                if c == '"' {
                    in_string = !in_string;
                    continue;
                }
                if !in_string {
                    if c == '{' {
                        depth += 1;
                    } else if c == '}' {
                        depth -= 1;
                        if depth == 0 {
                            return String::from(&trimmed[start..start + i + 1]);
                        }
                    }
                }
            }
        }

        // Fallback: return the whole thing
        String::from(trimmed)
    }

    /// Get the current tick count.
    pub fn ticks(&self) -> u64 {
        self.tick_count
    }
}
