//! Phase 3: Syscall DSL — The semantic intent substrate replacing POSIX APIs.
//!
//! Defines the full vocabulary of system operations that the LLM can invoke.
//! The dispatcher routes parsed JSON syscalls to hardware-level kernel functions.

use serde::{Deserialize, Serialize};
use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;

/// A system call command issued by the LLM executive.
#[derive(Debug, Deserialize, Serialize)]
pub struct Syscall<'a> {
    pub command: &'a str,

    // Process management
    #[serde(default)]
    pub name: Option<&'a str>,
    #[serde(default)]
    pub priority: Option<u32>,
    #[serde(default)]
    pub pid: Option<u32>,

    // Filesystem operations
    #[serde(default)]
    pub path: Option<&'a str>,
    #[serde(default)]
    pub data: Option<&'a str>,

    // Context/memory management
    #[serde(default)]
    pub context_size: Option<usize>,
    #[serde(default)]
    pub target: Option<&'a str>,

    // UI operations (Phase 7)
    #[serde(default)]
    pub ui_tree: Option<&'a str>,
    #[serde(default)]
    pub window_id: Option<u32>,

    // Inter-process communication
    #[serde(default)]
    pub message: Option<&'a str>,
    #[serde(default)]
    pub dest_pid: Option<u32>,

    // Error reporting
    #[serde(default)]
    pub error: Option<&'a str>,
}

/// The result of executing a syscall.
#[derive(Debug)]
pub enum SyscallResult {
    /// Operation succeeded with an optional message.
    Ok(String),
    /// A new process was spawned, returning its PID.
    ProcessSpawned(u32),
    /// Process was killed.
    ProcessKilled(u32),
    /// Filesystem read returned data.
    FileData(String),
    /// State query result.
    StateInfo(String),
    /// Process was compressed into latent space.
    ProcessCompressed(u32),
    /// Process was hydrated from latent space.
    ProcessHydrated(u32),
    /// UI was rendered/updated.
    UiUpdated(u32),
    /// An error occurred.
    Error(SyscallError),
}

/// Errors that can occur during syscall dispatch.
#[derive(Debug)]
pub enum SyscallError {
    /// The command is not recognized.
    UnknownCommand(String),
    /// A required parameter was missing.
    MissingParameter(String),
    /// The target process doesn't exist.
    ProcessNotFound(u32),
    /// An operation violated security policy.
    SecurityViolation(String),
    /// JSON parse failed.
    ParseError(String),
    /// The operation is logically impossible.
    ImpossibleOperation(String),
}

impl core::fmt::Display for SyscallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SyscallError::UnknownCommand(cmd) => write!(f, "Unknown command: '{}'", cmd),
            SyscallError::MissingParameter(p) => write!(f, "Missing parameter: '{}'", p),
            SyscallError::ProcessNotFound(pid) => write!(f, "Process {} not found", pid),
            SyscallError::SecurityViolation(msg) => write!(f, "Security violation: {}", msg),
            SyscallError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            SyscallError::ImpossibleOperation(msg) => write!(f, "Impossible: {}", msg),
        }
    }
}

impl SyscallResult {
    /// Convert to a human-readable status string for feeding back into LLM context.
    pub fn to_context_string(&self) -> String {
        match self {
            SyscallResult::Ok(msg) => format!("[OK] {}", msg),
            SyscallResult::ProcessSpawned(pid) => format!("[OK] Process spawned with PID {}", pid),
            SyscallResult::ProcessKilled(pid) => format!("[OK] Process {} terminated", pid),
            SyscallResult::FileData(data) => format!("[DATA] {}", data),
            SyscallResult::StateInfo(info) => format!("[STATE] {}", info),
            SyscallResult::ProcessCompressed(pid) => {
                format!("[OK] Process {} compressed to latent space", pid)
            }
            SyscallResult::ProcessHydrated(pid) => {
                format!("[OK] Process {} hydrated into active context", pid)
            }
            SyscallResult::UiUpdated(wid) => format!("[OK] Window {} updated", wid),
            SyscallResult::Error(e) => format!("[ERROR] {}", e),
        }
    }
}

/// The syscall dispatcher — routes parsed commands to kernel operations.
pub struct Dispatcher;

impl Dispatcher {
    /// Parse a JSON string and dispatch the syscall.
    pub fn parse_and_dispatch(json: &str) -> SyscallResult {
        match serde_json_core::from_str::<Syscall>(json) {
            Ok((syscall, _)) => Self::dispatch(syscall),
            Err(_) => SyscallResult::Error(SyscallError::ParseError(
                format!("Failed to parse syscall JSON: {}", &json[..json.len().min(100)]),
            )),
        }
    }

    /// Dispatch a parsed syscall to its handler.
    pub fn dispatch(syscall: Syscall) -> SyscallResult {
        match syscall.command {
            // ---- Process Management ----
            "spawn_process" => {
                let name = syscall.name.unwrap_or("unnamed");
                let priority = syscall.priority.unwrap_or(0);
                log::info!(
                    "DISPATCH: spawn_process '{}' priority={}",
                    name, priority
                );
                // PID assignment is handled by SystemState; return a placeholder
                SyscallResult::ProcessSpawned(0)
            }

            "kill_process" => {
                let pid = match syscall.pid {
                    Some(p) => p,
                    None => {
                        return SyscallResult::Error(SyscallError::MissingParameter(
                            String::from("pid"),
                        ))
                    }
                };

                // Security: prevent killing PID 1 (the executive)
                if pid == 1 {
                    return SyscallResult::Error(SyscallError::SecurityViolation(
                        String::from("Cannot kill PID 1 (Executive). This is an axiomatic constraint."),
                    ));
                }

                log::info!("DISPATCH: kill_process PID={}", pid);
                SyscallResult::ProcessKilled(pid)
            }

            "yield" | "yield_process" => {
                let msg = match syscall.error {
                    Some(e) => format!("Process yielded. (LLM Error: {})", e),
                    None => String::from("Process yielded execution."),
                };
                log::info!("DISPATCH: yield_process - {}", msg);
                SyscallResult::Ok(msg)
            }

            // ---- Filesystem Operations ----
            "read_fs" => {
                let path = syscall.path.unwrap_or("/");
                log::info!("DISPATCH: read_fs '{}'", path);
                // Mock filesystem response
                SyscallResult::FileData(format!(
                    "Contents of '{}': [simulated data]", path
                ))
            }

            "write_fs" => {
                let path = syscall.path.unwrap_or("/tmp/output");
                let data = syscall.data.unwrap_or("");
                log::info!("DISPATCH: write_fs '{}' ({} bytes)", path, data.len());
                SyscallResult::FileData(String::from("mock"))
            }

            "list_fs" => {
                let path = syscall.path.unwrap_or("/");
                log::info!("DISPATCH: list_fs '{}'", path);
                SyscallResult::FileData(format!(
                    "Listing '{}': [etc, home, var, tmp]", path
                ))
            }

            // ---- Context/Memory Management (Phase 4) ----
            "allocate_context" => {
                let size = syscall.context_size.unwrap_or(1024);
                log::info!("DISPATCH: allocate_context size={}", size);
                SyscallResult::Ok(format!("Allocated {} token context window", size))
            }

            "free_context" => {
                let pid = syscall.pid.unwrap_or(0);
                log::info!("DISPATCH: free_context for PID={}", pid);
                SyscallResult::Ok(format!("Freed context for PID {}", pid))
            }

            "hydrate_process" => {
                let pid = match syscall.pid {
                    Some(p) => p,
                    None => {
                        return SyscallResult::Error(SyscallError::MissingParameter(
                            String::from("pid"),
                        ))
                    }
                };
                log::info!("DISPATCH: hydrate_process PID={}", pid);
                SyscallResult::ProcessHydrated(pid)
            }

            "compress_process" => {
                let pid = match syscall.pid {
                    Some(p) => p,
                    None => {
                        return SyscallResult::Error(SyscallError::MissingParameter(
                            String::from("pid"),
                        ))
                    }
                };
                log::info!("DISPATCH: compress_process PID={}", pid);
                SyscallResult::ProcessCompressed(pid)
            }

            // ---- UI Operations (Phase 7) ----
            "render_ui" => {
                let window_id = syscall.window_id.unwrap_or(0);
                log::info!("DISPATCH: render_ui window_id={}", window_id);
                SyscallResult::UiUpdated(window_id)
            }

            "update_ui" => {
                let window_id = syscall.window_id.unwrap_or(0);
                log::info!("DISPATCH: update_ui window_id={}", window_id);
                SyscallResult::UiUpdated(window_id)
            }

            "create_window" => {
                let name = syscall.name.unwrap_or("Window");
                log::info!("DISPATCH: create_window '{}'", name);
                SyscallResult::UiUpdated(0)
            }

            "destroy_window" => {
                let window_id = match syscall.window_id {
                    Some(w) => w,
                    None => {
                        return SyscallResult::Error(SyscallError::MissingParameter(
                            String::from("window_id"),
                        ))
                    }
                };
                log::info!("DISPATCH: destroy_window id={}", window_id);
                SyscallResult::UiUpdated(window_id)
            }

            // ---- Inter-Process Communication ----
            "send_message" => {
                let dest = syscall.dest_pid.unwrap_or(1);
                let msg = syscall.message.unwrap_or("");
                log::info!("DISPATCH: send_message to PID={}: '{}'", dest, msg);
                // Return the actual message so the shell can display it
                SyscallResult::Ok(String::from(msg))
            }

            "say" | "print" | "notify" => {
                let msg = syscall.message.unwrap_or("");
                log::info!("DISPATCH: say '{}'", msg);
                SyscallResult::Ok(String::from(msg))
            }

            // ---- Introspection ----
            "query_state" => {
                log::info!("DISPATCH: query_state");
                SyscallResult::StateInfo(String::from(
                    "System operational. Use 'spawn', 'kill', 'read', 'write' commands.",
                ))
            }

            // ---- Unknown ----
            unknown => {
                log::warn!("DISPATCH: Unknown command '{}'", unknown);
                SyscallResult::Error(SyscallError::UnknownCommand(String::from(unknown)))
            }
        }
    }
}

/// All valid syscall command names, for grammar enforcement (Phase 3).
pub const VALID_COMMANDS: &[&str] = &[
    "spawn_process",
    "kill_process",
    "yield",
    "yield_process",
    "read_fs",
    "write_fs",
    "list_fs",
    "allocate_context",
    "free_context",
    "hydrate_process",
    "compress_process",
    "render_ui",
    "update_ui",
    "create_window",
    "destroy_window",
    "send_message",
    "say",
    "query_state",
];
