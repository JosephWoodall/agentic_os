//! Phase 4: System State — Process management with attention scores, state compression,
//! and event logging for the Probabilistic State-Space OS.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use core::fmt::Write;

/// The lifecycle state of a process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessState {
    /// Process is actively running and in the LLM's context window.
    Active,
    /// Process state has been compressed to latent space (dormant).
    Compressed,
    /// Process is temporarily suspended (waiting for I/O, etc.)
    Suspended,
    /// Process has terminated.
    Terminated,
}

impl core::fmt::Display for ProcessState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProcessState::Active => write!(f, "ACTIVE"),
            ProcessState::Compressed => write!(f, "COMPRESSED"),
            ProcessState::Suspended => write!(f, "SUSPENDED"),
            ProcessState::Terminated => write!(f, "TERMINATED"),
        }
    }
}

/// A process in the Agentic OS.
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u32,
    pub name: String,
    /// The current lifecycle state.
    pub state: ProcessState,
    /// Textual history of actions performed by/on this process.
    pub history: String,
    /// Bayesian attention score (Pillar 1: Compute as Inference).
    /// Higher values mean the executive should focus more on this process.
    pub attention_score: f32,
    /// Priority level (user-assigned or LLM-assigned).
    pub priority: u32,
    /// Compressed latent representation of the process state (Pillar 2).
    pub compressed_state: Option<Vec<f32>>,
    /// Number of ticks since last attention.
    pub ticks_since_attention: u32,
}

impl Process {
    pub fn new(pid: u32, name: &str, priority: u32) -> Self {
        Self {
            pid,
            name: String::from(name),
            state: ProcessState::Active,
            history: String::from("Process spawned."),
            attention_score: 1.0, // Start with neutral attention
            priority,
            compressed_state: None,
            ticks_since_attention: 0,
        }
    }

    /// Append an event to the process history.
    pub fn log_event(&mut self, event: &str) {
        self.history.push('\n');
        self.history.push_str(event);
        // Keep history bounded
        if self.history.len() > 512 {
            let truncated = &self.history[self.history.len() - 400..];
            self.history = format!("...{}", truncated);
        }
    }

    /// Update the Bayesian attention score based on activity.
    pub fn update_attention(&mut self, received_attention: bool) {
        if received_attention {
            self.ticks_since_attention = 0;
            // Decay attention slightly after being serviced
            self.attention_score *= 0.9;
        } else {
            self.ticks_since_attention += 1;
            // Attention grows with neglect (Bayesian prior update)
            self.attention_score += 0.1 * (self.priority as f32 + 1.0);
            // Cap at a maximum
            if self.attention_score > 100.0 {
                self.attention_score = 100.0;
            }
        }
    }
}

/// A system event for the event log.
#[derive(Debug, Clone)]
pub struct SystemEvent {
    pub tick: u64,
    pub event_type: EventType,
    pub message: String,
}

/// Types of system events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    ProcessSpawn,
    ProcessKill,
    ProcessCompress,
    ProcessHydrate,
    SyscallOk,
    SyscallError,
    UserInput,
    LlmOutput,
    SecurityViolation,
}

impl core::fmt::Display for EventType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EventType::ProcessSpawn => write!(f, "SPAWN"),
            EventType::ProcessKill => write!(f, "KILL"),
            EventType::ProcessCompress => write!(f, "COMPRESS"),
            EventType::ProcessHydrate => write!(f, "HYDRATE"),
            EventType::SyscallOk => write!(f, "SYSCALL_OK"),
            EventType::SyscallError => write!(f, "SYSCALL_ERR"),
            EventType::UserInput => write!(f, "USER_INPUT"),
            EventType::LlmOutput => write!(f, "LLM_OUTPUT"),
            EventType::SecurityViolation => write!(f, "SECURITY"),
        }
    }
}

/// The global system state containing all processes and the event log.
pub struct SystemState {
    pub processes: Vec<Process>,
    pub next_pid: u32,
    /// Ring buffer of recent system events.
    pub event_log: VecDeque<SystemEvent>,
    /// Maximum events to retain.
    pub max_events: usize,
    /// Current tick number.
    pub current_tick: u64,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            processes: Vec::new(),
            next_pid: 1,
            event_log: VecDeque::with_capacity(64),
            max_events: 64,
            current_tick: 0,
        }
    }

    /// Spawn a new process and return its PID.
    pub fn spawn(&mut self, name: &str) -> u32 {
        self.spawn_with_priority(name, 0)
    }

    /// Spawn a new process with a given priority.
    pub fn spawn_with_priority(&mut self, name: &str, priority: u32) -> u32 {
        let pid = self.next_pid;
        self.next_pid += 1;
        self.processes.push(Process::new(pid, name, priority));
        self.log_event(EventType::ProcessSpawn, format!("PID {} '{}' spawned", pid, name));
        pid
    }

    /// Kill a process by PID. Returns true if found.
    pub fn kill(&mut self, pid: u32) -> bool {
        if let Some(proc) = self.processes.iter_mut().find(|p| p.pid == pid) {
            proc.state = ProcessState::Terminated;
            proc.log_event("Process terminated.");
            self.log_event(EventType::ProcessKill, format!("PID {} killed", pid));
            true
        } else {
            false
        }
    }

    /// Compress a process's state into latent space.
    pub fn compress(&mut self, pid: u32, latent: Vec<f32>) -> bool {
        if let Some(proc) = self.processes.iter_mut().find(|p| p.pid == pid) {
            proc.state = ProcessState::Compressed;
            proc.compressed_state = Some(latent);
            proc.log_event("State compressed to latent space.");
            self.log_event(EventType::ProcessCompress, format!("PID {} compressed", pid));
            true
        } else {
            false
        }
    }

    /// Hydrate a process from compressed state back to active.
    pub fn hydrate(&mut self, pid: u32) -> bool {
        if let Some(proc) = self.processes.iter_mut().find(|p| p.pid == pid) {
            if proc.state == ProcessState::Compressed {
                proc.state = ProcessState::Active;
                proc.compressed_state = None;
                proc.log_event("State hydrated from latent space.");
                self.log_event(EventType::ProcessHydrate, format!("PID {} hydrated", pid));
                return true;
            }
        }
        false
    }

    /// Get the process with the highest attention score (for scheduling).
    pub fn highest_attention_process(&self) -> Option<&Process> {
        self.processes
            .iter()
            .filter(|p| p.state == ProcessState::Active)
            .max_by(|a, b| a.attention_score.partial_cmp(&b.attention_score).unwrap_or(core::cmp::Ordering::Equal))
    }

    /// Update attention scores for all active processes.
    pub fn update_all_attention(&mut self, focused_pid: Option<u32>) {
        for proc in self.processes.iter_mut() {
            if proc.state == ProcessState::Active {
                let focused = focused_pid == Some(proc.pid);
                proc.update_attention(focused);
            }
        }
    }

    /// Get active process count.
    pub fn active_count(&self) -> usize {
        self.processes.iter().filter(|p| p.state == ProcessState::Active).count()
    }

    /// Log a system event.
    pub fn log_event(&mut self, event_type: EventType, message: String) {
        if self.event_log.len() >= self.max_events {
            self.event_log.pop_front();
        }
        self.event_log.push_back(SystemEvent {
            tick: self.current_tick,
            event_type,
            message,
        });
    }

    /// Serialize system state into a string for the LLM context window.
    pub fn serialize_for_llm(&self) -> String {
        let mut s = String::from("[SYSTEM_STATE]\n");
        let _ = write!(s, "Tick: {} | Active Processes: {} | Total: {}\n",
            self.current_tick, self.active_count(), self.processes.len());

        // Active processes (sorted by attention score)
        let mut active: Vec<&Process> = self.processes.iter()
            .filter(|p| p.state == ProcessState::Active)
            .collect();
        active.sort_by(|a, b| b.attention_score.partial_cmp(&a.attention_score).unwrap_or(core::cmp::Ordering::Equal));

        for p in &active {
            let _ = write!(s, "PID={} | NAME={} | ATTN={:.1} | PRI={} | {}\n",
                p.pid, p.name, p.attention_score, p.priority,
                if p.history.len() > 60 { &p.history[p.history.len()-57..] } else { &p.history });
        }

        // Compressed processes (just names)
        let compressed: Vec<&Process> = self.processes.iter()
            .filter(|p| p.state == ProcessState::Compressed)
            .collect();
        if !compressed.is_empty() {
            s.push_str("[DORMANT] ");
            for p in &compressed {
                let _ = write!(s, "PID={}({}) ", p.pid, p.name);
            }
            s.push('\n');
        }

        // Recent events
        let recent_events: Vec<&SystemEvent> = self.event_log.iter().rev().take(5).collect();
        if !recent_events.is_empty() {
            s.push_str("[RECENT_EVENTS]\n");
            for event in recent_events.iter().rev() {
                let _ = write!(s, "T{} [{}] {}\n", event.tick, event.event_type, event.message);
            }
        }

        s
    }

    /// Find a process by PID.
    pub fn find_process(&self, pid: u32) -> Option<&Process> {
        self.processes.iter().find(|p| p.pid == pid)
    }

    /// Find a mutable process by PID.
    pub fn find_process_mut(&mut self, pid: u32) -> Option<&mut Process> {
        self.processes.iter_mut().find(|p| p.pid == pid)
    }

    /// Clean up terminated processes.
    pub fn gc_terminated(&mut self) {
        self.processes.retain(|p| p.state != ProcessState::Terminated);
    }
}
