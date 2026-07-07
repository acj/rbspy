/// Core types used throughout rbspy: StackFrame and StackTrace
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;
use std::{self, convert::From};

use anyhow::{Error, Result};
use clap::ValueEnum;
use remoteprocess::Pid;
use thiserror::Error;

use crate::core::process::Process;
use crate::ui::*;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub(crate) struct Header {
    pub sample_rate: Option<u32>,
    pub rbspy_version: Option<String>,
    pub start_time: Option<SystemTime>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub name: String,
    pub relative_path: String,
    pub absolute_path: Option<String>,
    pub lineno: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct StackTrace {
    pub trace: Vec<StackFrame>,
    pub pid: Option<Pid>,
    pub thread_id: Option<usize>,
    pub time: Option<SystemTime>,
    pub on_cpu: Option<bool>,
}

pub type StackTraceFn = fn(
    usize,
    usize,
    Option<usize>,
    &Process,
    Pid,
    bool,
    &StackScannerCache,
) -> Result<Option<StackTrace>>;

pub type IsMaybeThreadFn = fn(usize, usize, &Process, &[proc_maps::MapRange]) -> bool;

pub type GetExecutionContextFn = fn(usize, usize, &Process, &StackScannerCache) -> Result<usize>;

/// State that is expensive to discover but stable for the lifetime of the profiled process.
/// Keeping it here lets the per-sample stack scanners avoid re-discovering it on every sample,
/// which reduces the number of cross-process memory reads rbspy makes (and hence its CPU usage).
#[derive(Debug, Default)]
pub struct StackScannerCache {
    // Address of the word inside the main ractor struct that holds the pointer to the current
    // execution context (ruby 3+). The word's address is stable, but the pointer stored in it
    // changes as the target switches threads, so it must be re-read on every sample.
    ec_pointer_slot: Cell<Option<usize>>,
    // Resolved C function names, keyed by (method entry address, method definition address,
    // method id). Method ids are never reused within a process, so entries stay valid even if
    // the method entry object is garbage collected.
    cfunc_names: RefCell<HashMap<(usize, usize, usize), String>>,
    // Resolved class paths, keyed by (method entry address, method definition address). Used on
    // ruby 3.3+, where every method frame's name is prefixed with the owning class's path.
    classpaths: RefCell<HashMap<(usize, usize), (String, bool)>>,
    // Fully resolved frame details, keyed by (iseq address, method entry address, method
    // definition address). Everything about a frame except its line number is stable for that
    // triple, so on a cache hit only the line number has to be re-resolved.
    frames: RefCell<HashMap<(usize, usize, usize), CachedFrameInfo>>,
}

/// Frame details that are stable for a given (iseq, method entry, method definition) triple,
/// cached so that a frame only has to be fully resolved the first time it is seen.
#[derive(Debug, Clone)]
pub struct CachedFrameInfo {
    pub name: String,
    pub relative_path: String,
    pub absolute_path: Option<String>,
    /// Address of the iseq constant body that line numbers are resolved from. It also guards
    /// against stale entries: it must match the body pointer freshly read from the control
    /// frame's iseq on every sample, so a replaced iseq forces a full re-resolution.
    pub body_ptr: usize,
}

impl StackScannerCache {
    // Far more entries than any real process's method or iseq count; bounds memory use if a
    // target somehow generates method entries or iseqs without limit.
    const MAX_ENTRIES_PER_CACHE: usize = 100_000;

    pub fn ec_pointer_slot(&self) -> Option<usize> {
        self.ec_pointer_slot.get()
    }

    pub fn set_ec_pointer_slot(&self, slot: usize) {
        self.ec_pointer_slot.set(Some(slot));
    }

    pub fn clear_ec_pointer_slot(&self) {
        self.ec_pointer_slot.set(None);
    }

    pub fn cfunc_name(&self, key: &(usize, usize, usize)) -> Option<String> {
        self.cfunc_names.borrow().get(key).cloned()
    }

    pub fn store_cfunc_name(&self, key: (usize, usize, usize), name: &str) {
        let mut names = self.cfunc_names.borrow_mut();
        if names.len() < Self::MAX_ENTRIES_PER_CACHE {
            names.insert(key, name.to_string());
        }
    }

    pub fn classpath(&self, key: &(usize, usize)) -> Option<(String, bool)> {
        self.classpaths.borrow().get(key).cloned()
    }

    pub fn store_classpath(&self, key: (usize, usize), classpath: &str, singleton: bool) {
        let mut classpaths = self.classpaths.borrow_mut();
        if classpaths.len() < Self::MAX_ENTRIES_PER_CACHE {
            classpaths.insert(key, (classpath.to_string(), singleton));
        }
    }

    pub fn frame(&self, key: &(usize, usize, usize)) -> Option<CachedFrameInfo> {
        self.frames.borrow().get(key).cloned()
    }

    pub fn store_frame(&self, key: (usize, usize, usize), info: CachedFrameInfo) {
        let mut frames = self.frames.borrow_mut();
        if frames.len() < Self::MAX_ENTRIES_PER_CACHE {
            frames.insert(key, info);
        }
    }
}

#[derive(Error, Debug)]
pub enum MemoryCopyError {
    #[error("The operation completed successfully")]
    OperationSucceeded,
    #[error("Permission denied when reading from process. If you're not running as root, try again with sudo. If you're using Docker, try passing `--cap-add=SYS_PTRACE` to `docker run`")]
    PermissionDenied,
    #[error("Failed to copy memory address {:x}", _0)]
    Io(usize, std::io::Error),
    #[error("Process isn't running")]
    ProcessEnded,
    #[error("Copy error: {}", _0)]
    Message(String),
    #[error("Tried to read invalid memory address {:x}", _0)]
    InvalidAddressError(usize),
}

impl StackFrame {
    pub fn path(&self) -> &str {
        match self.absolute_path {
            Some(ref p) => p.as_ref(),
            None => self.relative_path.as_ref(),
        }
    }

    // we use this stack frame when there's a C function that we don't recognize in the stack. This
    // would be a constant but it has strings in it so it can't be.
    pub fn unknown_c_function() -> StackFrame {
        StackFrame {
            name: "(unknown) [c function]".to_string(),
            relative_path: "(unknown)".to_string(),
            absolute_path: None,
            lineno: None,
        }
    }
}

impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lineno = match self.lineno {
            Some(lineno) => format!(":{}", lineno.to_string()),
            None => "".to_string(),
        };
        write!(f, "{} - {}{}", self.name, self.path(), lineno)
    }
}

impl Ord for StackFrame {
    fn cmp(&self, other: &StackFrame) -> Ordering {
        self.path()
            .cmp(other.path())
            .then(self.name.cmp(&other.name))
            .then(self.lineno.cmp(&other.lineno))
    }
}

impl PartialOrd for StackFrame {
    fn partial_cmp(&self, other: &StackFrame) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl StackTrace {
    pub fn new_empty() -> StackTrace {
        StackTrace {
            pid: None,
            trace: Vec::new(),
            thread_id: None,
            time: None,
            on_cpu: None,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, StackFrame> {
        self.trace.iter()
    }
}

impl fmt::Display for StackTrace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let frames: Vec<String> = self.iter().rev().map(|s| s.to_string()).collect();
        write!(f, "{}", frames.join("\n"))
    }
}

impl From<Error> for MemoryCopyError {
    fn from(error: Error) -> Self {
        let addr = *error.downcast_ref::<usize>().unwrap_or(&0);
        let error = std::io::Error::last_os_error();

        if error.kind() == std::io::ErrorKind::PermissionDenied {
            return MemoryCopyError::PermissionDenied;
        }

        match error.raw_os_error() {
            // Sometimes Windows returns this error code
            Some(0) => MemoryCopyError::OperationSucceeded,
            // On *nix EFAULT means that the address was invalid
            Some(14) => MemoryCopyError::InvalidAddressError(addr),
            _ => MemoryCopyError::Io(addr, error),
        }
    }
}

/// File formats into which rbspy can convert its recorded traces

// The values of this enum get translated directly to command line arguments. Make them
// lowercase so that we don't have camelcase command line arguments
#[derive(ValueEnum, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[allow(non_camel_case_types)]
pub enum OutputFormat {
    flamegraph,
    collapsed,
    callgrind,
    speedscope,
    pprof,
    summary,
    summary_by_line,
}

impl OutputFormat {
    pub fn outputter(self, flame_min_width: f64) -> Box<dyn output::Outputter> {
        match self {
            OutputFormat::flamegraph => Box::new(output::Flamegraph::new(flame_min_width)),
            OutputFormat::collapsed => Box::new(output::Collapsed::default()),
            OutputFormat::callgrind => Box::new(output::Callgrind(callgrind::Stats::new())),
            OutputFormat::speedscope => Box::new(output::Speedscope(speedscope::Stats::new())),
            OutputFormat::pprof => Box::new(output::Pprof(pprof::Stats::new())),
            OutputFormat::summary => Box::new(output::Summary(summary::Stats::new())),
            OutputFormat::summary_by_line => Box::new(output::SummaryLine(summary::Stats::new())),
        }
    }

    pub fn extension(&self) -> String {
        match *self {
            OutputFormat::flamegraph => "flamegraph.svg",
            OutputFormat::collapsed => "collapsed.txt",
            OutputFormat::callgrind => "callgrind.txt",
            OutputFormat::speedscope => "speedscope.json",
            OutputFormat::pprof => "profile.pb.gz",
            OutputFormat::summary => "summary.txt",
            OutputFormat::summary_by_line => "summary_by_line.txt",
        }
        .to_string()
    }

    pub fn possible_values() -> impl Iterator<Item = clap::builder::PossibleValue> {
        Self::value_variants()
            .iter()
            .filter_map(ValueEnum::to_possible_value)
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "flamegraph" => Ok(OutputFormat::flamegraph),
            "collapsed" => Ok(OutputFormat::collapsed),
            "callgrind" => Ok(OutputFormat::callgrind),
            "speedscope" => Ok(OutputFormat::speedscope),
            "pprof" => Ok(OutputFormat::pprof),
            "summary" => Ok(OutputFormat::summary),
            "summary-by-line" => Ok(OutputFormat::summary_by_line),
            _ => Err(anyhow::format_err!("Unknown output format: {}", s)),
        }
    }
}
