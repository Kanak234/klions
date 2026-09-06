//! Thin platform layer — the only place with OS-specific code.
//!
//! Declared directly against libc so the toolchain keeps its
//! zero-external-dependency property (DC-003).

use std::sync::atomic::{AtomicBool, Ordering};

/// EIR-009: is the stream a terminal? Governs colour and progress output.
#[cfg(unix)]
pub fn is_tty(fd: i32) -> bool {
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(fd) == 1 }
}

#[cfg(not(unix))]
pub fn is_tty(_fd: i32) -> bool {
    false
}

pub fn stdout_is_tty() -> bool {
    is_tty(1)
}
pub fn stderr_is_tty() -> bool {
    is_tty(2)
}

/// NO_COLOR is honoured whenever it is set to any value (EIR-009).
pub fn no_color_env() -> bool {
    std::env::var_os("NO_COLOR").is_some()
}

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// FR-RT-011: SIGINT sets a flag; the training loop checks it between batches
/// and unwinds cleanly rather than aborting mid-update.
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

pub fn clear_interrupt() {
    INTERRUPTED.store(false, Ordering::Relaxed);
}

#[cfg(unix)]
extern "C" fn handle_sigint(_sig: i32) {
    // Only an atomic store: the sole async-signal-safe operation here.
    INTERRUPTED.store(true, Ordering::Relaxed);
}

#[cfg(unix)]
pub fn install_interrupt_handler() {
    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    const SIGINT: i32 = 2;
    unsafe {
        signal(SIGINT, handle_sigint);
    }
}

#[cfg(not(unix))]
pub fn install_interrupt_handler() {}

/// Monotonic milliseconds since process start, for `time_ms()` and `--time`.
pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Terminal width, used to size progress bars. 80 when unknown.
pub fn terminal_width() -> usize {
    if let Ok(v) = std::env::var("COLUMNS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 20 {
                return n;
            }
        }
    }
    80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupt_flag_round_trips() {
        clear_interrupt();
        assert!(!interrupted());
        INTERRUPTED.store(true, Ordering::Relaxed);
        assert!(interrupted());
        clear_interrupt();
        assert!(!interrupted());
    }

    #[test]
    fn now_ms_moves_forward() {
        let a = now_ms();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(now_ms() >= a);
    }

    #[test]
    fn terminal_width_has_a_sane_default() {
        assert!(terminal_width() >= 20);
    }
}
