//! A fixed-size, lock-free "latest value wins" channel over POSIX shared
//! memory, shared between this Rust process and a Python process in the
//! same pattern on both sides (`py/policyserve/shm.py`). This is the
//! mechanism System 1 and System 2 hand data across without either one
//! ever blocking on the other (design doc §8.1): a classic seqlock --
//! the writer brackets its write with an odd/even sequence counter, the
//! reader retries (bounded) if it catches a write in progress or
//! straddling one, and on exhausting its retries it simply keeps
//! whatever it already had. No lock is ever held across the slow side's
//! own work, by construction: a write here is one `memcpy`-sized
//! operation, never "call the reasoner, then unlock."
//!
//! Layout: `[seq: u64][payload: N bytes]`. `seq` starts and stays even
//! between writes; a writer bumps it odd before writing the payload and
//! back to even after, so a reader catching an odd value (or a
//! before/after mismatch) knows it raced a write and retries.
//!
//! Python's side can't take a real hardware memory fence the way
//! `AtomicU64` does here -- a deliberate, documented limitation (see
//! `py/policyserve/shm.py`), relying instead on x86-64/ARM64's guarantee
//! that an aligned 8-byte load/store doesn't tear. Correct for this
//! project's purposes; a production system mixing languages this way
//! would want an explicit review of that guarantee on its actual target
//! hardware, not an assumption.
//!
//! Lifetime: whichever side creates the segment (`create`) owns
//! unlinking it; the other side only ever `open`s an existing one. This
//! project's convention (see `main.rs`) is that Python always creates,
//! Rust always opens -- so `Drop` here only ever unmaps and closes the
//! fd, never unlinks, avoiding a double-unlink race with Python's own
//! `multiprocessing.shared_memory` resource tracker.

use std::ffi::CString;
use std::io;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const SEQ_SIZE: usize = 8;

pub struct SeqLockChannel {
    ptr: *mut u8,
    fd: i32,
    map_size: usize,
    payload_size: usize,
}

impl SeqLockChannel {
    /// Opens a segment some other process (always Python, by this
    /// project's convention) already created. Retries briefly, since
    /// System 1 can start slightly before Python has created every
    /// channel it owns.
    pub fn open_existing(name: &str, payload_size: usize, wait: Duration) -> io::Result<Self> {
        let cname = shm_name(name)?;
        let map_size = SEQ_SIZE + payload_size;
        let deadline = Instant::now() + wait;
        loop {
            let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0o666) };
            if fd >= 0 {
                return Self::map(fd, map_size, payload_size);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::last_os_error());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn map(fd: i32, map_size: usize, payload_size: usize) -> io::Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                map_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            fd,
            map_size,
            payload_size,
        })
    }

    fn seq(&self) -> &AtomicU64 {
        unsafe { AtomicU64::from_ptr(self.ptr as *mut u64) }
    }

    fn payload_ptr(&self) -> *mut u8 {
        unsafe { self.ptr.add(SEQ_SIZE) }
    }

    /// Writes `data` as the new latest value. `data.len()` must equal
    /// this channel's `payload_size` -- a programmer error, not a
    /// runtime condition to recover from, hence the assert rather than
    /// a `Result`.
    pub fn write(&self, data: &[u8]) {
        assert_eq!(data.len(), self.payload_size, "payload size mismatch");
        let seq = self.seq();
        let cur = seq.load(Ordering::Relaxed);
        seq.store(cur.wrapping_add(1), Ordering::Release);
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.payload_ptr(), self.payload_size);
        }
        seq.store(cur.wrapping_add(2), Ordering::Release);
    }

    /// Reads the latest value into `out` (`out.len()` must equal
    /// `payload_size`). Returns `false` if every retry raced a
    /// concurrent write -- vanishingly rare in practice (writes are a
    /// handful of nanoseconds), but never blocks waiting for one to
    /// resolve: the caller keeps whatever it already had, by design.
    pub fn read(&self, out: &mut [u8], max_retries: u32) -> bool {
        assert_eq!(out.len(), self.payload_size, "payload size mismatch");
        let seq = self.seq();
        for _ in 0..max_retries {
            let s1 = seq.load(Ordering::Acquire);
            if !s1.is_multiple_of(2) {
                continue; // writer mid-write
            }
            unsafe {
                ptr::copy_nonoverlapping(self.payload_ptr(), out.as_mut_ptr(), self.payload_size);
            }
            let s2 = seq.load(Ordering::Acquire);
            if s1 == s2 {
                return true;
            }
            // s1 != s2: a write straddled this read -- retry.
        }
        false
    }
}

impl Drop for SeqLockChannel {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.map_size);
            libc::close(self.fd);
        }
    }
}

// SeqLockChannel's only shared mutable state is the mmap'd region
// itself, accessed exclusively through atomics and plain byte copies of
// a fixed size -- safe to hand across a thread boundary in this process
// the same way it's already safe to hand across the process boundary to
// Python.
unsafe impl Send for SeqLockChannel {}
unsafe impl Sync for SeqLockChannel {}

fn shm_name(name: &str) -> io::Result<CString> {
    // POSIX shm names are required to start with a single leading '/'
    // and contain no further '/' -- matches exactly what Python's
    // `multiprocessing.shared_memory.SharedMemory(name=...)` produces,
    // confirmed empirically (not assumed) before relying on it here.
    CString::new(format!("/{name}")).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // A real bug this test harness hit once and is now written to never
    // hit again: macOS's `shm_open`/`ftruncate` has no equivalent of
    // Python's `multiprocessing.resource_tracker` to clean up a segment
    // left behind by a process that panicked before reaching its own
    // cleanup call -- and macOS additionally refuses a *second*
    // `ftruncate` on an already-sized segment (`EINVAL`), so a leaked
    // segment from a previous failed run doesn't just linger harmlessly,
    // it makes the *next* run's `create` fail too, with a confusing
    // error far from the actual cause. Fixed two ways at once: every
    // test gets a name unique to this process (so one run's segments can
    // never collide with another's, including a previous crashed run's
    // leftovers), and cleanup happens via `Drop` on `UnlinkGuard`, so it
    // still runs even when an `assert!` inside the test panics.
    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn unique_test_name() -> String {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("policyserve_test_{}_{}", std::process::id(), n)
    }

    struct UnlinkGuard {
        name: String,
    }

    impl Drop for UnlinkGuard {
        fn drop(&mut self) {
            let cname = shm_name(&self.name).unwrap();
            unsafe {
                libc::shm_unlink(cname.as_ptr());
            }
        }
    }

    fn create_for_test(payload_size: usize) -> (SeqLockChannel, UnlinkGuard) {
        let name = unique_test_name();
        let cname = shm_name(&name).unwrap();
        // O_EXCL: a name collision here is a real bug (the "unique" name
        // generator broke), not something to silently paper over by
        // reusing whatever's already there.
        let fd = unsafe {
            libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o666,
            )
        };
        assert!(fd >= 0, "shm_open failed: {}", io::Error::last_os_error());
        let map_size = SEQ_SIZE + payload_size;
        let rc = unsafe { libc::ftruncate(fd, map_size as libc::off_t) };
        assert_eq!(rc, 0, "ftruncate failed: {}", io::Error::last_os_error());
        let chan = SeqLockChannel::map(fd, map_size, payload_size).unwrap();
        (chan, UnlinkGuard { name })
    }

    fn open_for_test(name: &str, payload_size: usize) -> SeqLockChannel {
        let cname = shm_name(name).unwrap();
        let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0o666) };
        assert!(
            fd >= 0,
            "shm_open (open-only) failed: {}",
            io::Error::last_os_error()
        );
        SeqLockChannel::map(fd, SEQ_SIZE + payload_size, payload_size).unwrap()
    }

    #[test]
    fn a_write_is_visible_to_a_read_on_the_same_channel() {
        let (chan, _guard) = create_for_test(8);
        chan.write(&42u64.to_le_bytes());
        let mut out = [0u8; 8];
        assert!(chan.read(&mut out, 10));
        assert_eq!(u64::from_le_bytes(out), 42);
    }

    #[test]
    fn repeated_writes_are_each_visible_in_order() {
        let (chan, _guard) = create_for_test(8);
        let mut out = [0u8; 8];
        for i in 0..1000u64 {
            chan.write(&i.to_le_bytes());
            assert!(chan.read(&mut out, 10));
            assert_eq!(u64::from_le_bytes(out), i);
        }
    }

    #[test]
    fn concurrent_writer_and_reader_never_see_a_torn_value() {
        // Payload is (counter, counter) -- any read where the two
        // halves disagree would mean the reader caught a torn/partial
        // write despite the seqlock, which must never happen.
        let (chan, guard) = create_for_test(16);
        let chan2 = open_for_test(&guard.name, 16);

        let writer = std::thread::spawn(move || {
            for i in 0u64..200_000 {
                let mut buf = [0u8; 16];
                buf[0..8].copy_from_slice(&i.to_le_bytes());
                buf[8..16].copy_from_slice(&i.to_le_bytes());
                chan.write(&buf);
            }
        });

        let reader = std::thread::spawn(move || {
            let mut out = [0u8; 16];
            let mut saw_any = false;
            for _ in 0..200_000 {
                if chan2.read(&mut out, 50) {
                    let a = u64::from_le_bytes(out[0..8].try_into().unwrap());
                    let b = u64::from_le_bytes(out[8..16].try_into().unwrap());
                    assert_eq!(a, b, "torn read: halves disagree ({a} vs {b})");
                    saw_any = true;
                }
            }
            assert!(saw_any, "reader never observed a value");
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }
}
