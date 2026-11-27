//! A file on disk, read into memory the caller owns.

use std::fs;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tessera_engine::{Filed, Reads};

/// The least share of a read for one thread.
const SHARE: usize = 4 << 20;
/// Reader threads for one read, at most.
const READERS: usize = 8;
/// File offsets, lengths and memory a direct read keeps to.
const BLOCK: u64 = 4096;
/// Bytes one direct read stages.
const STAGE: usize = 4 << 20;

/// `O_DIRECT`, on the targets whose value is known here.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const DIRECT: Option<i32> = Some(0o200000);
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const DIRECT: Option<i32> = Some(0o40000);
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
const DIRECT: Option<i32> = None;

/// The file opened past the cache, where the system can.
fn uncached(path: &Path) -> Option<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut open = fs::OpenOptions::new();
    open.read(true).custom_flags(DIRECT?).open(path).ok()
}

/// True when `file` resides on a local macOS mount.
#[cfg(target_os = "macos")]
pub(crate) fn is_local(file: &fs::File) -> bool {
    use std::os::unix::io::AsRawFd;
    const MNT_LOCAL: u32 = 0x0000_1000;
    #[repr(C)]
    struct StatFs {
        f_bsize: u32,
        f_iosize: i32,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_fsid: [i32; 2],
        f_owner: u32,
        f_type: u32,
        f_flags: u32,
        f_fssubtype: u32,
        f_fstypename: [u8; 16],
        f_mntonname: [u8; 1024],
        f_mntfromname: [u8; 1024],
        f_flags_ext: u32,
        f_reserved: [u32; 7],
    }
    unsafe extern "C" {
        fn fstatfs(fd: i32, buf: *mut StatFs) -> i32;
    }
    let mut st = std::mem::MaybeUninit::<StatFs>::zeroed();
    let rc = unsafe { fstatfs(file.as_raw_fd(), st.as_mut_ptr()) };
    if rc != 0 {
        return true;
    }
    let st = unsafe { st.assume_init() };
    (st.f_flags & MNT_LOCAL) != 0
}

/// Non-macOS targets keep threaded reads.
#[cfg(not(target_os = "macos"))]
pub(crate) fn is_local(_file: &fs::File) -> bool {
    true
}

pub(crate) struct Disk {
    path: PathBuf,
    cached: fs::File,
    local: bool,
    /// Opened on the first direct read.
    direct: OnceLock<Option<fs::File>>,
    /// Free staging buffers, kept for the next read.
    stages: Mutex<Vec<Vec<u8>>>,
}

impl Disk {
    pub(crate) fn new(path: &Path, cached: fs::File) -> Self {
        let local = is_local(&cached);
        Self {
            path: path.to_path_buf(),
            cached,
            local,
            direct: OnceLock::new(),
            stages: Mutex::new(Vec::new()),
        }
    }

    /// Fills `into` from `at`, single-stream on remote macOS mounts.
    pub(crate) fn read(&self, at: u64, into: &mut [u8], reads: Reads) -> io::Result<()> {
        let direct = match reads {
            Reads::Cached => None,
            Reads::Direct => self.direct.get_or_init(|| uncached(&self.path)).as_ref(),
        };
        let threads = if self.local {
            (into.len() / SHARE).clamp(1, READERS)
        } else {
            1
        };
        if threads == 1 {
            return self.fill(at, into, direct);
        }
        let per = into.len().div_ceil(threads);
        std::thread::scope(|s| {
            let mut readers = Vec::with_capacity(threads);
            for (i, part) in into.chunks_mut(per).enumerate() {
                let from = at + (i * per) as u64;
                readers.push(s.spawn(move || self.fill(from, part, direct)));
            }
            let mut done = Ok(());
            for r in readers {
                done = done.and(r.join().expect("a reader thread"));
            }
            done
        })
    }

    /// One thread's bytes, past the cache when `direct` is given.
    fn fill(&self, at: u64, into: &mut [u8], direct: Option<&fs::File>) -> io::Result<()> {
        let Some(file) = direct else {
            return self.cached.read_exact_at(into, at);
        };
        let spare = self.stages.lock().unwrap().pop();
        let mut stage = spare.unwrap_or_else(|| vec![0u8; STAGE + BLOCK as usize]);
        let done = staged(file, at, into, &mut stage);
        self.stages.lock().unwrap().push(stage);
        match done {
            // Some file systems refuse direct reads.
            Err(e) if e.kind() == io::ErrorKind::InvalidInput => {
                self.cached.read_exact_at(into, at)
            }
            done => done,
        }
    }
}

/// Reads whole blocks into `stage` and copies out the span.
fn staged(file: &fs::File, at: u64, into: &mut [u8], stage: &mut [u8]) -> io::Result<()> {
    let block = BLOCK as usize;
    let skip = stage.as_ptr().align_offset(block);
    let stage = &mut stage[skip..skip + STAGE];
    let mut done = 0;
    while done < into.len() {
        let want = at + done as u64;
        let from = want / BLOCK * BLOCK;
        let lead = (want - from) as usize;
        let need = (lead + into.len() - done).next_multiple_of(block);
        let got = file.read_at(&mut stage[..need.min(STAGE)], from)?;
        if got <= lead {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let take = (got - lead).min(into.len() - done);
        into[done..done + take].copy_from_slice(&stage[lead..lead + take]);
        done += take;
    }
    Ok(())
}

/// One tensor's bytes in a file on disk.
pub(crate) struct Span {
    pub(crate) disk: Arc<Disk>,
    pub(crate) at: u64,
    pub(crate) bytes: usize,
}

impl Filed for Span {
    fn bytes(&self) -> usize {
        self.bytes
    }

    fn read(&self, into: &mut [u8], reads: Reads) -> io::Result<()> {
        self.disk.read(self.at, into, reads)
    }
}
