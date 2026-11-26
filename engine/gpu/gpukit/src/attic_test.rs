use crate::attic::{
    SLICE, SpanPlan, json_bool, json_str, json_u64, key_ok, lcm, plan_span, read_key,
};
#[cfg(target_os = "linux")]
use crate::attic::{Session, Vmm};
#[cfg(target_os = "linux")]
use std::cell::RefCell;
#[cfg(target_os = "linux")]
use tessera_engine_settings::Attic;

#[cfg(target_os = "linux")]
#[derive(Default)]
struct MockVmm {
    gran: Option<usize>,
    fail_import: bool,
    fail_whole_access: bool,
    next_va: RefCell<u64>,
    next_h: RefCell<u64>,
    mapped: RefCell<Vec<(u64, usize)>>,
    unmapped: RefCell<Vec<(u64, usize)>>,
    freed_va: RefCell<Vec<(u64, usize)>>,
    syncs: RefCell<usize>,
}

#[cfg(target_os = "linux")]
impl MockVmm {
    fn new(gran: usize) -> Self {
        Self {
            gran: Some(gran),
            next_va: RefCell::new(0x1_0000_0000),
            next_h: RefCell::new(1),
            ..Self::default()
        }
    }
}

#[cfg(target_os = "linux")]
impl Vmm for MockVmm {
    fn granularity(&self) -> Option<usize> {
        self.gran
    }

    fn import_fd(&self, fd: i32) -> Result<u64, String> {
        if self.fail_import || fd < 0 {
            return Err("mock import failed".into());
        }
        let mut h = self.next_h.borrow_mut();
        *h += 1;
        Ok(*h)
    }

    fn create(&self, _size: usize) -> Result<u64, String> {
        let mut h = self.next_h.borrow_mut();
        *h += 1;
        Ok(*h)
    }

    fn release(&self, _handle: u64) {}

    fn reserve(&self, size: usize, _align: usize) -> Result<u64, String> {
        let mut va = self.next_va.borrow_mut();
        let out = *va;
        *va += size as u64 + SLICE as u64;
        Ok(out)
    }

    fn free_va(&self, va: u64, size: usize) {
        self.freed_va.borrow_mut().push((va, size));
    }

    fn map(&self, va: u64, size: usize, _handle: u64) -> Result<(), String> {
        self.mapped.borrow_mut().push((va, size));
        Ok(())
    }

    fn unmap(&self, va: u64, size: usize) {
        self.unmapped.borrow_mut().push((va, size));
    }

    fn set_access(&self, _va: u64, size: usize) -> Result<(), String> {
        if self.fail_whole_access && size > 6 * SLICE {
            return Err("cross-segment access refused".into());
        }
        Ok(())
    }

    fn sync(&self) {
        *self.syncs.borrow_mut() += 1;
    }
}

#[cfg(target_os = "linux")]
mod fake_atticd {
    use crate::attic::{json_str, json_u64};
    use std::ffi::{c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread::{self, JoinHandle};

    pub(super) struct Server {
        pub(super) path: PathBuf,
        pub(super) free: Arc<AtomicU64>,
        pub(super) disconnects: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
        join: Option<JoinHandle<()>>,
    }

    fn send_reply(sock: i32, body: &str, fd: i32) {
        let mut iov = libc::iovec {
            iov_base: body.as_ptr() as *mut c_void,
            iov_len: body.len(),
        };
        let mut ctrl = [0u64; 8];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if fd >= 0 {
            msg.msg_control = ctrl.as_mut_ptr().cast();
            msg.msg_controllen = unsafe { libc::CMSG_SPACE(size_of::<c_int>() as u32) } as _;
            unsafe {
                let c = libc::CMSG_FIRSTHDR(&msg);
                (*c).cmsg_level = libc::SOL_SOCKET;
                (*c).cmsg_type = libc::SCM_RIGHTS;
                (*c).cmsg_len = libc::CMSG_LEN(size_of::<c_int>() as u32) as _;
                std::ptr::copy_nonoverlapping(
                    (&raw const fd).cast::<u8>(),
                    libc::CMSG_DATA(c),
                    size_of::<c_int>(),
                );
            }
        }
        let _ = unsafe { libc::sendmsg(sock, &msg, libc::MSG_NOSIGNAL) };
    }

    fn serve_client(cfd: i32, key: &str, total: u64, free: &AtomicU64, refuse_once: &AtomicBool) {
        let mut buf = [0u8; 2048];
        let mut lent = 0u64;
        loop {
            let n = unsafe { libc::recv(cfd, buf.as_mut_ptr().cast(), buf.len(), 0) };
            if n <= 0 {
                break;
            }
            let text = std::str::from_utf8(&buf[..n as usize]).unwrap_or("");
            if json_str(text, "key").as_deref() != Some(key) {
                send_reply(cfd, r#"{"ok":false,"error":"bad key"}"#, -1);
                continue;
            }
            match json_str(text, "op").as_deref() {
                Some("info") => {
                    let f = free.load(Ordering::Relaxed);
                    let body = format!(r#"{{"ok":true,"size":{total},"free":{f},"largest":{f}}}"#);
                    send_reply(cfd, &body, -1);
                }
                Some("alloc") => {
                    let want = json_u64(text, "size").unwrap_or(0);
                    let cur = free.load(Ordering::Relaxed);
                    if refuse_once.swap(false, Ordering::Relaxed) {
                        free.store(cur / 2, Ordering::Relaxed);
                        send_reply(cfd, r#"{"ok":false,"error":"shrunk"}"#, -1);
                        continue;
                    }
                    if want == 0 || want > cur {
                        send_reply(cfd, r#"{"ok":false,"error":"no room"}"#, -1);
                        continue;
                    }
                    let mut pipes = [-1i32; 2];
                    assert_eq!(unsafe { libc::pipe(pipes.as_mut_ptr()) }, 0);
                    free.fetch_sub(want, Ordering::Relaxed);
                    lent += want;
                    let body = format!(r#"{{"ok":true,"size":{want}}}"#);
                    send_reply(cfd, &body, pipes[0]);
                    unsafe {
                        libc::close(pipes[0]);
                        libc::close(pipes[1]);
                    }
                }
                _ => send_reply(cfd, r#"{"ok":false,"error":"unknown op"}"#, -1),
            }
        }
        unsafe { libc::close(cfd) };
        free.fetch_add(lent, Ordering::Relaxed);
    }

    impl Server {
        pub(super) fn start(path: &Path, key: &str, total: u64, refuse_once: bool) -> Self {
            let _ = std::fs::remove_file(path);
            let lfd = unsafe {
                libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0)
            };
            assert!(lfd >= 0);
            let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
            let raw = path.as_os_str().as_bytes();
            assert!(raw.len() < addr.sun_path.len());
            for (d, &s) in addr.sun_path.iter_mut().zip(raw.iter()) {
                *d = s as libc::c_char;
            }
            let len = size_of::<libc::sockaddr_un>() as libc::socklen_t;
            assert_eq!(unsafe { libc::bind(lfd, (&raw const addr).cast(), len) }, 0);
            assert_eq!(unsafe { libc::listen(lfd, 8) }, 0);
            let free = Arc::new(AtomicU64::new(total));
            let disconnects = Arc::new(AtomicU64::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let key = key.to_string();
            let (f, d, st) = (free.clone(), disconnects.clone(), stop.clone());
            let r1 = Arc::new(AtomicBool::new(refuse_once));
            let join = thread::spawn(move || {
                while !st.load(Ordering::Relaxed) {
                    let mut pfd = libc::pollfd {
                        fd: lfd,
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    if unsafe { libc::poll(&mut pfd, 1, 20) } <= 0 {
                        continue;
                    }
                    let cfd =
                        unsafe { libc::accept(lfd, std::ptr::null_mut(), std::ptr::null_mut()) };
                    if cfd >= 0 {
                        serve_client(cfd, &key, total, &f, &r1);
                        d.fetch_add(1, Ordering::Relaxed);
                    }
                }
                unsafe { libc::close(lfd) };
            });
            Self {
                path: path.to_path_buf(),
                free,
                disconnects,
                stop,
                join: Some(join),
            }
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(j) = self.join.take() {
                let _ = j.join();
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[test]
fn plans_and_keys() {
    assert_eq!(lcm(SLICE, 1 << 20), SLICE);
    assert_eq!(
        plan_span(3 * SLICE + 1, SLICE, 2 * SLICE as u64),
        SpanPlan {
            total: 4 * SLICE,
            carve: 2 * SLICE,
            tail: 2 * SLICE,
        }
    );
    assert!(key_ok("abc_123-XYZ"));
    assert!(!key_ok("") && !key_ok("bad key") && !key_ok("bad\"key") && !key_ok("bad\\key"));
    let json = r#"{"ok":true,"size":4194304,"free":2097152,"error":"no \"ok\":false"}"#;
    assert_eq!(json_bool(json, "ok"), Some(true));
    assert_eq!(json_u64(json, "free"), Some(2097152));
    assert_eq!(json_str(json, "error").as_deref(), Some("no \"ok\":false"));

    let dir = tessera_testkit::scratch(&format!("attic-key-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let key_path = dir.join("k");
    std::fs::write(&key_path, "  secret_key_1\n").unwrap();
    assert_eq!(read_key(&key_path).unwrap(), "secret_key_1");
    std::fs::write(&key_path, vec![b'a'; 4097]).unwrap();
    assert!(read_key(&key_path).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "linux")]
#[test]
fn borrow_failopen_and_release() {
    use std::sync::atomic::Ordering;
    let dir = tessera_testkit::scratch(&format!("at-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let sock = dir.join("a.sock");
    let key_path = dir.join("key");
    std::fs::write(&key_path, "testkey_42\n").unwrap();
    let cfg = Attic {
        socket: sock.clone(),
        key_file: key_path.clone(),
    };
    let srv = fake_atticd::Server::start(&sock, "testkey_42", (4 * SLICE) as u64, false);
    let mut vmm = MockVmm::new(SLICE);
    vmm.fail_whole_access = true;
    let mut sess = Session::default();

    // Disabled config returns None without touching the daemon.
    assert_eq!(sess.alloc(None, &vmm, 4 * SLICE), None);
    assert!(!sess.has_conn());

    // Borrow 4 MiB carveout with 4 MiB tail.
    let va = sess.alloc(Some(&cfg), &vmm, 8 * SLICE).expect("borrowed");
    assert_eq!(sess.borrowed(), (4 * SLICE) as u64);
    assert_eq!(srv.free.load(Ordering::Relaxed), 0);
    assert_eq!(vmm.mapped.borrow().len(), 2);
    assert_eq!(sess.alloc(Some(&cfg), &vmm, SLICE), None);
    assert!(sess.free(&vmm, va));
    assert_eq!((sess.borrowed(), sess.has_conn()), (0, false));
    assert_eq!((*vmm.syncs.borrow(), vmm.unmapped.borrow().len()), (1, 2));
    for _ in 0..50 {
        if srv.free.load(Ordering::Relaxed) == (4 * SLICE) as u64 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(srv.free.load(Ordering::Relaxed), (4 * SLICE) as u64);
    assert!(srv.disconnects.load(Ordering::Relaxed) >= 1);

    // Wrong key and missing socket fail open cleanly.
    std::fs::write(&key_path, "wrong_key\n").unwrap();
    let mut bad = Session::default();
    assert_eq!(bad.alloc(Some(&cfg), &vmm, 2 * SLICE), None);
    let missing = Attic {
        socket: dir.join("nope.sock"),
        key_file: key_path.clone(),
    };
    let mut miss = Session::default();
    assert_eq!(miss.alloc(Some(&missing), &vmm, 2 * SLICE), None);

    // Refusal triggers one info retry and maps the smaller carveout.
    std::fs::write(&key_path, "testkey_42\n").unwrap();
    let sock2 = dir.join("b.sock");
    let srv2 = fake_atticd::Server::start(&sock2, "testkey_42", (4 * SLICE) as u64, true);
    let cfg2 = Attic {
        socket: sock2,
        key_file: key_path,
    };
    let mut retry = Session::default();
    let va2 = retry.alloc(Some(&cfg2), &vmm, 4 * SLICE).expect("retried");
    assert_eq!(retry.borrowed(), (2 * SLICE) as u64);
    retry.release_all(&vmm);
    drop((va2, srv, srv2));
    let _ = std::fs::remove_dir_all(&dir);
}
