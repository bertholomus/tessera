use std::path::PathBuf;
use tessera_kv::{PageStore, page_keys};

#[test]
fn pages_persist_and_evict() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-store-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let seed = [11u64, 29];
    let p1 = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let p2 = [1u32, 2, 3, 4, 9, 10, 11, 12];
    let k1 = page_keys(seed, &p1, 4);
    let k2 = page_keys(seed, &p2, 4);
    assert_eq!(k1.len(), 2);
    assert_eq!(k1[0], k2[0]);
    assert_ne!(k1[1], k2[1]);
    assert_ne!(page_keys([11, 30], &p1, 4)[0], k1[0]);

    let mut store = PageStore::open(&dir, 32).unwrap();
    assert!(store.save(k1[0], &[1u8; 16]).unwrap());
    // Saving an identical page skips disk writes.
    assert!(!store.save(k1[0], &[1u8; 16]).unwrap());
    assert!(store.save(k1[1], &[2u8; 16]).unwrap());
    assert_eq!(store.stats().saves, 2);
    assert_eq!(store.prefix(&k1), 2);
    assert_eq!(store.prefix(&k2), 1);

    // Touching k1[0] keeps it hot when k2[1] evicts k1[1].
    let mut got = [0u8; 16];
    assert!(store.load(k1[0], &mut got).unwrap());
    assert_eq!(got, [1u8; 16]);
    assert!(store.save(k2[1], &[3u8; 16]).unwrap());
    assert_eq!(store.stats().evictions, 1);
    assert!(store.has(k1[0]));
    assert!(!store.has(k1[1]));
    assert!(store.has(k2[1]));
    drop(store);

    // Reopening recovers existing pages from disk.
    let mut reopened = PageStore::open(&dir, 32).unwrap();
    assert_eq!(reopened.prefix(&k2), 2);
    let mut tail = [0u8; 16];
    assert!(reopened.load(k2[1], &mut tail).unwrap());
    assert_eq!(tail, [3u8; 16]);
    std::fs::remove_dir_all(&dir).unwrap();
}
