//! Pages of kept rows, shared between runs by refcount.

mod pool;
mod prefix;
mod reuse;
mod store;

pub use pool::{Error, PageCopy, Pool, RunId, Stats, Tag};
pub use prefix::Prefixes;
pub use reuse::{Begin, Why, begin};
pub use store::{DiskKv, DiskMsg, PageKey, PageStore, StoreStats, decode_msg, page_keys};

#[cfg(test)]
mod pool_test;
#[cfg(test)]
mod reuse_test;
