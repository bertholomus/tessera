//! Which retained run a prompt continues.

use crate::{Pool, RunId, Tag};
use std::collections::HashMap;

struct Entry {
    run: RunId,
    history: Vec<u32>,
    usable: usize,
}

/// One compressed edge in the chunk prefix trie.
struct Node {
    owner: usize,
    start: usize,
    end: usize,
    edges: HashMap<u64, Vec<usize>>,
    ends: Vec<usize>,
}

/// The tokens of retained runs, indexed by chunk prefixes.
pub struct Prefixes {
    grid: usize,
    kept: Vec<Entry>,
    nodes: Vec<Node>,
}

/// Fast 64-bit hash of one `grid`-token chunk.
fn hash_chunk(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in tokens {
        h ^= u64::from(t);
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ (h >> 33)
}

/// Common chunk-aligned token count between `a` and `b`.
fn common_chunks(a: &[u32], b: &[u32], step: usize) -> usize {
    let n = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (n / step) * step
}

fn empty_node(owner: usize, start: usize, end: usize) -> Node {
    Node {
        owner,
        start,
        end,
        edges: HashMap::new(),
        ends: Vec::new(),
    }
}

impl Prefixes {
    /// Skips come in whole `grid` steps.
    pub fn new(grid: usize) -> Self {
        Self {
            grid,
            kept: Vec::new(),
            nodes: vec![empty_node(0, 0, 0)],
        }
    }

    /// Records the tokens behind a retained run's `valid` rows.
    pub fn keep(&mut self, run: RunId, history: Vec<u32>, valid: usize) {
        let step = self.grid.max(1);
        let usable = (history.len().min(valid) / step) * step;
        let idx = self.kept.len();
        self.kept.push(Entry {
            run,
            history,
            usable,
        });
        if usable > 0 {
            self.insert(idx);
        }
    }

    /// The newest run with the longest skip.
    pub fn find(
        &mut self,
        pool: &Pool,
        tag: Tag,
        prompt: &[u32],
        limit: usize,
    ) -> Option<(RunId, usize)> {
        if self.kept.iter().any(|e| pool.rows(e.run).is_err()) {
            self.prune(pool);
        }
        let step = self.grid.max(1);
        let cap = (prompt.len().min(limit) / step) * step;
        if cap == 0 || self.kept.is_empty() {
            return None;
        }
        let mut path = Vec::new();
        let (mut cur, mut off) = (0usize, 0usize);
        while off < cap {
            let chunk = &prompt[off..off + step];
            let Some(cid) = self.child(cur, off, chunk, step) else {
                break;
            };
            let node = &self.nodes[cid];
            let edge = &self.kept[node.owner].history[off..node.end];
            let matched = off + common_chunks(edge, &prompt[off..cap], step);
            path.push((cid, matched));
            if matched < node.end || matched == cap {
                break;
            }
            cur = cid;
            off = matched;
        }
        let mut skip_child = None;
        for &(nid, matched) in path.iter().rev() {
            if let Some(idx) = self.newest_in(nid, skip_child, pool, tag) {
                return Some((self.kept[idx].run, matched));
            }
            skip_child = Some(nid);
        }
        None
    }

    /// Drops evicted runs and rebuilds the compressed trie.
    fn prune(&mut self, pool: &Pool) {
        self.kept.retain(|e| pool.rows(e.run).is_ok());
        self.nodes.clear();
        self.nodes.push(empty_node(0, 0, 0));
        for idx in 0..self.kept.len() {
            if self.kept[idx].usable > 0 {
                self.insert(idx);
            }
        }
    }

    /// Child of `nid` whose leading `step` tokens match `chunk`.
    fn child(&self, nid: usize, off: usize, chunk: &[u32], step: usize) -> Option<usize> {
        let h = hash_chunk(chunk);
        let bucket = self.nodes[nid].edges.get(&h)?;
        bucket
            .iter()
            .copied()
            .find(|&cid| &self.kept[self.nodes[cid].owner].history[off..off + step] == chunk)
    }

    /// Inserts `idx` into the compressed chunk trie.
    fn insert(&mut self, idx: usize) {
        let step = self.grid.max(1);
        let usable = self.kept[idx].usable;
        let (mut cur, mut off) = (0usize, 0usize);
        loop {
            if off == usable {
                self.nodes[cur].ends.push(idx);
                return;
            }
            let chunk_hash = hash_chunk(&self.kept[idx].history[off..off + step]);
            let found = self.nodes[cur].edges.get(&chunk_hash).and_then(|bucket| {
                bucket.iter().position(|&cid| {
                    let owner = self.nodes[cid].owner;
                    self.kept[owner].history[off..off + step]
                        == self.kept[idx].history[off..off + step]
                })
            });
            let Some(pos) = found else {
                let leaf = self.nodes.len();
                let mut node = empty_node(idx, off, usable);
                node.ends.push(idx);
                self.nodes.push(node);
                self.nodes[cur]
                    .edges
                    .entry(chunk_hash)
                    .or_default()
                    .push(leaf);
                return;
            };
            let cid = self.nodes[cur].edges[&chunk_hash][pos];
            let (owner, end) = (self.nodes[cid].owner, self.nodes[cid].end);
            let common = common_chunks(
                &self.kept[owner].history[off..end],
                &self.kept[idx].history[off..usable],
                step,
            );
            let split = off + common;
            if split == end {
                cur = cid;
                off = split;
                continue;
            }
            let mid = self.nodes.len();
            let old_hash = hash_chunk(&self.kept[owner].history[split..split + step]);
            let mut mid_node = empty_node(owner, off, split);
            mid_node.edges.insert(old_hash, vec![cid]);
            self.nodes[cid].start = split;
            if split == usable {
                mid_node.ends.push(idx);
                self.nodes.push(mid_node);
            } else {
                let leaf = mid + 1;
                let new_hash = hash_chunk(&self.kept[idx].history[split..split + step]);
                mid_node.edges.entry(new_hash).or_default().push(leaf);
                let mut leaf_node = empty_node(idx, split, usable);
                leaf_node.ends.push(idx);
                self.nodes.push(mid_node);
                self.nodes.push(leaf_node);
            }
            self.nodes[cur]
                .edges
                .get_mut(&chunk_hash)
                .expect("edge bucket")[pos] = mid;
            return;
        }
    }

    /// Newest entry in `root` matching `tag`, skipping `exclude`.
    fn newest_in(
        &self,
        root: usize,
        exclude: Option<usize>,
        pool: &Pool,
        tag: Tag,
    ) -> Option<usize> {
        let mut stack = vec![root];
        let mut best: Option<usize> = None;
        while let Some(nid) = stack.pop() {
            let node = &self.nodes[nid];
            for &idx in node.ends.iter().rev() {
                if best.is_some_and(|b| idx <= b) {
                    break;
                }
                if pool.tag(self.kept[idx].run) == Ok(tag) {
                    best = Some(idx);
                    break;
                }
            }
            for bucket in node.edges.values() {
                for &cid in bucket {
                    if Some(cid) != exclude {
                        stack.push(cid);
                    }
                }
            }
        }
        best
    }
}
