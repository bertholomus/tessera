//! A host copy of the page table, by slot.

use std::ops::Range;

/// Each slot's lane names its pages, in order.
pub struct Lanes {
    /// Rows per page.
    pub page: usize,
    /// Physical pages in the shared pool.
    pool: usize,
    /// Entries per slot.
    width: usize,
    table: Vec<u32>,
    /// Dirty entry range within each slot's lane.
    dirty: Vec<Range<usize>>,
}

impl Lanes {
    /// Page table for `slots`, backed by `pool` physical pages.
    pub fn new(page: usize, entries: usize, slots: usize, pool: usize) -> Self {
        let pool = pool.max(1);
        let width = entries / slots;
        let table = match pool < entries {
            true => vec![0; entries],
            false => (0..entries as u32).collect(),
        };
        Self {
            page,
            pool,
            width,
            table,
            dirty: vec![0..width; slots],
        }
    }

    /// Physical pages in the pool.
    pub fn pages(&self) -> usize {
        self.pool
    }

    /// Whether slots share fewer physical pages than full capacity.
    pub fn shared(&self) -> bool {
        self.pool < self.table.len()
    }

    /// Resets the table to its initial page map.
    pub fn reset(&mut self) {
        let shared = self.shared();
        let slots = self.dirty.len();
        for s in 0..slots {
            let base = s * self.width;
            let mut lo = self.width;
            let mut hi = 0;
            for i in 0..self.width {
                let want = if shared { 0 } else { (base + i) as u32 };
                if self.table[base + i] != want {
                    self.table[base + i] = want;
                    lo = lo.min(i);
                    hi = i + 1;
                }
            }
            if lo < hi {
                mark(&mut self.dirty[s], lo..hi);
            }
        }
    }

    /// Writes a slot's first pages.
    pub fn set(&mut self, slot: u32, pages: &[u32]) {
        let s = slot as usize;
        let first = s * self.width;
        let lane = &mut self.table[first..][..pages.len()];
        let mut lo = pages.len();
        let mut hi = 0;
        for (i, (&old, &new)) in lane.iter().zip(pages).enumerate() {
            if old != new {
                lo = lo.min(i);
                hi = i + 1;
            }
        }
        if lo < hi {
            lane[lo..hi].copy_from_slice(&pages[lo..hi]);
            mark(&mut self.dirty[s], lo..hi);
        }
    }

    /// Dirty `(table_index, pages)` slices for rank broadcast.
    pub fn pending(&self) -> Vec<(u32, &[u32])> {
        let mut out = Vec::new();
        for (s, range) in self.dirty.iter().enumerate() {
            if range.start < range.end {
                let start = s * self.width + range.start;
                let end = s * self.width + range.end;
                out.push((start as u32, &self.table[start..end]));
            }
        }
        out
    }

    /// Applies rank 0's patches to the local table.
    pub fn apply(&mut self, patches: &[(u32, Vec<u32>)]) {
        for (offset, values) in patches {
            let start = *offset as usize;
            let end = start + values.len();
            if values.is_empty() || end > self.table.len() {
                continue;
            }
            if self.table[start..end] == values[..] {
                continue;
            }
            self.table[start..end].copy_from_slice(values);
            let mut pos = start;
            while pos < end {
                let s = pos / self.width;
                let lo = pos % self.width;
                let stop = ((s + 1) * self.width).min(end);
                mark(&mut self.dirty[s], lo..lo + (stop - pos));
                pos = stop;
            }
        }
    }

    /// Dirty `(byte_offset, bytes)` slices to upload to device.
    pub fn take_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        let mut out = Vec::new();
        for (s, range) in self.dirty.iter_mut().enumerate() {
            if range.start >= range.end {
                continue;
            }
            let r = std::mem::replace(range, 0..0);
            let start = s * self.width + r.start;
            let end = s * self.width + r.end;
            let bytes: Vec<u8> = self.table[start..end]
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect();
            out.push((start * 4, bytes));
        }
        out
    }
}

fn mark(cur: &mut Range<usize>, add: Range<usize>) {
    if cur.start >= cur.end {
        *cur = add;
    } else {
        *cur = cur.start.min(add.start)..cur.end.max(add.end);
    }
}
