//! A host copy of the page table, by slot.

/// Each slot's lane names its pages, in order.
pub(crate) struct Lanes {
    /// Rows per page.
    pub page: usize,
    /// Physical pages in the shared pool.
    pool: usize,
    /// Entries per slot.
    width: usize,
    table: Vec<u32>,
    /// Whether the device's copy is old.
    stale: bool,
}

impl Lanes {
    /// Page table for `slots`, backed by `pool` physical pages.
    pub fn new(page: usize, entries: usize, slots: usize, pool: usize) -> Self {
        let pool = pool.max(1);
        let table = match pool < entries {
            true => vec![0; entries],
            false => (0..entries as u32).collect(),
        };
        Self {
            page,
            pool,
            width: entries / slots,
            table,
            stale: true,
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
        let len = self.table.len();
        if self.shared() {
            if self.table.iter().any(|&p| p != 0) {
                self.table.fill(0);
                self.stale = true;
            }
            return;
        }
        let own = 0..len as u32;
        if !own.clone().eq(self.table.iter().copied()) {
            self.table = own.collect();
            self.stale = true;
        }
    }

    /// Writes a slot's first pages.
    pub fn set(&mut self, slot: u32, pages: &[u32]) {
        let first = slot as usize * self.width;
        self.table[first..][..pages.len()].copy_from_slice(pages);
        self.stale = true;
    }

    /// The table, if the device's copy is old.
    pub fn pending(&self) -> Option<&[u32]> {
        self.stale.then_some(&self.table)
    }

    /// Takes rank 0's table.
    pub fn replace(&mut self, table: Vec<u32>) {
        if table != self.table {
            self.table = table;
            self.stale = true;
        }
    }

    /// The table's bytes, if the device's copy is old.
    pub fn changed(&mut self) -> Option<Vec<u8>> {
        let stale = std::mem::take(&mut self.stale);
        let bytes = self.table.iter().flat_map(|p| p.to_le_bytes());
        stale.then(|| bytes.collect())
    }
}
