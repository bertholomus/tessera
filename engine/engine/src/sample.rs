//! Token sampling from logits with temperature, top-k, top-p and min-p.

use crate::Allowed;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Sampling settings for one answer. Zero temperature is greedy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
        }
    }
}

impl Sampling {
    /// Whether this request uses greedy argmax decoding.
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Per-sequence deterministic PRNG and token sampler.
pub(crate) struct Sampler {
    cfg: Sampling,
    state: u64,
}

impl Sampler {
    pub(crate) fn new(cfg: Sampling) -> Self {
        Self {
            cfg,
            state: cfg.seed.unwrap_or_else(fresh_seed),
        }
    }

    /// Picks one token from `logits`, respecting `allowed` when given.
    pub(crate) fn pick(&mut self, logits: &[f32], allowed: Option<&Allowed>) -> u32 {
        if let Some(Allowed::One(id)) = allowed {
            return *id;
        }
        if self.cfg.greedy() {
            return argmax(logits, allowed);
        }
        let mut max = f32::NEG_INFINITY;
        each_allowed(logits, allowed, |_, v| {
            if v > max {
                max = v;
            }
        });
        if !max.is_finite() {
            return argmax(logits, allowed);
        }
        let temp = self.cfg.temperature;
        let mut floor = max - 20.0 * temp;
        if self.cfg.min_p > 0.0 {
            floor = floor.max(max + temp * self.cfg.min_p.ln());
        }
        let mut cands = Vec::new();
        each_allowed(logits, allowed, |id, v| {
            if v >= floor {
                cands.push((id, v));
            }
        });
        if cands.is_empty() {
            return argmax(logits, allowed);
        }
        filter_top(&mut cands, self.cfg.top_k, self.cfg.top_p, max, temp);
        let weights: Vec<f32> = cands
            .iter()
            .map(|&(_, v)| ((v - max) / temp).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let target = self.next_unit() * total;
        let mut acc = 0.0f32;
        for (&(id, _), &w) in cands.iter().zip(&weights) {
            acc += w;
            if target < acc {
                return id;
            }
        }
        cands[cands.len() - 1].0
    }

    /// A uniform float in `[0, 1)` from SplitMix64.
    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }
}

/// Calls `f` for every allowed finite `(id, logit)`.
fn each_allowed(logits: &[f32], allowed: Option<&Allowed>, mut f: impl FnMut(u32, f32)) {
    match allowed {
        None | Some(Allowed::Any) => {
            for (i, &v) in logits.iter().enumerate() {
                if v.is_finite() {
                    f(i as u32, v);
                }
            }
        }
        Some(Allowed::One(id)) => {
            if let Some(&v) = logits.get(*id as usize)
                && v.is_finite()
            {
                f(*id, v);
            }
        }
        Some(Allowed::Few(ids)) => {
            for &id in ids {
                if let Some(&v) = logits.get(id as usize)
                    && v.is_finite()
                {
                    f(id, v);
                }
            }
        }
        Some(Allowed::Except(banned)) => {
            for (i, &v) in logits.iter().enumerate() {
                if v.is_finite() && !banned.contains(&(i as u32)) {
                    f(i as u32, v);
                }
            }
        }
        Some(Allowed::Bits(bits)) => {
            for (w, &word) in bits.iter().enumerate() {
                let mut left = word;
                while left != 0 {
                    let b = left.trailing_zeros() as usize;
                    let idx = w * 32 + b;
                    if let Some(&v) = logits.get(idx)
                        && v.is_finite()
                    {
                        f(idx as u32, v);
                    }
                    left &= left - 1;
                }
            }
        }
    }
}

/// Lowest-index argmax over allowed tokens.
fn argmax(logits: &[f32], allowed: Option<&Allowed>) -> u32 {
    let mut best_id = 0u32;
    let mut best = f32::NEG_INFINITY;
    let mut found = false;
    each_allowed(logits, allowed, |id, v| {
        if !found || v > best || (v == best && id < best_id) {
            best = v;
            best_id = id;
            found = true;
        }
    });
    best_id
}

/// Orders candidates by descending logit, then ascending id.
fn by_logit(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

/// Applies top-k and top-p truncation in place.
fn filter_top(cands: &mut Vec<(u32, f32)>, top_k: usize, top_p: f32, max: f32, temp: f32) {
    if top_k > 0 && cands.len() > top_k {
        cands.select_nth_unstable_by(top_k - 1, by_logit);
        cands.truncate(top_k);
    }
    if top_p < 1.0 && cands.len() > 1 {
        cands.sort_unstable_by(by_logit);
        let total: f32 = cands.iter().map(|&(_, v)| ((v - max) / temp).exp()).sum();
        let limit = top_p * total;
        let mut acc = 0.0f32;
        let mut keep = cands.len();
        for (i, &(_, v)) in cands.iter().enumerate() {
            acc += ((v - max) / temp).exp();
            if acc >= limit {
                keep = i + 1;
                break;
            }
        }
        cands.truncate(keep.max(1));
    }
}

/// Fresh entropy seed from time and a monotonic counter.
fn fresh_seed() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos
        ^ NEXT
            .fetch_add(1, AtomicOrdering::Relaxed)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
}
