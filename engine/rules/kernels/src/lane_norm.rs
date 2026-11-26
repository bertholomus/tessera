//! RMS norm, one SIMD group or one block per row.

use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, ToArg, Tree, sites,
};

/// SIMD groups sharing one wide row in block mode.
pub(crate) const WARPS: u32 = 8;

/// Threads per row in block mode.
pub(crate) const BLOCK_THREADS: u32 = WARPS * LANES as u32;

/// Minimum width to split across a whole block.
pub(crate) const BLOCK_MIN: usize = 512;

/// Minimum width that prefers block mode for prefill batches.
const WIDE_BLOCK_MIN: usize = 1024;

/// Maximum rows where narrow block-wide RMS norm wins.
pub(crate) const BLOCK_ROWS: usize = 16;

/// Maximum rows where wide block-wide RMS norm wins.
const WIDE_BLOCK_ROWS: usize = 2048;

/// Maximum wide trips held in registers across passes.
pub(crate) const CACHE_TRIPS: u32 = 8;

/// Per-thread register slots for cached pass-1 values.
pub(crate) const CACHE_SLOTS: usize = (CACHE_TRIPS * WIDE) as usize;

/// Trip count cached in registers, or zero when too wide.
pub(crate) fn cache_trips(width: usize, block: bool) -> u32 {
    let stride = if block { BLOCK_THREADS as usize } else { LANES } * WIDE as usize;
    let trips = width.div_ceil(stride);
    if trips <= CACHE_TRIPS as usize {
        trips as u32
    } else {
        0
    }
}

/// Whether `rows` of `width` should run one block per row.
pub(crate) fn use_block(rows: usize, width: usize) -> bool {
    (rows <= BLOCK_ROWS && width >= BLOCK_MIN)
        || (rows <= WIDE_BLOCK_ROWS && width >= WIDE_BLOCK_MIN)
}

/// Reduces per-thread partial sums across all warps in a block.
#[kernel]
pub(crate) fn block_sum(acc: f32) -> f32 {
    let mut red = shared::<f32>(WARPS);
    if lane() == 0 {
        red[lid() / LANES as u32] = acc;
    }
    barrier();
    let mut tot = 0.0f32;
    for w_i in 0..WARPS {
        tot += red[w_i];
    }
    tot
}

/// Sums a row's squares, then scales the row.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn norm_lanes<const WIDE: u32, const AHEAD: u32, const WEIGHTED: bool>(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] block: bool,
    #[build] use_cache: bool,
    #[build] cached: u32,
) {
    let group_w = if block { BLOCK_THREADS } else { LANES as u32 };
    let row = gid() / group_w;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let first = if block { lid() } else { lane() };
    let mut vals = [0.0f32; CACHE_SLOTS];
    let mut acc = 0.0f32;
    if use_cache {
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let v = x.load4(at + i as u64);
                let idx = (t * WIDE) as usize;
                vals[idx] = v.x;
                vals[idx + 1] = v.y;
                vals[idx + 2] = v.z;
                vals[idx + 3] = v.w;
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
            let v = x.load4(at + i as u64);
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    acc = sum32(acc);
    let tot = if block { block_sum(acc) } else { acc };
    let inv = rsqrt(tot / width as f32 + eps);
    if use_cache {
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let idx = (t * WIDE) as usize;
                let mut o = F32x4::zero();
                if WEIGHTED {
                    let g = w.load4(i as u64);
                    o.x = vals[idx] * inv * g.x;
                    o.y = vals[idx + 1] * inv * g.y;
                    o.z = vals[idx + 2] * inv * g.z;
                    o.w = vals[idx + 3] * inv * g.w;
                } else {
                    o.x = vals[idx] * inv;
                    o.y = vals[idx + 1] * inv;
                    o.z = vals[idx + 2] * inv;
                    o.w = vals[idx + 3] * inv;
                }
                out.store4(at + i as u64, o);
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
            let v = x.load4(at + i as u64);
            let mut o = F32x4::zero();
            if WEIGHTED {
                let g = w.load4(i as u64);
                o.x = v.x * inv * g.x;
                o.y = v.y * inv * g.y;
                o.z = v.z * inv * g.z;
                o.w = v.w * inv * g.w;
            } else {
                o.x = v.x * inv;
                o.y = v.y * inv;
                o.z = v.z * inv;
                o.w = v.w * inv;
            }
            out.store4(at + i as u64, o);
        }
    }
}

/// An RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = match (self.weighted, self.block) {
        (true, true) => "block_norm",
        (false, true) => "block_rms",
        (true, false) => "lane_norm",
        (false, false) => "lane_rms",
    },
    launches = norm_launches,
    shape = row_shape(self.rows, self.block),
    work = self.rows * self.width,
    needs = row_needs(self.block),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = norm_accepts,
    emits = norm_emits,
    source = norm_source,
    load = norm_load,
    unproven,
)]
pub(crate) struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
    pub block: bool,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

/// Launch grid for `rows` in block or lane mode.
pub(crate) fn row_shape(rows: usize, block: bool) -> Shape {
    if block {
        Shape::Grid {
            across: rows,
            down: 1,
            threads: BLOCK_THREADS as usize,
        }
    } else {
        Shape::Line(rows * LANES)
    }
}

/// Device requirements for block or lane row kernels.
pub(crate) fn row_needs(block: bool) -> Needs {
    if block {
        Needs {
            lanes: LANES,
            group_bytes: 4 * WARPS as usize,
            group_threads: BLOCK_THREADS as usize,
            ..Needs::default()
        }
    } else {
        Needs::lanes()
    }
}

fn norm_accepts(k: &LaneNorm, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == k.w_form.name(),
        _ => !form.narrow(),
    }
}

fn norm_emits(k: &LaneNorm, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn norm_body(k: &LaneNorm) -> tessera_codegen::ir::Body {
    let cached = cache_trips(k.width, k.block);
    let use_cache = cached > 0;
    match k.weighted {
        true => norm_lanes_ir::<WIDE, AHEAD, true>(
            k.x_form, k.w_form, k.out_form, k.block, use_cache, cached,
        ),
        false => norm_lanes_ir::<WIDE, AHEAD, false>(
            k.x_form, k.w_form, k.out_form, k.block, use_cache, cached,
        ),
    }
}

fn norm_launches(k: &LaneNorm, step: &Step) -> Vec<Launch> {
    let body = norm_body(k);
    let scalar = |name: &str| match name {
        "rows" => k.rows.arg(),
        "width" => k.width.arg(),
        "eps" => k.eps.arg(),
        other => panic!("LaneNorm has no field `{other}`"),
    };
    vec![Launch::of(
        &body,
        step,
        false,
        &scalar,
        row_shape(k.rows, k.block),
    )]
}

fn norm_source(k: &LaneNorm) -> Source {
    let body = norm_body(k);
    let base = k.name();
    let tag = match cache_trips(k.width, k.block) {
        0 => String::new(),
        n => format!("_c{n}"),
    };
    let entry = match (k.x_form.name(), k.w_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_{base}{tag}"),
        (x, w, o) => format!("plan_{base}{tag}_{x}_{w}_{o}"),
    };
    Source::named(body, entry)
}

fn norm_load(k: &LaneNorm, _: usize) -> Option<Load> {
    let body = norm_body(k);
    let elems = k.rows * k.width;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let cached = cache_trips(k.width, k.block) > 0;
    let mut reads = if cached {
        vec![x_bytes]
    } else {
        vec![x_bytes, x_bytes]
    };
    if k.weighted {
        reads.push(k.w_form.bytes(k.width) as f64);
    }
    let per_row = if k.block {
        BLOCK_THREADS as usize
    } else {
        LANES
    };
    let trips = if cached {
        1.0
    } else {
        (k.width as f64 / (per_row * WIDE as usize) as f64).ceil()
    };
    Some(Load {
        threads: k.rows * per_row,
        group: BLOCK_THREADS as usize,
        trips,
        trip: body.trip(),
        reads,
        written: k.out_form.bytes(elems) as f64,
        flight: k.x_form.bytes((WIDE * AHEAD) as usize) as f64,
        ..Load::default()
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    if !site.width.is_multiple_of(WIDE as usize) {
        return None;
    }
    let x_form = plain(tree.node(site.x).form);
    let w_form = site
        .operand
        .map_or(x_form, |(o, _)| plain(tree.node(o).form));
    let out_form = writable(tree.node(at).form);
    let rows = site.len / site.width;
    let block = use_block(rows, site.width);
    let kernel = LaneNorm {
        rows,
        width: site.width,
        eps,
        weighted,
        block,
        x_form,
        w_form,
        out_form,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("lane_norm", propose);
