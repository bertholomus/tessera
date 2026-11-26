//! Metal tile-streaming trellis matmul for decode and prefill.

use tessera_graph::{Op, Order, Stage};
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

const BLOCK: usize = 128;
const THREADS: usize = 256;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("trellis_metal.metal");
const DEC_SMEM: usize = 4 * 128 * 4;
const PF16_SMEM: usize = (16 * 132 + 8 * 16 * 20) * 4;
const PF32_SMEM: usize = (32 * 132 + 8 * 16 * 20) * 4;
const GUESS: f64 = 2.0e-11;

fn parse_form(name: &str) -> Option<(u32, u32)> {
    match name {
        "trellis_k2" => Some((2, 0)),
        "trellis_k3" => Some((3, 0)),
        "trellis_k4" => Some((4, 0)),
        "trellis_k2_mul1" => Some((2, 1)),
        "trellis_k3_mul1" => Some((3, 1)),
        "trellis_k4_mul1" => Some((4, 1)),
        "trellis_k2_pmcgx" => Some((2, 2)),
        "trellis_k3_pmcgx" => Some((3, 2)),
        "trellis_k4_pmcgx" => Some((4, 2)),
        _ => None,
    }
}

fn split_k(m: usize, n: usize, k: usize) -> usize {
    let ibs = k / BLOCK;
    if m > 4 || ibs <= 1 {
        return 1;
    }
    let goal: usize = if m <= 2 { 40 } else { 20 };
    let want = goal.div_ceil(n / BLOCK).clamp(1, 8).min(ibs);
    (1..=ibs.min(8))
        .filter(|d| ibs.is_multiple_of(*d))
        .min_by_key(|d| (d.abs_diff(want), *d < want))
        .unwrap_or(1)
}

fn fixed(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry,
    }
}

fn tile_rows(m: usize) -> usize {
    if m <= 4 {
        m
    } else if m <= 16 {
        16
    } else {
        32
    }
}

fn entry(t: &TrellisMetal) -> &'static str {
    let m = t.shape[0];
    match (m, t.book, t.bits) {
        (1, 0, 2) => "tm_k2_d1",
        (1, 0, 3) => "tm_k3_d1",
        (1, 0, 4) => "tm_k4_d1",
        (1, _, _) => "tm_gen_d1",
        (2..=4, _, _) => "tm_dec4",
        (5..=16, 0, 2) => "tm_k2_pf16",
        (5..=16, 0, 3) => "tm_k3_pf16",
        (5..=16, 0, 4) => "tm_k4_pf16",
        (5..=16, _, _) => "tm_gen_pf16",
        (_, 0, 2) => "tm_k2_pf32",
        (_, 0, 3) => "tm_k3_pf32",
        (_, 0, 4) => "tm_k4_pf32",
        _ => "tm_gen_pf32",
    }
}

fn smem_bytes(m: usize) -> usize {
    if m <= 4 {
        DEC_SMEM
    } else if m <= 16 {
        PF16_SMEM
    } else {
        PF32_SMEM
    }
}

fn metal_needs(t: &TrellisMetal) -> Needs {
    let m = t.shape[0];
    Needs {
        lanes: 32,
        group_bytes: smem_bytes(m),
        group_threads: THREADS,
        tile: m > 4,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

/// Dense Metal trellis matmul with tile streaming and MMA prefill.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "trellis_turned" } else { "trellis_metal" }),
    sizes = [self.shape[0], self.shape[1], self.shape[2]],
    work = self.shape[0] * self.shape[1] * self.shape[2],
    guess = GUESS,
    accepts = accepts,
    emits = f32_only,
    needs = metal_needs(self),
    order = Order::PerConfig,
    source = fixed(entry(self)),
    launches = launches,
    load = load,
)]
struct TrellisMetal {
    shape: [usize; 3],
    splits: usize,
    bits: u32,
    book: u32,
    turned: bool,
    form: &'static dyn Stored,
}

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn accepts(k: &TrellisMetal, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        _ => form.name() == "f32",
    }
}

fn load(t: &TrellisMetal, _: usize) -> Option<Load> {
    let [m, n, k] = t.shape;
    let down = m.div_ceil(tile_rows(m));
    let groups = (n / BLOCK) * t.splits * down;
    let mut reads = vec![t.form.bytes(n * k) as f64, (4 * m * k) as f64];
    if t.turned {
        reads.push((4 * k) as f64);
        if t.splits == 1 {
            reads.push((4 * n) as f64);
        }
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (k / BLOCK / t.splits) as f64,
        reads,
        written: (4 * t.splits * m * n) as f64,
        flight: 1024.0,
        group_bytes: smem_bytes(m) as f64,
        ..Load::default()
    })
}

fn launches(t: &TrellisMetal, step: &Step) -> Vec<Launch> {
    let [m, n, k] = t.shape;
    let shape = Shape::Grid {
        across: (n / BLOCK) * t.splits,
        down: m.div_ceil(tile_rows(m)),
        threads: THREADS,
    };
    let suh = if t.turned { step.read(2) } else { step.read(0) };
    let svh = if t.turned && t.splits == 1 {
        step.read(3)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        suh,
        svh,
        step.out(),
        Arg::count(m),
        Arg::count(n),
        Arg::count(k),
        Arg::count(t.splits),
        Arg::value(&t.bits),
        Arg::value(&t.book),
        Arg::count(usize::from(t.turned)),
    ];
    vec![Launch { args, shape }]
}

/// Deterministic split-K reduction and optional output rotation on Metal.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "trellis_reduce",
    sizes = [self.shape[0], self.shape[1], self.splits],
    work = self.shape[0] * self.shape[1] * self.splits,
    guess = GUESS,
    accepts = reduce_accepts,
    emits = f32_only,
    needs = reduce_needs(),
    order = Order::PerConfig,
    source = fixed("tm_reduce"),
    launches = reduce_launches,
    load = reduce_load,
)]
struct MetalReduce {
    shape: [usize; 2],
    splits: usize,
    turned: bool,
}

fn reduce_accepts(_: &MetalReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn reduce_needs() -> Needs {
    Needs {
        lanes: 32,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn reduce_load(r: &MetalReduce, _: usize) -> Option<Load> {
    let [m, n] = r.shape;
    let mut reads = vec![(4 * r.splits * m * n) as f64];
    if r.turned {
        reads.push((4 * n) as f64);
    }
    Some(Load {
        threads: (n / BLOCK) * THREADS,
        group: THREADS,
        trips: r.splits as f64,
        reads,
        written: (4 * m * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_launches(r: &MetalReduce, step: &Step) -> Vec<Launch> {
    let [m, n] = r.shape;
    let shape = Shape::Grid {
        across: n / BLOCK,
        down: 1,
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let args = vec![
        step.read(0),
        svh,
        step.out(),
        Arg::count(m),
        Arg::count(n),
        Arg::count(r.splits),
        Arg::count(usize::from(r.turned)),
    ];
    vec![Launch { args, shape }]
}

fn step_args<'a>(tree: &'a Tree, t: T, term: Term) -> Option<&'a [T]> {
    let node = tree.node(t);
    (node.term == term && !tree.root(t)).then_some(&node.args[..])
}

fn drafts(t: TrellisMetal, x: T, w: T, suh: Option<T>, svh: Option<T>) -> Vec<Draft> {
    let [m, n, _] = t.shape;
    let (splits, turned) = (t.splits, t.turned);
    if splits == 1 {
        let mut reads = vec![x, w];
        if let (Some(u), Some(v)) = (suh, svh) {
            reads.extend([u, v]);
        }
        return vec![Draft::new(t, reads, m * n)];
    }
    let mut s_reads = vec![Read::Node(x), Read::Node(w)];
    if let Some(u) = suh {
        s_reads.push(Read::Node(u));
    }
    let mut r_reads = vec![Read::Draft(0)];
    if let Some(v) = svh {
        r_reads.push(Read::Node(v));
    }
    let red = MetalReduce {
        shape: [m, n],
        splits,
        turned,
    };
    vec![
        Draft::new(t, s_reads, splits * m * n),
        Draft::new(red, r_reads, m * n),
    ]
}

fn check(tree: &Tree, x: T, w: T, turned: bool) -> Option<Result<TrellisMetal, String>> {
    let form = tree.node(w).form;
    if !form.name().starts_with("trellis") {
        return None;
    }
    let Some((bits, book)) = parse_form(form.name()) else {
        return Some(Err(format!("unsupported trellis form {}", form.name())));
    };
    let [n, k] = tree.node(w).shape[..] else {
        return Some(Err("trellis weight is not 2D".into()));
    };
    if !tree.leaf(w) {
        return Some(Err("trellis weight is not a direct leaf".into()));
    }
    if !n.is_multiple_of(BLOCK) || !k.is_multiple_of(BLOCK) {
        return Some(Err(format!("shape {n}x{k} must divide by {BLOCK}")));
    }
    if tree.node(x).form.name() != "f32" {
        return Some(Err(format!("input form {}", tree.node(x).form.name())));
    }
    let m = tree.elements(x) / k;
    if m == 0 {
        return Some(Err("m=0".into()));
    }
    Some(Ok(TrellisMetal {
        shape: [m, n, k],
        splits: split_k(m, n, k),
        bits,
        book,
        turned,
        form,
    }))
}

fn eval_metal(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let turn = Term::Stage(Stage::Hadamard {
        block: BLOCK as u32,
    });
    let mul = Term::Stage(Stage::Mul);
    let node = tree.node(at);
    if node.term == mul {
        let (back, svh) = (node.args[0], *node.args.get(1)?);
        let &[product] = step_args(tree, back, turn.clone())? else {
            return None;
        };
        let &[turned, w] = step_args(tree, product, Term::Op(Op::Matmul))? else {
            return None;
        };
        let &[scaled] = step_args(tree, turned, turn)? else {
            return None;
        };
        let &[x, suh] = step_args(tree, scaled, mul)? else {
            return None;
        };
        if node.form.name() != "f32" {
            return Some(Err(format!("output form {}", node.form.name())));
        }
        let kernel = match check(tree, x, w, true)? {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        if tree.elements(suh) != kernel.shape[2] || tree.elements(svh) != kernel.shape[1] {
            return Some(Err("rotation scale length mismatch".into()));
        }
        if tree.node(suh).form.name() != "f32" || tree.node(svh).form.name() != "f32" {
            return Some(Err("rotation scales must be f32".into()));
        }
        return Some(Ok(drafts(kernel, x, w, Some(suh), Some(svh))));
    }
    if node.term == Term::Op(Op::Matmul) {
        let (x, w) = (node.args[0], node.args[1]);
        if !tree.node(w).form.name().starts_with("trellis") {
            return None;
        }
        if node.form.name() != "f32" {
            return Some(Err(format!("output form {}", node.form.name())));
        }
        let kernel = match check(tree, x, w, false)? {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        return Some(Ok(drafts(kernel, x, w, None, None)));
    }
    None
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    eval_metal(tree, at)?.ok()
}

fn why(tree: &Tree, at: T) -> Option<String> {
    eval_metal(tree, at)?.err()
}

tessera_lower::rule!("trellis_metal", propose, why = why, lacks = LACKS);
