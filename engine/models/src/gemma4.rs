//! Gemma 4 text: a causal decoder with optional per-layer embeddings.

use crate::eg2::{Attention, FULL_HEAD_DIM, PER_LAYER, full_layers};
use crate::keys::{At, Keys};
use crate::shared::{Attend, Dense, Side, Stack, sizes};
use serde_json::json;
use tessera_trace as trace;
use tessera_trace::hf::*;
use trace::DimName::{Context, Ffn, HeadDim, Heads, KvHeads, Rows, Seqs};
use trace::{
    Act, Composite, Dim, DimExpr, DimName, Embed, Graph, Kept, Mask, Model, Norm, Pairs, PathSeg,
    Role, Scalar, Size, Tensor, TensorType, Weights,
};

/// Full-attention key-head count when different from sliding layers.
const FULL_KV_HEADS: DimName = DimName::named("FullKvHeads");
/// Double-wide MLP intermediate width on KV-shared layers (E2B).
const DOUBLE_FFN: DimName = DimName::named("DoubleFfn");

pub struct Config {
    /// Its `head_dim` is a sliding layer's.
    pub stack: Stack,
    pub kv_heads: u64,
    /// Full layers' key-head count.
    pub full_kv_heads: u64,
    /// Every n-th layer attends to the whole text.
    pub full_every: u64,
    pub full_head_dim: u64,
    /// Pairs a full layer's rotary turns.
    pub full_turned: u32,
    /// Per-layer embedding width. Zero omits PLE blocks.
    pub per_layer: u64,
    /// Keys a sliding layer sees, its own included.
    pub window: u64,
    pub theta: f32,
    pub full_theta: f32,
    /// Layers from this one on read an earlier layer's keys.
    pub first_shared: u64,
    /// Full layers reuse the key projection as values.
    pub k_eq_v: bool,
    /// Shared layers double the MLP intermediate width.
    pub double_wide_mlp: bool,
    pub eps: f32,
    /// Logits become `cap * tanh(logits / cap)`.
    pub cap: Option<f32>,
}

impl Config {
    /// The config `k` holds, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let text = At::text();
        let rope = text.under(ROPE_PARAMETERS);
        let [sliding, full] = [SLIDING_ATTENTION, FULL_ATTENTION].map(|key| rope.under(key));
        let model_type: String = k.read(&text, MODEL_TYPE);
        if !matches!(model_type.as_str(), "gemma4_text" | "gemma4_unified_text") {
            k.misfit(
                &text.path(MODEL_TYPE.name),
                "want gemma4_text or gemma4_unified_text",
            );
        }
        let fixed = [
            (HIDDEN_ACTIVATION.name, json!("gelu_pytorch_tanh")),
            (ATTENTION_BIAS.name, json!(false)),
            ("enable_moe_block", json!(false)),
            (TIED.name, json!(true)),
        ];
        for (key, want) in fixed {
            k.fix(&text, key, want);
        }
        k.fix(&sliding, ROPE_TYPE.name, json!("default"));
        k.fix(&full, ROPE_TYPE.name, json!("proportional"));
        let sizes = sizes(k, &text);
        let [vocab, _, layers, ..] = sizes;
        let per_layer: u64 = k.read_or(&text, PER_LAYER_INPUT, 0);
        if per_layer > 0 {
            k.fix(&text, "vocab_size_per_layer_input", json!(vocab));
        }
        let kv_heads = k.read(&text, KV_HEADS);
        let full_kv_heads = k
            .read_or(&text, "num_global_key_value_heads", None)
            .unwrap_or(kv_heads);
        let full_head_dim: u64 = k.read(&text, "global_head_dim");
        let part = k.read_or(&full, PARTIAL_ROTARY, 1.0) as f32;
        let shared_at = "num_kv_shared_layers";
        let shared: u64 = k.read_or(&text, shared_at, 0);
        let full_every = full_layers(k, layers);
        let first_shared = layers.saturating_sub(shared);
        // Each kind needs an own layer to keep its keys.
        if first_shared < layers && first_shared < full_every {
            let why = "a kind of layer has no earlier layer to read keys from";
            k.misfit(&text.path(shared_at), why);
        }
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads,
            full_kv_heads,
            full_every,
            full_head_dim,
            full_turned: (part * full_head_dim as f32) as u32 / 2,
            per_layer,
            window: k.read(&text, SLIDING_WINDOW),
            theta: k.read(&sliding, ROPE_THETA),
            full_theta: k.read(&full, ROPE_THETA),
            first_shared,
            k_eq_v: k.read_or(&text, "attention_k_eq_v", false),
            double_wide_mlp: k.read_or(&text, "use_double_wide_mlp", false),
            eps: k.read(&text, EPS),
            cap: k.read_or(&text, "final_logit_softcapping", None),
        }
    }

    /// Whether layer `l` attends to the whole text.
    pub fn full(&self, l: u64) -> bool {
        (l + 1).is_multiple_of(self.full_every)
    }

    /// Whether layer `l` is its kind's last own layer.
    pub fn hands_on(&self, l: u64) -> bool {
        if self.first_shared >= self.stack.layers {
            return false;
        }
        let later = (l + 1..self.first_shared).any(|n| self.full(n) == self.full(l));
        l < self.first_shared && !later
    }

    /// Layer `l`'s head width.
    pub fn width(&self, l: u64) -> DimExpr {
        match self.full(l) {
            true => FULL_HEAD_DIM.into(),
            false => HeadDim.into(),
        }
    }

    /// Layer `l`'s key-value head count.
    pub fn kv_dim(&self, l: u64) -> DimExpr {
        match self.full(l) && self.full_kv_heads != self.kv_heads {
            true => FULL_KV_HEADS.into(),
            false => KvHeads.into(),
        }
    }

    /// Layer `l`'s MLP intermediate width.
    pub fn ffn_dim(&self, l: u64) -> DimExpr {
        match self.double_wide_mlp && l >= self.first_shared {
            true => DOUBLE_FFN.into(),
            false => Ffn.into(),
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        let mut extra = vec![
            (KvHeads, self.kv_heads),
            (FULL_HEAD_DIM, self.full_head_dim),
        ];
        if self.full_kv_heads != self.kv_heads {
            extra.push((FULL_KV_HEADS, self.full_kv_heads));
        }
        if self.per_layer > 0 {
            extra.push((PER_LAYER, self.per_layer));
        }
        if self.double_wide_mlp {
            extra.push((DOUBLE_FFN, self.stack.ffn * 2));
        }
        self.stack.dims(&extra)
    }
}

/// Full layer whose values reuse `k`: `S0` width, `S1` heads.
#[derive(Weights)]
pub struct TiedKv<D: Dense> {
    #[shape(Hidden)]
    pub ln: Norm,
    #[shape(Hidden, Heads * S0)]
    pub q: D,
    #[shape(Hidden, S1 * S0)]
    pub k: D,
    #[shape(S0)]
    pub q_norm: Norm,
    #[shape(S0)]
    pub k_norm: Norm,
    #[shape(Heads * S0, Hidden)]
    pub o: D,
    #[shape(Hidden)]
    pub post: Norm,
}

/// A layer that reads another's keys: `S0` head width.
#[derive(Weights)]
pub struct Queries<D: Dense> {
    #[shape(Hidden)]
    pub ln: Norm,
    #[shape(Hidden, Heads * S0)]
    pub q: D,
    #[shape(S0)]
    pub q_norm: Norm,
    #[shape(Heads * S0, Hidden)]
    pub o: D,
    #[shape(Hidden)]
    pub post: Norm,
}

/// One layer's attention weights.
pub enum Block<D: Dense> {
    Own(Attention<D>),
    Tied(TiedKv<D>),
    Shared(Queries<D>),
}

/// One layer's feed-forward block and scalar: `S0` width.
#[derive(Weights)]
struct FeedForward<D: Dense> {
    #[shape(Hidden)]
    pub pre_ff: Norm,
    #[shape(Hidden, S0)]
    pub gate: D,
    #[shape(Hidden, S0)]
    pub up: D,
    #[shape(S0, Hidden)]
    pub down: D,
    #[shape(Hidden)]
    pub post_ff: Norm,
    #[shape()]
    pub scalar: Scalar,
}

/// One layer's per-layer embedding gate weights.
#[derive(Weights)]
struct PleGate<D: Dense> {
    /// This layer's rows of the shared per-layer projection.
    #[shape(Hidden, PER_LAYER)]
    pub ple_in: D,
    #[shape(Hidden, PER_LAYER)]
    pub ple_gate: D,
    #[shape(PER_LAYER, Hidden)]
    pub ple_proj: D,
    #[shape(Hidden)]
    pub ple_norm: Norm,
}

/// Shared PLE weights, declared only when `per_layer > 0`.
#[derive(Weights)]
struct PleShared {
    /// Each token's per-layer input, the layers side by side.
    #[shape(Vocab, Layers * PER_LAYER)]
    pub ple_embed: Embed,
    #[shape(PER_LAYER)]
    pub ple_norm: Norm,
}

/// The non-layer weights outside attention.
#[derive(Weights)]
struct Gemma4 {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// Each layer's attention, shaped for its kind.
pub fn attention<D: Dense>(m: &Model, c: &Config) -> Vec<Block<D>> {
    let mut path = vec![PathSeg::Field("attention".into())];
    let mut out = Vec::new();
    for l in 0..c.stack.layers {
        path.push(PathSeg::Index(l as u32));
        let (width, kv) = (c.width(l), c.kv_dim(l));
        let heads = if l >= c.first_shared {
            Block::Shared(Queries::declare(m, &mut path, &[width]))
        } else if c.k_eq_v && c.full(l) {
            Block::Tied(TiedKv::declare(m, &mut path, &[width, kv]))
        } else {
            Block::Own(Attention::declare(m, &mut path, &[width, kv]))
        };
        path.pop();
        out.push(heads);
    }
    out
}

/// Each layer's feed-forward and optional PLE weights.
fn layer_blocks<D: Dense>(m: &Model, c: &Config) -> (Vec<FeedForward<D>>, Vec<PleGate<D>>) {
    let mut path = vec![PathSeg::Field("layers".into())];
    let (mut ffs, mut ples) = (Vec::new(), Vec::new());
    for l in 0..c.stack.layers {
        path.push(PathSeg::Index(l as u32));
        ffs.push(FeedForward::declare(m, &mut path, &[c.ffn_dim(l)]));
        if c.per_layer > 0 {
            ples.push(PleGate::declare(m, &mut path, &[]));
        }
        path.pop();
    }
    (ffs, ples)
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// Keys and values a layer attends over.
type Kv = [Tensor; 2];

/// State one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// Each row's slot and the kept rows, for ragged rows.
    step: Option<(&'a Tensor, &'a [Cache])>,
}

impl Pass<'_> {
    /// Axes before the hidden one.
    fn lead(&self) -> Vec<DimExpr> {
        match self.step {
            Some(_) => vec![Rows.into()],
            None => vec![Seqs.into(), Context.into()],
        }
    }

    /// The residual rows after the last layer.
    fn trunk<D: Dense>(
        &self,
        w: &Gemma4,
        ple_shared: Option<&PleShared>,
        layers: (&[FeedForward<D>], &[PleGate<D>]),
        heads: &[Block<D>],
        tokens: &Tensor,
    ) -> Tensor {
        let (m, c) = (self.m, self.c);
        let (eps, hidden, per) = (c.eps, c.stack.hidden as f32, c.per_layer);
        let (ffs, ples) = layers;
        let inputs = m.embed(tokens, &w.embed).scale(hidden.sqrt());
        let table = ple_shared.map(|p| m.embed(tokens, &p.ple_embed).scale((per as f32).sqrt()));
        let last = self.lead().len() as u32;
        let mut x = inputs.clone();
        // Each kind's latest own keys and values, sliding first.
        let mut kept: [Option<Kv>; 2] = [None, None];
        for (l, ff) in ffs.iter().enumerate() {
            let (l_u64, full) = (l as u64, c.full(l as u64) as usize);
            let shared = kept[full].clone();
            let attend = || self.attend(l_u64, &heads[l], &x, shared);
            // Each kind's last own layer hands its keys on.
            let attn = match c.hands_on(l_u64) {
                true => {
                    let [attn, k, v] = m.composite(Composite::Attention, attend);
                    kept[full] = Some([k, v]);
                    attn
                }
                false => {
                    let [attn] = m.composite(Composite::Attention, || [attend()[0].clone()]);
                    attn
                }
            };
            let x_mid = x.add(&attn);
            let h = x_mid.rms_norm(&ff.pre_ff, eps);
            let [mlp] = m.composite(Composite::Mlp, || {
                let inner = ff.gate.apply(&h).act(Act::GeluTanh).mul(&ff.up.apply(&h));
                [ff.down.apply(&inner).rms_norm(&ff.post_ff, eps)]
            });
            let mut out = x_mid.add(&mlp);
            if let (Some(tab), Some(ps)) = (&table, ple_shared) {
                let pg = &ples[l];
                let own = tab.slice(last, l_u64 * per, per);
                let ctx = pg
                    .ple_in
                    .apply(&inputs)
                    .scale(hidden.powf(-0.5))
                    .rms_norm(&ps.ple_norm, eps);
                let ple = ctx.add(&own).scale(0.5f32.sqrt());
                let gated = pg.ple_gate.apply(&out).act(Act::GeluTanh).mul(&ple);
                let o = pg.ple_proj.apply(&gated).rms_norm(&pg.ple_norm, eps);
                out = out.add(&o);
            }
            x = out.mul_scalar(&ff.scalar);
        }
        x
    }

    /// The output, then the keys and values it read.
    fn attend<D: Dense>(
        &self,
        l: u64,
        heads: &Block<D>,
        x: &Tensor,
        shared: Option<Kv>,
    ) -> [Tensor; 3] {
        let c = self.c;
        let full = c.full(l);
        let (theta, mask, turned) = match full {
            true => (c.full_theta, Mask::Causal, Some(c.full_turned)),
            false => (c.theta, Mask::Back(c.window - 1), None),
        };
        let (ln, q, q_norm, o, post) = match heads {
            Block::Own(a) => (&a.ln, &a.q, &a.q_norm, &a.o, &a.post),
            Block::Tied(a) => (&a.ln, &a.q, &a.q_norm, &a.o, &a.post),
            Block::Shared(a) => (&a.ln, &a.q, &a.q_norm, &a.o, &a.post),
        };
        let h = x.rms_norm(ln, c.eps);
        let shape = [Heads.into(), c.kv_dim(l), c.width(l)];
        let at = Attend::new(o, self.lead(), shape);
        let at = at
            .rotary(self.positions, theta, Pairs::Halves)
            .turning(turned);
        let turn = |t: Tensor, side: Side| match side {
            Side::Query => t.rms_norm(q_norm, c.eps),
            Side::Key => match heads {
                Block::Own(a) => t.rms_norm(&a.k_norm, c.eps),
                Block::Tied(a) => t.rms_norm(&a.k_norm, c.eps),
                Block::Shared(_) => t,
            },
            Side::Value => t.rms(c.eps),
        };
        let q = at.side(q, &h, Side::Query, turn);
        let [k, v] = match heads {
            Block::Own(a) => {
                let [k, v] = [(&a.k, Side::Key), (&a.v, Side::Value)];
                let [k, v] = [k, v].map(|(w, side)| at.side(w, &h, side, turn));
                self.append_kv(l, [k, v])
            }
            Block::Tied(a) => {
                let [k, v] = at.tied_kv(&a.k, &h, turn);
                self.append_kv(l, [k, v])
            }
            Block::Shared(_) => shared.expect("an earlier layer's keys"),
        };
        let out = match self.step {
            Some((slots, _)) => {
                let by = [slots, self.positions];
                self.m.attend_rows(mask, 1.0, [&q, &k, &v], by, None)
            }
            None => self.m.attend(mask, 1.0, [&q, &k, &v], None),
        };
        [at.merge(out).rms_norm(post, c.eps), k, v]
    }

    fn append_kv(&self, l: u64, [k, v]: Kv) -> Kv {
        match self.step {
            Some((slots, caches)) => {
                let by = [slots, self.positions];
                let [keys, values] = &caches[l as usize];
                [k.append(keys, by), v.append(values, by)]
            }
            None => [k, v],
        }
    }

    /// Logits through the tied embedding.
    fn logits<D: Dense>(
        &self,
        w: &Gemma4,
        ple_shared: Option<&PleShared>,
        layers: (&[FeedForward<D>], &[PleGate<D>]),
        heads: &[Block<D>],
        tokens: &Tensor,
    ) -> Tensor {
        let [logits] = self.m.composite(Composite::Model, || {
            let x = self.trunk(w, ple_shared, layers, heads, tokens);
            [x.rms_norm(&w.norm, self.c.eps).unembed(&w.embed)]
        });
        match self.c.cap {
            Some(cap) => logits.scale(cap.recip()).act(Act::Tanh).scale(cap),
            None => logits,
        }
    }
}

/// Four layers, sliding then full. The last two share keys.
pub fn small() -> Config {
    let mut stack = Stack::small(256, 128, 256);
    stack.layers = 4;
    Config {
        stack,
        kv_heads: 1,
        full_kv_heads: 1,
        full_every: 2,
        full_head_dim: 256,
        full_turned: 32,
        per_layer: 128,
        window: 4,
        theta: 1e4,
        full_theta: 1e6,
        first_shared: 2,
        k_eq_v: false,
        double_wide_mlp: false,
        eps: 1e-6,
        cap: Some(30.0),
    }
}

crate::preset::preset!("gemma4", small: || stepper::<trace::Linear>(&small()));

/// One forward entry: tokens and positions in, logits out.
pub fn program<D: Dense>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Gemma4 = trace::declare(&m);
    let ple = (c.per_layer > 0).then(|| trace::declare::<PleShared>(&m));
    let (ffs, ples) = layer_blocks::<D>(&m, c);
    let heads = attention(&m, c);
    let [tokens, positions] = tessera_trace::tokens(&m);
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: None,
    };
    let logits = pass.logits(&w, ple.as_ref(), (&ffs, &ples), &heads, &tokens);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<D: Dense>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Gemma4 = trace::declare(&m);
    let ple = (c.per_layer > 0).then(|| trace::declare::<PleShared>(&m));
    let (ffs, ples) = layer_blocks::<D>(&m, c);
    let heads = attention(&m, c);
    let masked = c.stack.steps == crate::Steps::Masked;
    let ([tokens, positions, slots], allowed) = tessera_trace::step_ports(&m, masked);
    let mut caches = Vec::new();
    for l in 0..c.first_shared {
        let ty = TensorType::new(
            vec![Seqs.into(), Context.into(), c.kv_dim(l), c.width(l)],
            trace::DType::F32,
        );
        caches.push([m.kept(ty.clone()), m.kept(ty)]);
    }
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: Some((&slots, &caches)),
    };
    let logits = pass.logits(&w, ple.as_ref(), (&ffs, &ples), &heads, &tokens);
    tessera_trace::pick(&m, &logits, allowed.as_ref(), None)
}
