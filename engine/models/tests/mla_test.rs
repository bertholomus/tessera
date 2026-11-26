use tessera_graph::{Op, Pairs, Stage};
use tessera_interp::{Tensor, run};
use tessera_models::Stack;
use tessera_models::mla::{Config, Index, Moe, program};

/// Indexer RoPE distinguishes keys by position.
#[test]
fn indexer_rotates_prefix() {
    let mut cfg = Config {
        stack: Stack::sized([4, 8, 1, 1, 4, 4], 1, 6),
        dense_layers: 1,
        nope: 4,
        rope: 4,
        rope_theta: 10.0,
        q_latent: 4,
        latent: 4,
        dense_ffn: 4,
        experts: 2,
        top_k: 1,
        route_scale: 1.0,
        index: Some(Index {
            heads: 1,
            dim: 8,
            keep: 1,
            every: 1,
            rope_interleave: true,
        }),
        eps: 1e-5,
    };
    let g = program::<Moe>(&cfg);
    let ropes = |g: &tessera_graph::Graph, want: Pairs| {
        g.nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.op,
                    Op::Row(r) if r.0.iter().any(|s| matches!(s, Stage::Rope { pairs, .. } if *pairs == want))
                )
            })
            .count()
    };
    // Attention and indexer each rotate queries and keys.
    assert_eq!(ropes(&g, Pairs::Adjacent), 4);
    cfg.index.as_mut().unwrap().rope_interleave = false;
    assert_eq!(ropes(&program::<Moe>(&cfg), Pairs::Halves), 2);

    // Every position shares the same unrotated index key.
    let mut ws = Vec::with_capacity(g.weights.len());
    for w in &g.weights {
        let shape: Vec<usize> =
            w.ty.shape
                .0
                .iter()
                .map(|e| e.value(&g.dims, &|_| None).unwrap() as usize)
                .collect();
        let n: usize = shape.iter().product();
        let mut data = vec![0.0f32; n];
        let last = w.path.last().map(|s| format!("{s:?}")).unwrap_or_default();
        let full = format!("{:?}", w.path);
        if full.contains("embed") {
            for t in 0..4 {
                data[t * 8] = 1.0;
                if t == 1 {
                    data[t * 8 + 1] = 1.0;
                }
            }
        } else if full.contains("k_bias") {
            data[0] = 1.0;
        } else if last.contains("norm") && !full.contains("k_norm") {
            data.fill(1.0);
        } else if full.contains("q_a") || full.contains("wq") || full.contains("weights") {
            data[0] = 1.0;
        } else if full.contains("kv_a") {
            data[1] = 1.0;
        } else if full.contains("kv_b") {
            data[4 * 4] = 1.0;
        } else if full.contains("head") {
            data[8 + 2] = 1.0;
        } else if full.contains("\"o\"") {
            data[2 * 4] = 1.0;
        }
        ws.push(Tensor::f32(shape, data));
    }
    let toks = Tensor::u32(vec![1, 6], vec![0, 0, 0, 0, 0, 1]);
    let pos = Tensor::u32(vec![1, 6], vec![0, 1, 2, 3, 4, 5]);
    let out = run(&g, &ws, 0, &[toks, pos]).unwrap();
    let logits = out[0].floats().unwrap();
    // Position 5 must attend to position 5, not position 0.
    let pos5_tok1 = logits[5 * 4 + 1];
    assert!(
        pos5_tok1 > 0.5,
        "expected RoPE indexer to pick pos 5, got logit {pos5_tok1}"
    );
}
