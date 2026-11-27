use super::*;

#[test]
fn parses_hy4_config() {
    let json = r#"{
        "model_type": "hy_v4_flash",
        "architectures": ["HYV4FlashForConditionalGeneration"],
        "text_config": {
            "vocab_size": 120832,
            "hidden_size": 3072,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "v_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "intermediate_size": 13824,
            "moe_intermediate_size": 1536,
            "routed_expert_hidden_size": 1536,
            "n_routed_experts": 512,
            "num_experts_per_tok": 16,
            "routed_scaling_factor": 3.998,
            "swiglu_limit": 10.0,
            "first_k_dense_replace": 1,
            "sliding_window": 512,
            "swa_pattern": [0, 0, 1, 1],
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "attention_bias": false,
            "rope_parameters": {
                "deepseek_sparse_attention": { "rope_theta": 10000000 },
                "sliding_attention": { "rope_theta": 10000 }
            }
        }
    }"#;
    let cfg = parse_config(json, [1, 16]).unwrap();
    assert_eq!(cfg.stack.vocab, 120_832);
    assert_eq!(cfg.stack.heads, 32);
    assert_eq!(cfg.swa_heads, 64);
    assert_eq!(cfg.swa_pattern, vec![false, false, true, true]);
    assert_eq!(cfg.swiglu_limit, Some(10.0));
    assert!(crate::arch::find("hy4").is_some());
}

fn small_engram_slices() -> Vec<tessera_checkpoint::Slice> {
    use tessera_checkpoint::{Slice, Stored};
    let s = |name: String, shape: Vec<usize>| Slice {
        name,
        shape,
        stored: Stored::F32,
    };
    let mut out = vec![
        s("model.embed_tokens.weight".into(), vec![32, 256]),
        s("model.norm.weight".into(), vec![256]),
        s("lm_head.weight".into(), vec![32, 256]),
        s("model.hc_head.hc_head_fn.weight".into(), vec![4, 1024]),
        s("model.hc_head.hc_head_base".into(), vec![4]),
        s("model.hc_head.hc_head_scale".into(), vec![1]),
    ];
    for l in 0..2 {
        let (at, nh) = (format!("model.layers.{l}"), if l == 0 { 2 } else { 4 });
        for hc in ["hc_attn_layer", "hc_mlp_layer"] {
            out.push(s(format!("{at}.{hc}.hc_pre.hc_fn.weight"), vec![8, 1024]));
            out.push(s(format!("{at}.{hc}.hc_pre.hc_base"), vec![8]));
            out.push(s(format!("{at}.{hc}.hc_pre.hc_scale"), vec![2]));
        }
        for (name, shape) in [
            ("input_layernorm.weight", vec![256]),
            ("self_attn.q_a_proj.weight", vec![8, 256]),
            ("self_attn.q_a_layernorm.weight", vec![8]),
            ("self_attn.q_b_proj.weight", vec![nh * 8, 8]),
            ("self_attn.kv_a_proj_with_mqa.weight", vec![12, 256]),
            ("self_attn.kv_a_layernorm.weight", vec![8]),
            ("self_attn.kv_b_proj.weight", vec![nh * 12, 8]),
            ("self_attn.linear_gate.weight", vec![nh * 8, 256]),
            ("self_attn.o_proj.weight", vec![256, nh * 8]),
            ("self_attn.post_proj_norm.weight", vec![256]),
            ("self_attn.learnable_sink_param", vec![nh]),
            ("post_attention_layernorm.weight", vec![256]),
        ] {
            out.push(s(format!("{at}.{name}"), shape));
        }
    }
    for (name, shape) in [
        ("model.layers.0.mlp.post_fc2_norm.weight", vec![256]),
        ("model.layers.0.mlp.gate_proj.weight", vec![16, 256]),
        ("model.layers.0.mlp.up_proj.weight", vec![16, 256]),
        ("model.layers.0.mlp.down_proj.weight", vec![256, 16]),
        (
            "model.layers.1.engram_layer.value_proj.weight",
            vec![256, 8],
        ),
        ("model.layers.1.engram_layer.value_proj.bias", vec![256]),
        ("model.layers.1.engram_layer.unified_pre_gamma", vec![256]),
        ("model.layers.1.mlp.gate.weight", vec![4, 256]),
        ("model.layers.1.mlp.gate.e_score_correction_bias", vec![4]),
        (
            "model.layers.1.mlp.routed_expert_down_proj.weight",
            vec![128, 256],
        ),
        (
            "model.layers.1.mlp.routed_expert_up_proj.weight",
            vec![256, 128],
        ),
        ("model.layers.1.mlp.combine_output_norm.weight", vec![256]),
        ("model.layers.1.mlp.experts.signs_in", vec![128]),
        ("model.layers.1.mlp.experts.signs_out", vec![128]),
    ] {
        out.push(s(name.into(), shape));
    }
    for h in 0..2 {
        out.push(s(
            format!("model.layers.1.engram_layer.lookup_per_head_norms.{h}.weight"),
            vec![4],
        ));
    }
    for hc in 0..4 {
        let eng = "model.layers.1.engram_layer";
        out.push(s(format!("{eng}.key_projs.{hc}.weight"), vec![256, 8]));
        out.push(s(format!("{eng}.key_projs.{hc}.bias"), vec![256]));
        out.push(s(format!("{eng}.norm_key.{hc}.weight"), vec![256]));
        out.push(s(format!("{eng}.norm_query.{hc}.weight"), vec![256]));
        out.push(s(format!("{eng}.unified_per_hc_scales.{hc}"), vec![256]));
    }
    for (p, shs) in [
        ("gate", vec![128, 256]),
        ("up", vec![128, 256]),
        ("down", vec![256, 128]),
    ] {
        let m1 = "model.layers.1.mlp";
        out.push(s(format!("{m1}.experts.{p}.weight"), vec![4, 128, 128]));
        out.push(s(format!("{m1}.experts.{p}.su"), vec![4, 128]));
        out.push(s(format!("{m1}.experts.{p}.sv"), vec![4, 128]));
        out.push(s(
            format!("{m1}.shared_experts.{p}_proj.weight"),
            shs.clone(),
        ));
        out.push(s(format!("{m1}.shared_experts.{p}_proj.suh"), vec![shs[1]]));
        out.push(s(format!("{m1}.shared_experts.{p}_proj.svh"), vec![shs[0]]));
    }
    out
}

#[test]
fn engram_rules_resolve() {
    let mut cfg = Config::hy4_small();
    (cfg.stack.hidden, cfg.stack.ffn, cfg.routed_hidden) = (256, 128, 128);
    cfg.engram = Some(tessera_engine::hy4::Engram {
        layers: vec![1],
        heads: 2,
        head_dim: 4,
    });
    let g = program::<RotatedLatent>(&cfg);
    let claims = crate::names::resolve(&g, &small_engram_slices(), &[&rules(&cfg, true)]).unwrap();
    assert_eq!(claims.len(), g.weights.len());
}
