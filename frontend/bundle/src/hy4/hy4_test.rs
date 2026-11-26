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
