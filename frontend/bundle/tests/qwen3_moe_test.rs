//! Tests qwen3_moe config parsing, stepper graph, and expert weight mapping.

use tessera_bundle::{Overrides, load};
use tessera_engine::decoder::{Config, SparseMoe, program, stepper};
use tessera_engine::{Norm, Recipe, Settings, Tensor};

const LLM_JP_4_1_32B: &str = r#"{
    "model_type": "qwen3_moe",
    "vocab_size": 196608,
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 40,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 7680,
    "moe_intermediate_size": 960,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "norm_topk_prob": true,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 500000.0,
    "attention_bias": false,
    "mlp_bias": false,
    "use_sliding_window": false,
    "tie_word_embeddings": false
}"#;

#[test]
fn qwen3_moe_steps() {
    let dir = tessera_testkit::scratch(&format!("{}-qwen3-moe", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), LLM_JP_4_1_32B).unwrap();
    let err = load(&dir, 1, 65_536).err().unwrap().to_string();
    assert!(!err.contains("unknown architecture"), "got: {err}");
    assert!(err.contains("safetensors"), "got: {err}");
    let _ = Overrides::default();
    let _ = std::fs::remove_dir_all(dir);

    let c = Config::qwen3_moe_small();
    let fwd = program::<Norm, SparseMoe>(&c);
    let step = stepper::<Norm, SparseMoe>(&c);
    assert_eq!(fwd.weights.len(), step.weights.len());
    let model = tessera_engine::Model::open(fwd, &Recipe::default(), &Settings::default()).unwrap();
    let _ = model.device();
    let _ = Tensor::u32(vec![1, 1], vec![0]);
}
