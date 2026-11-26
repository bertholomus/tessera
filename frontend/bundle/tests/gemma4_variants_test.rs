//! Tests Gemma 4 E2B, 12B, and 31B variant configs.

use tessera_bundle::load;
use tessera_engine::Linear;
use tessera_engine::gemma4::{Config, program, stepper};
use tessera_engine::keys::{Keys, Offered};

const MEDERTAS_31B: &str = r#"{
    "model_type": "gemma4",
    "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 5376,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 21504,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": false,
        "attention_k_eq_v": true,
        "enable_moe_block": false,
        "use_double_wide_mlp": false,
        "tie_word_embeddings": true,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": 262144,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "layer_types": [
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
        ],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
        }
    }
}"#;

#[test]
fn gemma4_variants_trace() {
    let dir = tessera_testkit::scratch(&format!("{}-gemma4-var", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let unified = MEDERTAS_31B
        .replace("\"gemma4\"", "\"gemma4_unified\"")
        .replace("\"gemma4_text\"", "\"gemma4_unified_text\"");
    std::fs::write(dir.join("config.json"), unified).unwrap();
    let err = load(&dir, 1, 262_144).err().unwrap().to_string();
    assert!(!err.contains("unknown architecture"), "got: {err}");
    assert!(err.contains("safetensors"), "got: {err}");
    let _ = std::fs::remove_dir_all(dir);

    let offered = [Offered::parse("config.json", MEDERTAS_31B).unwrap()];
    let mut keys = Keys::new(&offered);
    let c31 = Config::read(&mut keys, 1, 262_144);
    assert!(keys.finish().is_ok());
    assert_eq!(c31.full_kv_heads, 4);
    assert_eq!(c31.per_layer, 0);
    assert!(c31.k_eq_v);
    let g31 = stepper::<Linear>(&c31);
    assert_eq!(g31.kept.len(), 120);

    let mut e2b = tessera_engine::gemma4::small();
    e2b.double_wide_mlp = true;
    let f = program::<Linear>(&e2b);
    let s = stepper::<Linear>(&e2b);
    assert_eq!(f.weights.len(), s.weights.len());
}
