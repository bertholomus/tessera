//! Verifies dense granite decoder registration and config parsing.

use tessera_bundle::load;

const CONFIG: &str = r#"{
    "model_type": "granite",
    "vocab_size": 49159,
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000000.0,
    "attention_bias": false,
    "mlp_bias": false,
    "tie_word_embeddings": false,
    "embedding_multiplier": 1.0,
    "attention_multiplier": 0.015625,
    "residual_multiplier": 1.0,
    "logits_scaling": 1.0
}"#;

#[test]
fn granite_dense_registers() {
    let dir = tessera_testkit::scratch(&format!("{}-granite-dense", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    let msg = load(&dir, 1, 131_072).err().unwrap().to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!msg.contains("unknown architecture"), "unexpected: {msg}");
    assert!(msg.contains("safetensors"), "expected weights error: {msg}");
}
