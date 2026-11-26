use std::path::Path;
use tessera_recipe::Recipe;

#[test]
fn glm53_gb10x4_reads() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/glm53-gb10x4");
    let recipe = Recipe::read(&dir).unwrap();
    assert_eq!(recipe.name, "glm53-gb10x4");
    assert_eq!(
        recipe.weights.hf.as_deref(),
        Some("kindlingai/glm-5.3-exlr8-k2-k3-k4@9e25c28a2e8808d8b2599d780521fd6af24e7558")
    );
    assert_eq!(
        recipe.lower.as_ref().and_then(|l| l.favor.as_deref()),
        Some("glm53_moe")
    );
    assert!(recipe.module().unwrap().join("Cargo.toml").is_file());
    let flags = recipe.flags();
    for want in [
        ("--task", Some("generate")),
        ("--manifest", Some("bit280")),
        ("--max-model-len", Some("1048576")),
        ("--slots", Some("8")),
        ("--kv-dtype", Some("fp8")),
        ("--kv-cache-tokens", Some("1048576")),
        ("--reuse-prefix", None),
        ("--moe", Some("tp")),
        ("--sums", Some("arx")),
    ] {
        assert!(
            flags
                .iter()
                .any(|(f, v)| f == want.0 && v.as_deref() == want.1),
            "missing flag {want:?}"
        );
    }
}
