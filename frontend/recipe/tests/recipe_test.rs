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

#[test]
fn hy4_flash_recipes_read() {
    for (name, slots, pool, extra) in [
        ("hy4-flash-pro6000", "4", "1048576", &[][..]),
        ("hy4-flash-gb10x1", "4", "4194304", &[][..]),
        (
            "hy4-flash-gb10x2",
            "8",
            "8388608",
            &[("--moe", Some("tp")), ("--sums", Some("arx"))][..],
        ),
    ] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/recipes")
            .join(name);
        let recipe = Recipe::read(&dir).unwrap();
        assert_eq!(recipe.name, name);
        let flags = recipe.flags();
        let mut want = vec![
            ("--task", Some("generate")),
            ("--manifest", Some("k4")),
            ("--max-model-len", Some("1048576")),
            ("--slots", Some(slots)),
            ("--kv-dtype", Some("fp8")),
            ("--kv-cache-tokens", Some(pool)),
        ];
        want.extend_from_slice(extra);
        for (f, v) in want {
            assert!(
                flags.iter().any(|(k, val)| k == f && val.as_deref() == v),
                "{name}: missing flag ({f:?}, {v:?})"
            );
        }
    }
}
