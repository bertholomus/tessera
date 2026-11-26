use crate::source::{Origin, Source, split};
use tessera_engine::progress::Progress;

fn origin(value: &str) -> Option<Origin> {
    let given = ["--model".to_string(), value.to_string()];
    split(&given).ok()?.0
}

#[test]
fn parses_hub_models() {
    let hub = |v| match origin(v) {
        Some(Origin::Hub(spec)) => Some(spec),
        _ => None,
    };
    assert_eq!(hub("hf:Qwen/Qwen3-0.6B").unwrap(), "Qwen/Qwen3-0.6B");
    assert_eq!(hub("hf:a/b@main").unwrap(), "a/b@main");
    assert!(matches!(origin("models/qwen"), Some(Origin::Model(_))));
    for bad in ["hf:", "hf:a", "hf:a/", "hf:a/b/c", "hf:a/b@"] {
        assert!(origin(bad).is_none(), "{bad}");
    }
}

/// A recipe's module rules ride in the serve options.
#[test]
fn options_carry_rules() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/recipes/glm53-moe"
    );
    let Ok(mut source) = Source::open(Some(Origin::Recipe(dir.into())), Progress::default()) else {
        panic!("the recipe opens");
    };
    source.cache = Some(tessera_testkit::scratch("recipes"));
    let Ok(options) = source.options(&[]) else {
        panic!("the recipe gives options");
    };
    let lower = options.lower.unwrap();
    let names: Vec<&str> = lower.rules.iter().map(|r| r.name).collect();
    assert_eq!(names, ["glm53_moe"]);
    assert_eq!(lower.favor.as_deref(), Some("glm53_moe"));
    assert!(options.module.is_some());
}

/// The built-in recipe reads, with its weights and a manifest.
#[test]
fn default_recipe_reads() {
    let Ok(source) = Source::open_or_kind(None, None, Progress::default()) else {
        panic!("the default recipe opens");
    };
    let recipe = source.recipe.unwrap();
    assert!(recipe.weights.hf.is_some());
    assert!(recipe.flags().iter().any(|(flag, _)| flag == "--manifest"));
}

/// `--recipe` and `--model` pair without fetching recipe weights.
#[test]
fn pairs_model_override() {
    let forward: Vec<String> = ["--recipe", "gemma-4-e4b", "--model", "local/weights"]
        .into_iter()
        .map(String::from)
        .collect();
    let reverse: Vec<String> = ["--model", "local/weights", "--recipe", "gemma-4-e4b"]
        .into_iter()
        .map(String::from)
        .collect();
    for given in [forward, reverse] {
        let (origin, rest) = split(&given).expect("paired flags split");
        assert!(rest.is_empty());
        assert!(matches!(origin, Some(Origin::Paired(_, _))));
        let mut source = Source::open(origin, Progress::default()).expect("source opens");
        source.cache = None;
        let options = source.options(&[]).expect("options build without cache");
        assert_eq!(options.model, std::path::Path::new("local/weights"));
        assert_eq!(options.name.as_deref(), Some("gemma-4-e4b"));
        assert_eq!(options.kv_dtype, None);
    }
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/recipes/glm53-moe"
    );
    let paired: Vec<String> = ["--recipe", dir, "--model", "local/weights"]
        .into_iter()
        .map(String::from)
        .collect();
    let (origin, _) = split(&paired).unwrap();
    let mut source = Source::open(origin, Progress::default()).unwrap();
    source.cache = Some(tessera_testkit::scratch("recipes"));
    let options = source.options(&[]).unwrap();
    assert_eq!(options.model, std::path::Path::new("local/weights"));
    assert_eq!(options.lower.unwrap().favor.as_deref(), Some("glm53_moe"));
    assert!(options.module.is_some());
    let dup_recipe: Vec<String> = ["--recipe", "gemma-4-e4b", "--recipe", "qwen3-0.6b"]
        .into_iter()
        .map(String::from)
        .collect();
    assert!(split(&dup_recipe).is_err());
    let dup_model: Vec<String> = ["--model", "a", "--model", "b"]
        .into_iter()
        .map(String::from)
        .collect();
    assert!(split(&dup_model).is_err());
}
