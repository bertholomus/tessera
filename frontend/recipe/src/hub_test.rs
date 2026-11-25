use crate::hub::{exlr8_files, manifest_widths, safe_path, wanted};

fn owned(list: &[&str]) -> Vec<String> {
    list.iter().map(|n| n.to_string()).collect()
}

#[test]
fn diffusers_folders() {
    let keep = [
        "model_index.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "tokenizer/merges.txt",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ];
    let mut names = owned(&keep);
    let drop = [
        "flux-2-klein-4b.safetensors",
        "vae/x.bin",
        "assets/a.png",
        "extra/a.json",
    ];
    names.extend(owned(&drop));
    let folders = owned(&["text_encoder", "tokenizer", "transformer", "vae"]);
    assert_eq!(wanted(&names, Some(&folders)), keep);
}

#[test]
fn plain_repo() {
    let names = owned(&[
        "config.json",
        "model.safetensors",
        "x.bin",
        "onnx/config.json",
    ]);
    assert_eq!(wanted(&names, None), ["config.json", "model.safetensors"]);
    assert!(safe_path("vae/config.json"));
    for bad in ["../x.json", "a/../b.json", "/etc/x.json", "a\\b.json"] {
        assert!(!safe_path(bad), "{bad}");
    }
}

#[test]
fn exlr8_one_width() {
    let names = owned(&[
        "config.json",
        "dense/exlr8-dense-000.safetensors",
        "dense/exlr8-extra-audio.safetensors",
        "dense/exlr8-extra-vision.safetensors",
        "dense/exlr8-vocab.safetensors",
        "manifests/exlr8-manifest-bit309.json",
        "experts/exlr8-layer-001.index.safetensors",
        "experts/exlr8-layer-001.k2.safetensors",
        "experts/exlr8-layer-001.k3.safetensors",
        "experts/exlr8-layer-001.k4.safetensors",
        "hashes/exlr8-layer-001.k3.sha256",
        "index-slices/tp4/k3/rank0.safetensors",
    ]);
    let at = |k| exlr8_files(&names, k, None).join(" ");
    let head = "config.json dense/exlr8-dense-000.safetensors dense/exlr8-vocab.safetensors \
                manifests/exlr8-manifest-bit309.json";
    let experts = "experts/exlr8-layer-001";
    assert_eq!(
        at(4),
        format!("{head} {experts}.index.safetensors {experts}.k4.safetensors")
    );
    assert_eq!(
        at(2),
        format!(
            "{head} {experts}.index.safetensors {experts}.k2.safetensors {experts}.k3.safetensors"
        )
    );
}

#[test]
fn exlr8_manifest_widths() {
    let manifest = r#"{
        "layers": {
            "3": { "gate": [2, 2], "up": 2, "down": 2, "shared": { "gate": 4, "up": 4, "down": 4 } },
            "23": { "gate": 3, "up": 3, "down": 3, "shared": { "gate": 4, "up": 4, "down": 4 } },
            "78": { "gate": 3, "up": 3, "down": 3 }
        }
    }"#;
    let widths = manifest_widths(manifest).unwrap();
    let names = owned(&[
        "config.json",
        "dense/exlr8-vocab.safetensors",
        "manifests/exlr8-manifest-mixed.json",
        "experts/exlr8-layer-003.index.safetensors",
        "experts/exlr8-layer-003.k2.safetensors",
        "experts/exlr8-layer-003.k3.safetensors",
        "experts/exlr8-layer-003.k4.safetensors",
        "experts/exlr8-layer-023.index.safetensors",
        "experts/exlr8-layer-023.k2.safetensors",
        "experts/exlr8-layer-023.k3.safetensors",
        "experts/exlr8-layer-023.k4.safetensors",
        "experts/exlr8-layer-078.index.safetensors",
        "experts/exlr8-layer-078.k3.safetensors",
    ]);
    let got = exlr8_files(&names, 3, Some(&widths));
    let want = [
        "config.json",
        "dense/exlr8-vocab.safetensors",
        "manifests/exlr8-manifest-mixed.json",
        "experts/exlr8-layer-003.index.safetensors",
        "experts/exlr8-layer-003.k2.safetensors",
        "experts/exlr8-layer-003.k4.safetensors",
        "experts/exlr8-layer-023.index.safetensors",
        "experts/exlr8-layer-023.k3.safetensors",
        "experts/exlr8-layer-023.k4.safetensors",
        "experts/exlr8-layer-078.index.safetensors",
        "experts/exlr8-layer-078.k3.safetensors",
    ];
    assert_eq!(got, want);
    assert!(
        manifest_widths(r#"{"dense": {"0": {"q": 4}}}"#)
            .unwrap()
            .is_empty()
    );
}
