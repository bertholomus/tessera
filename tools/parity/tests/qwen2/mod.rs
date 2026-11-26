//! Qwen2-14B: a decoder with Q, K, and V projection biases.

use tessera_models::decoder::{Config, Swiglu, program_with, stepper_with};
use tessera_parity::{DRIFT, Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Layer, Mlp, Model};
use tessera_trace::{Graph, Vector};

fn small() -> Graph {
    program_with::<(), Swiglu, Vector>(&Config::qwen2_small())
}

fn release() -> Graph {
    program_with::<(), Swiglu, Vector>(&Config::qwen2(8, 32_768))
}

const STEPPER: Stepper = Stepper {
    small: || stepper_with::<(), Swiglu, Vector>(&Config::qwen2_small()),
    release: || stepper_with::<(), Swiglu, Vector>(&Config::qwen2(1, 1024)),
    drift: DRIFT,
};

const SPEC: Spec = Spec {
    name: "qwen2",
    release,
    parameters: 14_770_033_664,
    regions: &[(Attention, 48), (Mlp, 48), (Layer, 48), (Model, 1)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[
        (include_str!("manifests/hf.json"), release),
        (include_str!("manifests/gguf.json"), release),
    ],
    golden: None,
};

suite!(SPEC);
