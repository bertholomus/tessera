//! Tencent Hy4-flash.

mod engram_test;

use tessera_models::hy4::{Config, PlainLatent, program, stepper};
use tessera_parity::{Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Layer, Mlp, Model, Moe};
use tessera_trace::Graph;

fn small() -> Graph {
    program::<PlainLatent>(&Config::hy4_small())
}

pub fn release() -> Graph {
    program::<PlainLatent>(&Config::hy4_flash(1, 1024))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<PlainLatent>(&Config::hy4_small()),
    release: || stepper::<PlainLatent>(&Config::hy4_flash(1, 1024)),
    drift: tessera_parity::MLA_DRIFT,
};

const SPEC: Spec = Spec {
    name: "hy4",
    release,
    parameters: 155_546_910_285,
    regions: &[
        (Attention, 42),
        (Mlp, 1),
        (Moe, 41),
        (Layer, 42),
        (Model, 1),
    ],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
