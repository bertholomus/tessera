use tessera_models::hy4::{Config, Engram, PlainLatent, program, stepper};

#[test]
fn engram_traces() {
    let mut cfg = Config::hy4_small();
    cfg.engram = Some(Engram {
        layers: vec![1],
        heads: 2,
        head_dim: 4,
    });
    let fwd = program::<PlainLatent>(&cfg);
    let stp = stepper::<PlainLatent>(&cfg);
    assert_eq!(fwd.weights.len(), stp.weights.len());
}
