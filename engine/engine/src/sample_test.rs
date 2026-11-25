use crate::Allowed;
use crate::sample::{Sampler, Sampling};

#[test]
fn greedy_matches_argmax() {
    let logits = [1.0, 3.0, 3.0, 2.0];
    let mut s = Sampler::new(Sampling::default());
    assert_eq!(s.pick(&logits, None), 1);
    assert_eq!(s.pick(&logits, Some(&Allowed::Few(vec![0, 3]))), 3);
    assert_eq!(s.pick(&logits, Some(&Allowed::One(0))), 0);
}

#[test]
fn seed_reproduces_stream() {
    let logits = [1.0, 1.5, 1.4, 0.8, 1.2];
    let cfg = Sampling {
        temperature: 0.8,
        seed: Some(42),
        ..Sampling::default()
    };
    let draw = |c| {
        let mut s = Sampler::new(c);
        (0..16).map(|_| s.pick(&logits, None)).collect::<Vec<_>>()
    };
    assert_eq!(draw(cfg), draw(cfg));
    let other = Sampling {
        seed: Some(7),
        ..cfg
    };
    assert_ne!(draw(cfg), draw(other));
}

#[test]
fn filters_restrict_support() {
    let logits = [10.0, 9.0, 1.0, 0.0];
    let cfg = Sampling {
        temperature: 1.0,
        top_k: 2,
        seed: Some(1),
        ..Sampling::default()
    };
    let mut s = Sampler::new(cfg);
    for _ in 0..64 {
        assert!(s.pick(&logits, None) <= 1);
    }
    let min_p_cfg = Sampling {
        temperature: 1.0,
        min_p: 0.5,
        seed: Some(2),
        ..Sampling::default()
    };
    let mut s = Sampler::new(min_p_cfg);
    for _ in 0..64 {
        assert_eq!(s.pick(&[10.0, 8.0, 0.0], None), 0);
    }
    let top_p_cfg = Sampling {
        temperature: 1.0,
        top_p: 0.6,
        seed: Some(3),
        ..Sampling::default()
    };
    let mut s = Sampler::new(top_p_cfg);
    for _ in 0..64 {
        assert_eq!(s.pick(&[10.0, 8.0, 0.0], None), 0);
    }
}
