//! Seeded sampling through the batch scheduler.

use tessera_engine::{Batch, Engine, Event, Fair, Recipe, Request, Sampling};
use tessera_lower::{Settings, Weight};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

fn sampled(engine: &mut Engine, seed: u64) -> Vec<u32> {
    let mut batch = Batch::new(engine, Box::new(Fair));
    let (tx, rx) = std::sync::mpsc::channel();
    let req = Request {
        tokens: vec![3, 9, 4],
        max_tokens: 6,
        sampling: Sampling {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 4,
            min_p: 0.05,
            seed: Some(seed),
        },
        ..Request::default()
    };
    batch.submit(req, Box::new(move |e| tx.send(e).is_ok()));
    while batch.step().unwrap() {}
    let mut ids = Vec::new();
    for e in rx.try_iter() {
        if let Event::Token(id) = e {
            ids.push(id);
        }
    }
    ids
}

#[test]
fn sampling_respects_seed() {
    let g = stepper::<Norm, Swiglu>(&Config::toy(2, 16));
    let stored: Vec<Weight> = weights(&g, &mut Noise(41))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let mut engine = Engine::open(g, 16, &Recipe::default(), &Settings::default()).unwrap();
    engine.load(&stored).unwrap();
    let a = sampled(&mut engine, 42);
    let b = sampled(&mut engine, 42);
    assert_eq!(a.len(), 6);
    assert_eq!(a, b);
}
