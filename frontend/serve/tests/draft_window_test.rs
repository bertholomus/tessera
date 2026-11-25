//! Integration test for sliding-window drafter tap capping and `--draft`.

use tessera_engine::decoder::{Swiglu, stepper};
use tessera_engine::{
    Drafter, Engine, Event, Norm, Pick, Recipe, Request, Settings, Speculate, Weight,
};
use tessera_models::decoder;
use tessera_models::dflash::{self, TargetSpec};
use tessera_serve::start::Options;
use tessera_testkit::{Noise, weights};

fn open_engine(g: tessera_engine::Graph, ctx: usize, w: &[Weight], s: &Settings) -> Engine {
    let mut eng = Engine::open(g, ctx, &Recipe::default(), s).unwrap();
    eng.load(w).unwrap();
    eng
}

#[test]
fn window_caps_taps() {
    let mut opts = Options::default();
    assert!(opts.flag("--draft", "/tmp/drafter".into()).is_some());

    let mut t_cfg = decoder::Config::toy(1, 48);
    t_cfg.tied = false;
    t_cfg.stack.taps = vec![1, 2];
    let t_graph = stepper::<Norm, Swiglu>(&t_cfg);
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: false,
    };
    let mut d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 20);
    d_cfg.sliding_window = Some(6);
    let d_graph = dflash::stepper::<Norm>(&d_cfg);
    let tw: Vec<Weight> = weights(&t_graph, &mut Noise(61))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let dw: Vec<Weight> = weights(&d_graph, &mut Noise(67))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let prompt: Vec<u32> = (0..18).map(|i| (i * 3 + 1) % 16).collect();
    let ids = |ev: &[Event]| -> Vec<u32> {
        ev.iter()
            .filter_map(|e| match e {
                Event::Token(id) => Some(*id),
                _ => None,
            })
            .collect()
    };
    let mut base = open_engine(t_graph.clone(), 48, &tw, &settings);
    let mut want = Vec::new();
    let req = Request {
        tokens: prompt.clone(),
        max_tokens: 6,
        ..Request::default()
    };
    base.generate(req, &mut |e| want.push(e)).unwrap();
    let mut target = open_engine(t_graph, 48, &tw, &settings);
    let mut drafter = Drafter::open(d_graph, 20, 4, 0, &Recipe::default(), &settings).unwrap();
    drafter.load(&dw).unwrap();
    assert_eq!(drafter.sliding_window(), Some(6));
    for chunk in [None, Some(5)] {
        let mut s = Speculate::new(&mut target, &mut drafter).unwrap();
        if let Some(rows) = chunk {
            s = s.with_prefill_chunk(rows);
        }
        let mut got = Vec::new();
        let req = Request {
            tokens: prompt.clone(),
            max_tokens: 6,
            ..Request::default()
        };
        let st = s
            .generate(req, &mut |e| {
                got.push(e);
                true
            })
            .unwrap();
        assert!(st.dflash_steps > 0);
        assert_eq!(ids(&got), ids(&want));
    }
}
