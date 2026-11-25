//! Integration test for `--kv-cache-dir` and persistent SSD KV caching.

use std::time::Duration;
use tessera_engine::comm::{Sockets, Transport};
use tessera_engine::decoder::{Swiglu, stepper};
use tessera_engine::{
    Batch, Engine, Event, Fair, Norm, Pick, Place, Recipe, Request, Settings, Weight,
};
use tessera_models::decoder;
use tessera_serve::start::Options;
use tessera_testkit::{Noise, tcp_pair, tcp_timeouts, weights};

fn open_disk_eng(links: Option<[Sockets; 2]>, recipe: &Recipe) -> Engine {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let g = stepper::<Norm, Swiglu>(&decoder::Config::toy(2, 128));
    let stored: Vec<Weight> = weights(&g, &mut Noise(3))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let mut eng = match links {
        Some([sums, ctrl]) => {
            let place = Place::mesh(2, Vec::new(), false);
            let c: Box<dyn Transport> = Box::new(ctrl);
            Engine::open_ranked(g, 128, (recipe, &settings), place, sums, c).unwrap()
        }
        None => Engine::open(g, 128, recipe, &settings).unwrap(),
    };
    eng.load(&stored).unwrap();
    eng
}

#[test]
fn disk_kv_restores() {
    let mut opts = Options::default();
    assert!(opts.flag("--kv-cache-dir", "/tmp/kv".into()).is_some());
    assert!(opts.flag("--kv-cache-disk-gb", "8".into()).is_some());
    assert!(opts.flag("--kv-cache-disk-gb", "0".into()).is_none());

    let pid = std::process::id();
    let (d, d0, d1) = (
        tessera_testkit::scratch(&format!("{pid}-dk")),
        tessera_testkit::scratch(&format!("{pid}-dk0")),
        tessera_testkit::scratch(&format!("{pid}-dk1")),
    );
    let recipe = Recipe {
        kept: Engine::paged_layout(),
        ..Recipe::default()
    };
    let p1: Vec<u32> = (0..70).map(|i| (i * 5 + 1) % 32).collect();
    let p2: Vec<u32> = p1.iter().copied().chain([4, 9, 13]).collect();
    let ids = |e: &[Event]| -> Vec<u32> {
        e.iter()
            .filter_map(|x| match x {
                Event::Token(id) => Some(*id),
                _ => None,
            })
            .collect()
    };
    let mut want = Vec::new();
    let req = Request {
        tokens: p2.clone(),
        max_tokens: 4,
        ..Request::default()
    };
    open_disk_eng(None, &recipe)
        .generate(req, &mut |e| want.push(e))
        .unwrap();
    let run_one = |eng: &mut Engine, tokens: Vec<u32>| -> Vec<Event> {
        let mut batch = Batch::new(eng, Box::new(Fair));
        batch.reuse();
        let (tx, rx) = std::sync::mpsc::channel();
        let req = Request {
            tokens,
            max_tokens: 4,
            ..Request::default()
        };
        batch.submit(req, Box::new(move |e| tx.send(e).is_ok()));
        while batch.step().unwrap() {}
        rx.try_iter().collect()
    };
    {
        let mut eng = open_disk_eng(None, &recipe);
        eng.set_kv_store(&d, 1 << 20).unwrap();
        let _ = run_one(&mut eng, p1.clone());
        assert_eq!(eng.kv_store_stats().unwrap().saves, 1);
    }
    let mut eng = open_disk_eng(None, &recipe);
    eng.set_kv_store(&d, 1 << 20).unwrap();
    let evs = run_one(&mut eng, p2.clone());
    assert_eq!(evs.first(), Some(&Event::Kept(64)));
    assert_eq!(ids(&evs), ids(&want));
    let st = eng.kv_store_stats().unwrap();
    assert_eq!((st.hits, st.saves), (1, 0));

    let make_links = || -> [Sockets; 2] {
        let (a, b) = tcp_pair().unwrap();
        for s in [&a, &b] {
            tcp_timeouts(s, Duration::from_secs(20)).unwrap();
        }
        [
            Sockets::over(0, vec![None, Some(a)]).unwrap(),
            Sockets::over(1, vec![Some(b), None]).unwrap(),
        ]
    };
    let run_tp = |tokens: Vec<u32>| -> Vec<Event> {
        let ([sums, others], [control, told]) = (make_links(), make_links());
        let (r1, fd) = (recipe.clone(), d1.clone());
        let f = std::thread::spawn(move || {
            let mut e = open_disk_eng(Some([others, told]), &r1);
            e.set_kv_store(&fd, 1 << 20).unwrap();
            e.follow().unwrap();
        });
        let mut lead = open_disk_eng(Some([sums, control]), &recipe);
        lead.set_kv_store(&d0, 1 << 20).unwrap();
        let evs = run_one(&mut lead, tokens);
        lead.dismiss().unwrap();
        f.join().unwrap();
        evs
    };
    let _ = run_tp(p1);
    let evs = run_tp(p2);
    assert_eq!(evs.first(), Some(&Event::Kept(64)));
    assert_eq!(ids(&evs), ids(&want));
    for dir in [d, d0, d1] {
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
