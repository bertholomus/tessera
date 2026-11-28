//! Each rule's plans, and its launches against the CPU interpreter.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_linear as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Rows, Seqs};
use tessera_graph::{DType, Dim, DimName, Graph, NodeId, verify};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Device, Limits, Pins, Plan, Platform, Settings, View, cut, lower, rules};
use tessera_run::open;
use tessera_run::testing::{host, lockstep, runs};
use tessera_testkit::graphs::{self, floats, ids};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Pairs, Role};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

/// The bounds every shared graph here is built with.
fn dims() -> Vec<Dim> {
    vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
    ]
}

/// Entry 0 against the interpreter. Returns the largest difference.
fn gap(g: &Graph, inputs: &[Tensor], device: &Device, settings: &Settings) -> f32 {
    let want = run(g, &[], 0, inputs).unwrap();
    let plan = || lower(g, 0, &SIZES, device).unwrap();
    let mut runners = vec![open(plan(), settings).unwrap()];
    // The host also runs each tree in lockstep.
    if settings.gpu().is_none() {
        runners.push(lockstep(plan(), settings).unwrap());
    }
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let mut worst = 0.0f32;
    for runner in &mut runners {
        let got = runner.run(&feeds).unwrap().remove(0);
        worst = worst.max(match want[0].dtype() {
            // Indices must match exactly.
            DType::U32 if got.indices().unwrap() == want[0].indices().unwrap() => 0.0,
            DType::U32 => f32::INFINITY,
            _ => max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap()),
        });
    }
    worst
}

/// The rules this crate registers.
fn ours() -> Vec<&'static tessera_lower::Rule> {
    let ours = rules().into_iter();
    ours.filter(|r| r.module.starts_with("tessera_kernels::"))
        .collect()
}

/// Every rule's launches under `settings`. Returns the rules used.
fn rules_match(settings: &Settings) -> Vec<&'static str> {
    let d = dims();
    // Rows of 600: more entries than threads for the argmax.
    let graphs = [
        graphs::rows(&d, 32),
        graphs::rows(&d, 600),
        graphs::attention(&d, [2, 2, 16]),
        graphs::rotation(&d),
        graphs::partial_rope(&d, 32),
        graphs::table(&d, 32, 16),
        graphs::neighbours(&d, [6, 9, 16]),
        graphs::picks(&d, 16),
        graphs::indexed(&d, [2, 16, 2]),
        graphs::indexed_rows(&d, [2, 16, 2]),
        graphs::residual_norm(&d, 32),
        graphs::residual_norm(&d, 600),
    ];
    let ours = ours();
    let mut used: Vec<&str> = Vec::new();
    for (g, rule) in graphs.iter().flat_map(|g| ours.iter().map(move |r| (g, r))) {
        let mut noise = Noise(7);
        let device = Device::guess(settings).favoring(rule.name);
        let plan = lower(g, 0, &SIZES, &device).unwrap();
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            used.extend((step.rule == rule.name).then_some(rule.name));
            if seen.contains(&&step.covers[..]) {
                continue;
            }
            seen.push(&step.covers);
            // The nodes one rule covered, as their own graph.
            let sub = cut(g, &step.covers);
            verify(&sub).unwrap();
            let ports = &sub.entries[0].inputs;
            let inputs: Vec<Tensor> = ports
                .iter()
                .map(|p| fill(&sub, &p.ty, &SIZES, &mut noise).unwrap())
                .collect();
            let off = gap(&sub, &inputs, &device, settings);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
    }
    used
}

#[test]
fn rules_match_cpu() {
    for settings in runs() {
        let used = rules_match(&settings);
        // A new rule needs a graph in `testkit::graphs`.
        for rule in ours() {
            let name = rule.name;
            assert!(used.contains(&name), "{name} is never exercised");
        }
    }
}

/// Two rotations by one set of positions, then a map.
fn rotations() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 16),
    ]);
    let port = || floats([Seqs, Rows, Heads, HeadDim]);
    let [x, y, at] = m.entry(EntryKind::Forward, [port(), port(), ids([Seqs, Rows])]);
    let turn = |t: &tessera_trace::Tensor| t.rope(&at, 1e4, Pairs::Halves);
    m.finish([(Role::Logits, &turn(&x).add(&turn(&y)).act(Act::Silu))])
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&tessera_lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

#[test]
fn rotations_share_angles() {
    let plan = lower(&rotations(), 0, &SIZES, &Device::guess(&host())).unwrap();
    assert_eq!((count(&plan, "angles"), count(&plan, "rope")), (1, 2));
}

#[test]
fn favoring_stays_local() {
    // Rope has no competing rule, so favoring it changes nothing.
    let g = rotations();
    let plain = lower(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
    let favored = Device::guess(&host()).favoring("rope");
    assert!(
        plain == lower(&g, 0, &SIZES, &favored).unwrap(),
        "a favored rule skewed its readers' choice"
    );
}

#[test]
fn trips_are_counted() {
    // Walking a row opens a line every sixteen floats.
    let norm = tessera_kernels::RmsNorm {
        rows: 4,
        width: 64,
        eps: 1e-5,
        weighted: true,
    };
    let trip = tessera_lower::Kernel::load(&norm, 64).unwrap().trip;
    assert_eq!((trip.loads, trip.stores), (3.0, 1.0));
    assert!((trip.fresh - 1.0 / 16.0).abs() < 1e-9, "{trip:?}");
    assert_eq!(trip.special, 0.0, "the root lies outside the loops");
}

#[test]
fn limits_bar_kernels() {
    let g = graphs::rows(&dims(), 32);
    let guess = Device::guess(&host()).favoring("lane_norm");
    assert!(names(&lower(&g, 0, &SIZES, &guess).unwrap()).contains(&"lane_norm"));
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let narrow = guess.on(Platform::assumed(wide));
    let picked = names(&lower(&g, 0, &SIZES, &narrow).unwrap());
    assert!(picked.contains(&"rms_norm") && !picked.contains(&"lane_norm"));
    // An uncalibrated GPU picks lane_norm on wide rows.
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    let wide_rows = graphs::rows(&dims(), 600);
    let plan = lower(&wide_rows, 0, &SIZES, &Device::default().on(gpu)).unwrap();
    assert!(names(&plan).contains(&"lane_norm"));
}

#[test]
fn cast_wide_narrow_match() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::TensorType;
    use tessera_lower::form::{FP8_FORM, Load};
    use tessera_trace::{Act, Port};
    let forms: [(&'static dyn Load, f32); 3] =
        [(F32_FORM, 1e-5), (BF16_FORM, 2.0), (FP8_FORM, 4.0)];
    for width in [8u64, 5u64] {
        let m = Model::new(vec![Dim::bounded(Rows, 4), Dim::fixed(Hidden, width)]);
        let [x] = m.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Logits,
                ty: TensorType::new([Rows, Hidden], DType::F32),
            }],
        );
        let g_u32 = m.finish([(Role::Tokens, &x.act(Act::Relu).cast(DType::U32))]);
        let vals: Vec<f32> = (0..3 * width as usize)
            .map(|i| match i % 4 {
                0 => 0.0,
                _ => (i as f32) * 1.5,
            })
            .collect();
        let inp_f = Tensor::f32(vec![3, width as usize], vals);
        let want_u = run(&g_u32, &[], 0, std::slice::from_ref(&inp_f)).unwrap();
        let m2 = Model::new(vec![Dim::bounded(Rows, 4), Dim::fixed(Hidden, width)]);
        let [u] = m2.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Hidden], DType::U32),
            }],
        );
        let g_f32 = m2.finish([(Role::Logits, &u.cast(DType::F32))]);
        let inp_u = Tensor::u32(
            vec![3, width as usize],
            (0..3 * width as u32).map(|i| i * 3).collect(),
        );
        let want_f = run(&g_f32, &[], 0, std::slice::from_ref(&inp_u)).unwrap();
        for (form, tol) in forms {
            let pins = Pins::default().activations(form).outputs(form);
            for settings in runs() {
                let dev = Device::guess(&settings).pinned(pins.clone());
                let p_u = lower(&g_u32, 0, &[(Rows, 3)], &dev).unwrap();
                let mut r_u = open(p_u, &settings).unwrap();
                let got_u = r_u.run(&[inp_f.view()]).unwrap();
                if form.name() == "f32" {
                    assert_eq!(got_u[0].indices().unwrap(), want_u[0].indices().unwrap());
                }
                let p_f = lower(&g_f32, 0, &[(Rows, 3)], &dev).unwrap();
                assert_eq!(p_f.steps.len(), 1, "direct narrow cast_f32");
                let tessera_lower::Source::Made(body) = p_f.steps[0].kernel.source() else {
                    panic!("expected Made source");
                };
                assert_eq!(body.entry.contains("_x4"), width.is_multiple_of(4));
                let mut r_f = open(p_f, &settings).unwrap();
                let got_f = r_f.run(&[inp_u.view()]).unwrap();
                let diff = max_abs_diff(got_f[0].floats().unwrap(), want_f[0].floats().unwrap());
                assert!(diff <= tol, "{} w={width}: {diff:e}", body.entry);
            }
        }
    }
}

#[test]
fn tables_keep_forms() {
    let g = graphs::table(&dims(), 32, 16);
    let guess = Device::guess(&host());
    let narrow = guess.clone().pinned(Pins::all(BF16_FORM));
    assert!(names(&lower(&g, 0, &SIZES, &narrow).unwrap()).contains(&"gather_bf16"));
    // A table pinned f32 stays f32.
    let pins = Pins::all(BF16_FORM).pin("embed", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &guess.pinned(pins)).unwrap();
    assert_eq!(names(&plan), ["gather"]);
}

#[test]
fn stage_rules() {
    let rules = rules();
    for s in tessera_graph::STAGES {
        if let tessera_graph::Lower::Kernel(name) = s.facts().lower {
            assert!(rules.iter().any(|r| r.name == name), "no rule {name}");
        }
    }
}

#[test]
fn wide_strided_narrow_forms() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::TensorType;
    use tessera_lower::Pins;
    use tessera_lower::form::{BF16_FORM, F32_FORM};
    use tessera_trace::{Act, Port};
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 16)]);
    let port = Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x] = m.entry(EntryKind::Forward, [port]);
    let sliced = x.act(Act::Silu).slice(1, 4, 8);
    let g = m.finish([(Role::Logits, &sliced)]);
    let dev = Device::default().pinned(
        Pins::all(BF16_FORM)
            .activations(BF16_FORM)
            .outputs(F32_FORM),
    );
    let plan = lower(&g, 0, &[(Rows, 4)], &dev).unwrap();
    let strided = plan.steps.iter().find(|s| s.rule == "view").unwrap();
    let tessera_lower::Source::Made(body) = strided.kernel.source() else {
        panic!("expected Made source")
    };
    assert_eq!(body.entry, "plan_strided_x4_bf16_f32");
    // Same-form narrow bf16->bf16 also uses vectorized x4.
    let dev_same = Device::default().pinned(
        Pins::all(BF16_FORM)
            .activations(BF16_FORM)
            .outputs(BF16_FORM),
    );
    let plan_same = lower(&g, 0, &[(Rows, 4)], &dev_same).unwrap();
    let strided_same = plan_same.steps.iter().find(|s| s.rule == "view").unwrap();
    let tessera_lower::Source::Made(body_same) = strided_same.kernel.source() else {
        panic!("expected Made source")
    };
    assert_eq!(body_same.entry, "plan_strided_x4_bf16_bf16");
    let mut noise = Noise(7);
    let inp = fill(&g, &g.entries[0].inputs[0].ty, &[(Rows, 4)], &mut noise).unwrap();
    let want = run(&g, &[], 0, std::slice::from_ref(&inp)).unwrap();
    for settings in runs() {
        let mut runner = open(plan.clone(), &settings).unwrap();
        let got = runner.run(&[inp.view()]).unwrap();
        let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
        assert!(diff < 2e-2, "diff={diff:e}");
    }
}
