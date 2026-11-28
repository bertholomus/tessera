//! Parity and dispatch tests for native Metal MLA attention.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Picks, Rows, Seqs};
use tessera_graph::{DType, Dim, DimExpr, Graph, Size, TensorType};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Backend, Device, Pins, Settings, View, lower};
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Mask, Model, Port, Role};

fn metal_settings() -> Option<Settings> {
    runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal))
}

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

fn make_rows(runs: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let slots = runs.iter().flat_map(|(s, _, n)| (0..*n).map(move |_| *s));
    let pos = runs
        .iter()
        .flat_map(|(_, p, n)| (0..*n).map(move |i| p + i));
    (slots.collect(), pos.collect())
}

fn mla_graph(
    mask: Mask,
    heads: u64,
    width: u64,
    slice_to: Option<u64>,
    picks: Option<u64>,
    max_ctx: u64,
) -> Graph {
    let mut dims = vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 64),
        Dim::bounded(Context, max_ctx),
        Dim::fixed(Heads, heads),
        Dim::fixed(HeadDim, width),
    ];
    if let Some(p) = picks {
        dims.push(Dim {
            name: Picks,
            size: Size::Fixed(p),
        });
    }
    let m = Model::new(dims);
    let q_ty = TensorType::new([Rows, Heads, HeadDim], DType::F32);
    let kv_ty = TensorType::new(
        vec![Rows.into(), DimExpr::Const(1), HeadDim.into()],
        DType::F32,
    );
    let u32_row = TensorType::new([Rows], DType::U32);
    let kept_ty = TensorType::new(
        vec![
            Seqs.into(),
            Context.into(),
            DimExpr::Const(1),
            HeadDim.into(),
        ],
        DType::F32,
    );
    let kept = m.kept(kept_ty);
    let (q, kv_in, slots, positions, ids) = if picks.is_some() {
        let [q, kv, slots, pos, ids] = m.entry(
            EntryKind::Step,
            [
                Port {
                    role: Role::Tokens,
                    ty: q_ty,
                },
                Port {
                    role: Role::Tokens,
                    ty: kv_ty,
                },
                Port {
                    role: Role::Slots,
                    ty: u32_row.clone(),
                },
                Port {
                    role: Role::Positions,
                    ty: u32_row,
                },
                Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, Picks], DType::U32),
                },
            ],
        );
        (q, kv, slots, pos, Some(ids))
    } else {
        let [q, kv, slots, pos] = m.entry(
            EntryKind::Step,
            [
                Port {
                    role: Role::Tokens,
                    ty: q_ty,
                },
                Port {
                    role: Role::Tokens,
                    ty: kv_ty,
                },
                Port {
                    role: Role::Slots,
                    ty: u32_row.clone(),
                },
                Port {
                    role: Role::Positions,
                    ty: u32_row,
                },
            ],
        );
        (q, kv, slots, pos, None)
    };
    let by = [&slots, &positions];
    let k = kv_in.append(&kept, by);
    let attn = m.attend_rows(mask, 0.08, [&q, &k, &k], by, ids.as_ref());
    let out = match slice_to {
        Some(lat) => attn.slice(2, 0u64, lat),
        None => attn,
    };
    m.finish([(Role::Logits, &out)])
}

fn run_mla(
    runner: &mut Option<Box<dyn Run>>,
    plan: tessera_lower::Plan,
    settings: &Settings,
    inputs: &[Tensor],
) -> Tensor {
    if let Some(existing) = runner.as_mut() {
        existing.replan(&plan).unwrap();
    } else {
        let paged = plan.table;
        let mut fresh = open(plan, settings).unwrap();
        if let Some(slot) = paged {
            let mut raw = Vec::with_capacity(slot.len * 4);
            for page in (0..slot.len as u32).rev() {
                raw.extend_from_slice(&page.to_le_bytes());
            }
            fresh.held_mut().upload(slot, &raw).unwrap();
        }
        *runner = Some(fresh);
    }
    let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
    runner.as_mut().unwrap().run(&views).unwrap().remove(0)
}

#[test]
fn mla_mma_matches_cpu() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(101);
    let forms: [(&'static dyn Stored, &'static dyn Stored, f32); 4] = [
        (F32_FORM, F32_FORM, 5e-4),
        (F32_FORM, BF16_FORM, 2e-2),
        (BF16_FORM, BF16_FORM, 3e-2),
        (F32_FORM, FP8_FORM, 0.2),
    ];
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    for (slice_to, heads) in [(Some(512u64), 32u64), (None, 20u64)] {
        let g = mla_graph(Mask::Causal, heads, 576, slice_to, None, 512);
        for &(act_form, kv_form, tol) in &forms {
            let mut pins = Pins::default();
            if act_form.name() != "f32" {
                pins = pins.activations(act_form);
            }
            if kv_form.name() != "f32" {
                pins = pins.kept(kv_form);
            }
            for &layout in &layouts {
                let (mut store, mut runner) = (Store::default(), None);
                let batches: [&[(u32, u32, u32)]; 2] = [
                    &[(0, 0, 19), (1, 0, 7)],
                    &[(0, 19, 1), (1, 7, 1), (2, 0, 5)],
                ];
                for spec in batches {
                    let (slots, positions) = make_rows(spec);
                    let n = slots.len();
                    let furthest = *positions.iter().max().unwrap() as u64 + 1;
                    let inputs = [
                        draw(vec![n, heads as usize, 576], &mut draws),
                        draw(vec![n, 1, 576], &mut draws),
                        Tensor::u32(vec![n], slots),
                        Tensor::u32(vec![n], positions),
                    ];
                    let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                    let sizes = [(Seqs, 4), (Rows, n as u64), (Context, furthest)];
                    let mut device = Device::guess(&settings)
                        .favoring("mla_metal")
                        .pinned(pins.clone());
                    device.kept = layout;
                    let plan = lower(&g, 0, &sizes, &device).unwrap();
                    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                    assert!(names.contains(&"mla_metal"), "{names:?}");
                    let got = run_mla(&mut runner, plan, &settings, &inputs);
                    let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                    assert!(
                        off < tol,
                        "mma slice={slice_to:?} heads={heads} {} {}: off by {off:e}",
                        kv_form.name(),
                        layout.name()
                    );
                }
            }
        }
    }
}

#[test]
fn mla_splitk_picked_match() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(202);
    // Split-K over 320 keys triggers splits > 1 and mla_metal_reduce.
    let g_dense = mla_graph(Mask::Causal, 16, 576, Some(512), None, 512);
    let (mut store, mut runner) = (Store::default(), None);
    let mut all_ctx = 0u32;
    for chunk in 0..5u32 {
        let count = 64u32;
        let (slots, positions) = make_rows(&[(0, all_ctx, count)]);
        all_ctx += count;
        let inputs = [
            draw(vec![count as usize, 16, 576], &mut draws),
            draw(vec![count as usize, 1, 576], &mut draws),
            Tensor::u32(vec![count as usize], slots),
            Tensor::u32(vec![count as usize], positions),
        ];
        let want = run_kept(&g_dense, &[], &mut store, 0, &inputs).unwrap();
        let sizes = [(Seqs, 4), (Rows, count as u64), (Context, all_ctx as u64)];
        let pins = Pins::default().kept(BF16_FORM);
        let mut device = Device::guess(&settings).pinned(pins);
        device.kept = &kept::PAGED16;
        let plan = lower(&g_dense, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        if all_ctx >= 256 {
            assert!(names.contains(&"mla_metal"), "unfavored: {names:?}");
        }
        let got = run_mla(&mut runner, plan, &settings, &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(
            off < 2e-2,
            "splitk chunk={chunk} ctx={all_ctx}: off {off:e}"
        );
    }
    // Single-row decode at ctx=320 triggers splits > 1 and mla_metal_reduce.
    let (slots, positions) = make_rows(&[(0, all_ctx, 1)]);
    all_ctx += 1;
    let inputs = [
        draw(vec![1, 16, 576], &mut draws),
        draw(vec![1, 1, 576], &mut draws),
        Tensor::u32(vec![1], slots),
        Tensor::u32(vec![1], positions),
    ];
    let want = run_kept(&g_dense, &[], &mut store, 0, &inputs).unwrap();
    let sizes = [(Seqs, 4), (Rows, 1), (Context, all_ctx as u64)];
    let mut device = Device::guess(&settings).pinned(Pins::default().kept(BF16_FORM));
    device.kept = &kept::PAGED16;
    let plan = lower(&g_dense, 0, &sizes, &device).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(names.contains(&"mla_metal"), "{names:?}");
    assert!(names.contains(&"mla_metal_reduce"), "{names:?}");
    let got = run_mla(&mut runner, plan, &settings, &inputs);
    let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    assert!(off < 2e-2, "splitk decode: off {off:e}");

    // Sparse picked keys with trailing sentinels and general width=48.
    for (width, slice_to) in [(576u64, Some(512u64)), (48u64, Some(32u64))] {
        let picks = 24u64;
        let g = mla_graph(Mask::Causal, 16, width, slice_to, Some(picks), 64);
        let (mut store, mut runner) = (Store::default(), None);
        let (slots, positions) = make_rows(&[(0, 0, 35)]);
        let mut ids_data = Vec::with_capacity(35 * picks as usize);
        for r in 0..35u32 {
            let valid = (r + 1).min(picks as u32);
            for p in 0..picks as u32 {
                if p < valid {
                    ids_data.push(r - p);
                } else {
                    ids_data.push(64);
                }
            }
        }
        let inputs = [
            draw(vec![35, 16, width as usize], &mut draws),
            draw(vec![35, 1, width as usize], &mut draws),
            Tensor::u32(vec![35], slots),
            Tensor::u32(vec![35], positions),
            Tensor::u32(vec![35, picks as usize], ids_data),
        ];
        let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
        let sizes = [(Seqs, 4), (Rows, 35), (Context, 35)];
        let device = Device::guess(&settings).favoring("mla_metal");
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(names.contains(&"mla_metal"), "picked w={width}: {names:?}");
        let got = run_mla(&mut runner, plan, &settings, &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 5e-4, "picked w={width}: off {off:e}");
    }
}

#[test]
fn mla_ring_window_matches() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(303);
    let g = mla_graph(Mask::Back(7), 16, 576, Some(512), None, 32);
    let (mut store, mut runner) = (Store::default(), None);
    for step in 0..4u32 {
        let pos = step * 5;
        let (slots, positions) = make_rows(&[(0, pos, 5)]);
        let inputs = [
            draw(vec![5, 16, 576], &mut draws),
            draw(vec![5, 1, 576], &mut draws),
            Tensor::u32(vec![5], slots),
            Tensor::u32(vec![5], positions),
        ];
        let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
        let sizes = [(Seqs, 4), (Rows, 5), (Context, 32)];
        let device = Device::guess(&settings).favoring("mla_metal");
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(names.contains(&"mla_metal"), "{names:?}");
        let got = run_mla(&mut runner, plan, &settings, &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 5e-4, "ring step={step}: off {off:e}");
    }
}
