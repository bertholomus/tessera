use crate::lockstep::{self, BLOCK, Dots, dots_form};
use tessera_codegen::form::{BF16_FORM, F32_FORM, Load};

#[test]
fn odd_length_dots() {
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..19).map(|i| (i % 3) as f32).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(lockstep::dot(&a, &b), want);
    let halves = BF16_FORM.pack(&b);
    assert_eq!(lockstep::dot_bf16(&a, &halves), want);
    let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(
        lockstep::dot_form(BF16_FORM, &bytes, 0, &halves, 0, 19),
        want
    );
}

#[test]
fn narrow_dots_match() {
    use tessera_codegen::form::{FP8_FORM, named};
    let f16 = named("f16").unwrap();
    let k = 131usize;
    let x: Vec<f32> = (0..(BLOCK + 3) * k)
        .map(|i| ((i * 19) % 23) as f32 * 0.25 - 2.75)
        .collect();
    let w: Vec<f32> = (0..BLOCK * k)
        .map(|i| ((i * 11) % 17) as f32 * 0.5 - 4.0)
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wf16 = f16.pack(&w);
    let wf16_ref = f16.unpack_all(&wf16);
    let wfp8 = FP8_FORM.pack(&w);
    let wfp8_ref = FP8_FORM.unpack_all(&wfp8);

    // Odd-length dot_form and direct helpers for f16 and fp8.
    let want_f16: f32 = x[..k].iter().zip(&wf16_ref[..k]).map(|(a, b)| a * b).sum();
    let want_fp8: f32 = x[..k].iter().zip(&wfp8_ref[..k]).map(|(a, b)| a * b).sum();
    assert!((lockstep::dot_form(f16, &xb, 0, &wf16, 0, k) - want_f16).abs() < 1e-3);
    assert!((lockstep::dot_form(FP8_FORM, &xb, 0, &wfp8, 0, k) - want_fp8).abs() < 1e-3);

    // Exhaustive binary16 widening check via tail and NEON dot_f16.
    let one = [1.0f32];
    let mut mask = [0.0f32; 8];
    for bits in 0u16..=u16::MAX {
        let raw = bits.to_le_bytes();
        let got = lockstep::dot_f16(&one, &raw);
        let want = f16.unpack(&raw, 0);
        if want.is_nan() {
            assert!(got.is_nan(), "f16 {bits:#06x}");
        } else {
            assert_eq!(got, want, "f16 {bits:#06x}");
        }
    }
    for range in [0u16..0x7C00, 0x8000u16..0xFC00] {
        for base in range.step_by(8) {
            let mut raw8 = [0u8; 16];
            for lane in 0..8 {
                let b = base + lane as u16;
                raw8[lane * 2..lane * 2 + 2].copy_from_slice(&b.to_le_bytes());
            }
            for lane in 0..8 {
                mask.fill(0.0);
                mask[lane] = 1.0;
                let got = lockstep::dot_f16(&mask, &raw8);
                let want = f16.unpack(&raw8[lane * 2..lane * 2 + 2], 0);
                assert_eq!(got, want, "neon f16 lane {lane} base {base:#06x}");
            }
        }
    }

    // Tile dots over f16, fp8, partial cols, and rows > BLOCK.
    let cases: [(&dyn Load, &[u8], &[f32]); 2] =
        [(f16, &wf16, &wf16_ref), (FP8_FORM, &wfp8, &wfp8_ref)];
    for (form, wb, wref) in cases {
        for (rows, cols) in [(3, 13), (BLOCK + 3, 21)] {
            let mut out = vec![0u8; rows * BLOCK * 4];
            let d = Dots {
                at: 0,
                stride: BLOCK,
                x_at: 0,
                w_at: 0,
                k,
                cols,
            };
            dots_form(form, &mut out, &xb, wb, &d, rows);
            for r in 0..rows {
                for c in 0..BLOCK {
                    let want: f32 = if c < cols {
                        (0..k).map(|i| x[r * k + i] * wref[c * k + i]).sum()
                    } else {
                        0.0
                    };
                    let at = (r * BLOCK + c) * 4;
                    let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                    assert!(
                        (got - want).abs() < 1e-2,
                        "{} r={r} c={c}: {got} vs {want}",
                        form.name()
                    );
                }
            }
        }
    }
}

/// Blocks of odd rows and depths against plain sums.
#[test]
fn dots_match_sums() {
    let (k, n) = (300, BLOCK);
    // Halves of small integers: exact in bf16 too.
    let value = |i: usize| (i % 13) as f32 * 0.5 - 3.0;
    let x: Vec<f32> = (0..BLOCK * k).map(value).collect();
    let w: Vec<f32> = (0..n * k).map(|i| value(i * 7)).collect();
    let wide: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let narrow: Vec<u8> = w
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let forms: [(&dyn Load, &[u8]); 2] = [(F32_FORM, &wide), (BF16_FORM, &narrow)];
    for (form, w_bytes) in forms {
        for (rows, cols) in [(1, n), (3, 13), (13, n), (BLOCK, 13)] {
            let mut out = vec![0u8; BLOCK * n * 4];
            let d = Dots {
                at: 0,
                stride: n,
                x_at: 0,
                w_at: 0,
                k,
                cols,
            };
            dots_form(form, &mut out, &xb, w_bytes, &d, rows);
            for (r, c) in (0..rows).flat_map(|r| (0..n).map(move |c| (r, c))) {
                // Columns past `cols` stay untouched.
                let want: f32 = match c < cols {
                    true => (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum(),
                    false => 0.0,
                };
                let at = (r * n + c) * 4;
                let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{} rows {rows}: {got} {want}",
                    form.name()
                );
            }
        }
    }
}
