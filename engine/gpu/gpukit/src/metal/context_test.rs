use super::Context;

const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

/// Pinned Metal memory is directly writable by host and GPU.
#[test]
fn pin_round_trip() {
    use crate::device::{Device, DeviceBuffer};
    use crate::launch::{KernelArgs, Launch};
    use tessera_mem::Backing;

    let Ok(ctx) = Context::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let mut buf = DeviceBuffer::pinned(&ctx, 16).expect("pin");
    let mapped = buf.mapped_mut().expect("host view");
    for (i, chunk) in mapped.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *chunk = (i as f32).to_le_bytes();
    }
    let src = "\
#include <metal_stdlib>
using namespace metal;
kernel void inc(device float* v [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] += 1.0f; }
";
    let kernel = ctx.compile_kernel(src, "inc").expect("compile");
    let mut args = KernelArgs::new();
    args.device_ptr(buf.device_ptr());
    ctx.launch(&kernel, (1, 1, 1), (4, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.finish().expect("finish");
    let mut bytes = [0u8; 16];
    buf.read_bytes(&mut bytes).expect("read");
    let out: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    let _ = ctx.unified();
}

/// Concurrent compiles share one set of pipelines.
#[test]
fn compile_race_free() {
    if Context::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                let ctx = Context::new().expect("device");
                for _ in 0..4 {
                    for e in entries {
                        ctx.compile_kernel(SRC, e).expect("pipeline compile");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked: a pipeline race?");
    }
}
