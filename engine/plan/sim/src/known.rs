//! Tables for the machines these models were fitted on.

use crate::nvidia::gb10;
use crate::{Chip, Fit, Pool};

/// The name an Apple M3 Pro reports through Metal.
const M3_PRO: &str = "metal Apple M3 Pro";

/// The name an Apple M4 reports through Metal.
const M4: &str = "metal Apple M4";

/// An Apple M3 Pro, through Metal. Its step varies twofold.
pub fn m3_pro() -> Pool {
    m3_pro_fit().pool(M3_PRO)
}

/// An Apple M4, through Metal.
pub(crate) fn m4() -> Pool {
    m4_fit().pool(M4)
}

/// Registers and group memory share one store per core.
fn m3_pro_fit() -> Fit {
    Fit {
        floor: 3e-6,
        spawn: 9e-11,
        step: 3.2e-8,
        curve: 1.4e-8,
        rate: 3.8e11,
        wait: 3.1e-7,
        near: 6.5e-8,
        cache: 4e6,
        cached: 2.54e11,
        read: 1.28e11,
        scan: 8.5e10,
        write: 1.15e11,
        tile: 3.9e12,
        cores: 14.0,
        store: 320e3,
        resident: 1024.0,
        held: 128.0,
        shared: 3e-8,
        banked: 1.8e11,
        mma: 5e-10,
        spill: 3.2e-7,
    }
}

/// Apple M4 constants from Metal micro-probes.
fn m4_fit() -> Fit {
    let mut f = m3_pro_fit();
    (f.floor, f.spawn, f.step, f.curve) = (2.8e-6, 3.1e-11, 2.5e-8, 1.24e-8);
    (f.rate, f.wait, f.near, f.cached) = (3.65e11, 2.56e-7, 5.7e-8, 7.8e11);
    (f.read, f.scan, f.write, f.tile, f.cores) = (1.02e11, 7.5e10, 9.0e10, 3.08e12, 10.0);
    (f.shared, f.mma) = (2.3e-8, 4.0e-10);
    f
}

/// The table's constants for a pool device, if any.
pub(crate) fn fit(device: &str) -> Option<Fit> {
    match device {
        M3_PRO => Some(m3_pro_fit()),
        M4 => Some(m4_fit()),
        _ => None,
    }
}

/// The table for a runner's device name. ROCm names none.
pub fn known(device: &str) -> Option<Box<dyn Chip>> {
    let chips: [Box<dyn Chip>; 3] = [Box::new(gb10()), Box::new(m3_pro()), Box::new(m4())];
    chips.into_iter().find(|c| c.name() == device)
}

/// A generic chip model from driver-reported specs.
pub fn generic(
    backend: &str,
    name: &str,
    units: usize,
    bandwidth: f64,
    group_bytes: usize,
) -> Option<Box<dyn Chip>> {
    if units == 0 || bandwidth <= 0.0 {
        return None;
    }
    match backend {
        "cuda" => Some(Box::new(crate::nvidia::scaled(
            name,
            units,
            bandwidth,
            group_bytes,
        ))),
        "metal" | "rocm" => {
            let mut fit = if backend == "metal" {
                m4_fit()
            } else {
                m3_pro_fit()
            };
            let scale = units as f64 / fit.cores;
            fit.cores = units as f64;
            fit.read = bandwidth * 0.85;
            fit.scan = bandwidth * 0.60;
            fit.write = bandwidth * 0.75;
            fit.cached = (fit.cached * scale).max(bandwidth * 2.0);
            fit.rate *= scale;
            fit.tile *= scale;
            Some(Box::new(fit.pool(name)))
        }
        _ => None,
    }
}
