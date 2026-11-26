//! A FLUX.2 transformer from a diffusers directory.

use crate::arch::{Arch, Spot, arch};
use crate::names::{Rule, repo};
use crate::parts::{Cut, Part};
use crate::{Overrides, Stream, Weights, numbered, pins, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::File;
use tessera_engine::DimName::Rows;
use tessera_engine::flux2::{Config, program};
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

struct Flux2Stream {
    graph: Graph,
    file: File,
    rules: Vec<Rule>,
}

impl Stream for Flux2Stream {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        numbered(&self.graph, &self.file, &[&self.rules], shares, take)
    }
}

fn part(_: &Arch, cut: &Cut) -> Result<Part> {
    let root = cut.root();
    let [seqs, text] = cut.sized(512);
    let sizes = [seqs, text, cut.at(Rows, 1024)];
    let graph = graph(&root, sizes, &Overrides::default())?;
    Ok(Part::new(graph, move |g| weights(&root, g)))
}

arch!(
    "flux2",
    [Spot::Class("Flux2Transformer2DModel")],
    rules: include_str!("../names/flux2.json"),
    part: part,
);

pub struct Transformer {
    pub graph: Graph,
    pub weights: Weights,
    pub pins: Pins,
}

/// `dir/transformer`, with `text` rows and up to `rows` image rows.
pub fn transformer(dir: &Path, sizes: [u64; 3], overrides: &Overrides) -> Result<Transformer> {
    let graph = graph(dir, sizes, overrides)?;
    let file = File::open(&dir.join("transformer/diffusion_pytorch_model.safetensors"))?;
    let rules = repo("flux2").unwrap_or_default();
    let pins = pins(&graph, &file, 0, &[&rules])?;
    let weights = Weights::Streamed(Box::new(Flux2Stream {
        graph: graph.clone(),
        file,
        rules,
    }));
    Ok(Transformer {
        graph,
        weights,
        pins,
    })
}

/// The transformer `dir/transformer/config.json` describes.
pub fn graph(dir: &Path, [seqs, text, rows]: [u64; 3], overrides: &Overrides) -> Result<Graph> {
    const FILE: &str = "transformer/config.json";
    let config = overrides.read(FILE, Some(&dir.join(FILE)), |k| {
        Config::read(k, seqs, text, rows)
    })?;
    Ok(program(&config))
}

/// The weights `graph` reads from `dir/transformer`, and their forms.
pub fn weights(dir: &Path, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
    let file = File::open(&dir.join("transformer/diffusion_pytorch_model.safetensors"))?;
    let rules = repo("flux2").unwrap_or_default();
    pull(graph, &file, 0, &[&rules])
}
