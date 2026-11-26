//! `fetch`: download a recipe and its weights.

use crate::source::{Source, Stop};
use tessera_tasks::Error;
use tessera_tasks::task::{Run, Sink, Task};

/// Download a model and print where it is.
#[derive(Task)]
#[task(example = "download the {kind} model")]
pub struct Fetch {}

impl Run<Source> for Fetch {
    fn run(self, source: Source, sink: &mut dyn Sink) -> Result<(), Error> {
        let mut line = |key: &str, dir: &std::path::Path| {
            sink.text(&format!("{key} {}\n", dir.display()));
        };
        if let Some(recipe) = &source.recipe {
            line("recipe", &recipe.dir);
            let (cache, token) = (source.cache.as_deref(), source.token.as_deref());
            let progress = &source.progress;
            let weights = match source.override_weights() {
                Some(dir) => dir,
                None => recipe.weights(cache, token, progress)?,
            };
            line("weights", &weights);
            for (flag, dir) in recipe.extras(cache, token, progress)? {
                line(flag.trim_start_matches('-'), &dir);
            }
            return Ok(());
        }
        if let Some(dir) = &source.fetched {
            line("weights", dir);
            return Ok(());
        }
        Err(Stop::Usage.into())
    }
}
