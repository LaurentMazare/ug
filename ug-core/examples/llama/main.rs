use rand::prelude::*;
use ug::{CpuDevice, Error, Result};

mod custom;
mod model;
use model::{Cache, Model, LB};

#[allow(unused)]
const UNK_TOKEN: u32 = 0;
const BOS_TOKEN: u32 = 1;
#[allow(unused)]
const EOS_TOKEN: u32 = 2;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "smol2-135m")]
    Smol2_135M,
    #[value(name = "smol2-360m")]
    Smol2_360M,
    #[value(name = "smol2-1.7b")]
    Smol2_1B7,
    #[value(name = "3.2-1b")]
    L32_1B,
    #[value(name = "3.2-3b")]
    L32_3B,
}

#[derive(clap::Parser, Debug)]
struct Args {
    #[arg(long)]
    tracing: bool,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    custom_softmax: bool,

    #[arg(long, default_value = "smol2-135m")]
    which: Which,

    #[arg(short, long, default_value_t = 20)]
    n_steps: usize,
}

fn main() -> Result<()> {
    use clap::Parser;
    let args = Args::parse();

    let _guard = if args.tracing {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let api = hf_hub::api::sync::Api::new().map_err(Error::wrap)?;
    let hf_repo = match args.which {
        Which::Smol2_135M => "HuggingFaceTB/SmolLM2-135M",
        Which::Smol2_360M => "HuggingFaceTB/SmolLM2-360M",
        Which::Smol2_1B7 => "HuggingFaceTB/SmolLM2-1.7B",
        Which::L32_1B => "meta-llama/Llama-3.2-1B",
        Which::L32_3B => "meta-llama/Llama-3.2-3B",
    };
    let api = api.model(hf_repo.to_string());
    let model_file = api.get("model.safetensors").map_err(Error::wrap)?;
    let tokenizer_file = api.get("tokenizer.json").map_err(Error::wrap)?;
    let config_file = api.get("config.json").map_err(Error::wrap)?;

    let cfg = serde_json::from_slice(&std::fs::read(config_file)?).map_err(Error::wrap)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
        .map_err(|v| Error::debug(format!("{v:?}")))?;
    let st = unsafe { ug::safetensors::MmapedSafetensors::new(model_file)? };
    let model = Model::new(&cfg, args.custom_softmax, &st)?;
    let mut cache = Cache::new(&cfg)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut last_token = BOS_TOKEN;
    let mut ccache = ug::cache::CompilationCache::default();
    for pos in 0..args.n_steps {
        let token_ids = LB::copy([last_token as i32].as_slice(), (1, 1), &CpuDevice)?;
        let pos = LB::copy([pos as i32].as_slice(), (1, 1), &CpuDevice)?;
        let tensor = model.fwd(&token_ids, &pos, &mut cache)?;
        let tensor = if args.custom_softmax {
            model::custom_softmax(&tensor)?
        } else {
            model::softmax(&tensor)?
        };

        let start_time = std::time::Instant::now();
        let schedule = ug::Schedule::create_one(&tensor)?;
        let dt_schedule = start_time.elapsed();

        let start_time = std::time::Instant::now();
        let schedule = schedule.compile_with_cache(&mut ccache)?;
        let dt_compile = start_time.elapsed();

        let start_time = std::time::Instant::now();
        schedule.run()?;
        let dt_run = start_time.elapsed();
        let prs = tensor.data_vec::<f32>()?.unwrap();
        let dist = rand_distr::WeightedIndex::new(prs).map_err(Error::wrap)?;
        last_token = dist.sample(&mut rng) as u32;
        let token = tokenizer.id_to_token(last_token);
        if args.verbose {
            println!(
                "gen {:.2}ms, comp: {:.2}ms, run: {:.2}ms, generated {token:?}",
                dt_schedule.as_secs_f32() * 1000.,
                dt_compile.as_secs_f32() * 1000.,
                dt_run.as_secs_f32() * 1000.,
            );
        } else if let Some(token) = token {
            use std::io::Write;

            let token = token.replace('Ġ', " ").replace('Ċ', "\n");
            print!("{token}");
            std::io::stdout().flush().map_err(Error::wrap)?;
        }
    }
    println!();

    Ok(())
}
