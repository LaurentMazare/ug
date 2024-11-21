// This benchmark has a setup similar to the triton benchmark:
// https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
use rand::Rng;
use ug::Result;

#[derive(clap::ValueEnum, Clone, Debug)]
enum Which {
    Exp,
    ExpBlock,
    Softmax,
    SsaSoftmax,
    SsaSoftmaxReduce,
}

#[derive(clap::Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "exp")]
    which: Which,

    #[arg(long, default_value_t = 4096)]
    n_rows: usize,

    #[arg(short, long)]
    verbose: bool,

    #[arg(short, long)]
    disable_metal_validation: bool,
}

fn run_one(args: &Args, n_cols: usize) -> Result<()> {
    let mut rng = rand::thread_rng();
    let n_rows = args.n_rows;
    let (ssa_kernel, block_dim) = match args.which {
        Which::Exp => (ug::samples::ssa::exp(n_cols)?, n_cols),
        Which::ExpBlock => {
            let bs = if n_cols <= 1024 {
                n_cols
            } else if n_cols % 1024 != 0 {
                512
            } else {
                1024
            };
            (ug::samples::ssa::exp_block(n_cols, bs)?, bs)
        }
        Which::SsaSoftmax => (ug::samples::ssa::softmax(n_rows, n_cols)?, n_cols),
        Which::SsaSoftmaxReduce => {
            let bs = if n_cols <= 1024 {
                n_cols
            } else if n_cols % 1024 != 0 {
                512
            } else {
                1024
            };
            (ug::samples::ssa::softmax_block(n_rows, n_cols, bs)?, bs)
        }
        Which::Softmax => {
            let kernel = ug::samples::op::softmax(n_rows, n_cols)?;
            let lower_opts =
                ug::lower_op::Opts::default().with_block_axis(0).with_thread_block(1, n_cols);
            (kernel.lower(&lower_opts)?, n_cols)
        }
    };
    let mut buf = vec![];
    ug_metal::code_gen::gen(&mut buf, "mykernel", &ssa_kernel)?;
    let metal_code = String::from_utf8(buf)?;
    if args.verbose {
        println!("SSA\n{ssa_kernel:?}");
        println!("METAL\n{metal_code}");
    }
    let device = ug_metal::runtime::Device::new()?;
    let launch_config = ug::lang::LaunchConfig {
        grid_dim: n_rows as u32,
        block_dim: block_dim as u32,
        shared_mem: 0,
    };
    let func = device.compile_metal(&metal_code, "mykernel", launch_config)?;
    let n_elements = n_rows * n_cols;
    let res = device.zeros::<f32>(n_elements)?;
    let arg: Vec<f32> = (0..n_elements).map(|_| rng.gen()).collect();
    let arg = device.slice_from_values(&arg)?;
    let cq = device.new_command_queue();
    let run = || {
        let cb = cq.new_command_buffer();
        let (arg, res) = (arg.buffer(), res.buffer());
        let encoder = cb.new_compute_command_encoder();
        let pl = func.pipeline()?;
        encoder.set_compute_pipeline_state(&pl);
        ug_metal::set_params!(encoder, (arg, res));
        encoder.use_resource(arg, metal::MTLResourceUsage::Read);
        encoder.use_resource(res, metal::MTLResourceUsage::Write);
        let grid_size = metal::MTLSize::new(n_rows as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(usize::min(block_dim, 1024) as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok::<_, ug::Error>(())
    };
    println!("warmup {:?}", ssa_kernel.flops_mem_per_thread()?);
    run()?;
    let start_time = std::time::Instant::now();
    let mut n_reps = 0;
    loop {
        run()?;
        n_reps += 1;
        let elapsed = start_time.elapsed().as_secs_f64();
        if elapsed > 2. {
            println!(
                "rows: {n_rows:4}    cols: {n_cols:4}    reps: {n_reps:4}    time-per-rep: {:.2}s    {:.3} GB/s",
                elapsed / (n_reps as f64),
                n_reps as f64 * n_rows as f64 * n_cols as f64 * 8e-9 / elapsed
            );
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    use clap::Parser;

    let args = Args::parse();
    println!("{args:?}");
    if !args.disable_metal_validation {
        std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1")
    }
    objc::rc::autoreleasepool(|| {
        for n_cols in [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096] {
            run_one(&args, n_cols)?;
        }
        Ok::<_, ug::Error>(())
    })?;
    Ok(())
}
