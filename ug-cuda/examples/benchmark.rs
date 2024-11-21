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
    ug_cuda::code_gen::gen(&mut buf, "mykernel", &ssa_kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    if args.verbose {
        println!("SSA\n{ssa_kernel:?}");
        println!("CUDA\n{cuda_code}");
    }
    let device = ug_cuda::runtime::Device::new(0)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = device.compile_cu(&cuda_code, "foo", "mykernel")?;
    let func = ug_cuda::runtime::Func::new(func, cfg);
    let n_elements = n_rows * n_cols;
    let res = device.zeros(n_elements)?;
    let arg: Vec<f32> = (0..n_elements).map(|_| rng.gen()).collect();
    let arg = device.slice_from_values(&arg)?;
    let run = || {
        unsafe { func.launch2((arg.slice::<f32>()?, res.slice::<f32>()?))? }
        device.synchronize()?;
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
    for n_cols in [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096] {
        run_one(&args, n_cols)?;
    }
    Ok(())
}
