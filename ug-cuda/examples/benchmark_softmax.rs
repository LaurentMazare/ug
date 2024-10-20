// This benchmark has a setup similar to the triton benchmark:
// https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
use anyhow::Result;
use rand::Rng;

const N_ROWS: usize = 4096;

fn slow_softmax(n_rows: usize, n_cols: usize) -> Result<()> {
    let mut rng = rand::thread_rng();
    let kernel = ug::samples::op::softmax(n_rows, n_cols)?;
    let kernel = kernel.lower(&Default::default())?;
    let mut buf = vec![];
    ug_cuda::code_gen::gen(&mut buf, "dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    println!("{cuda_code}");
    let device = ug_cuda::runtime::Device::new(0)?;
    let func = device.compile_cu(&cuda_code, "foo", "dotprod")?;
    let n_elements = n_rows * n_cols;
    let res = device.zeros(n_elements)?;
    let arg: Vec<f32> = (0..n_elements).map(|_| rng.gen()).collect();
    let arg = device.slice_from_values(&arg)?;
    let run = || {
        unsafe {
            func.launch2(arg.slice(), res.slice(), cudarc::driver::LaunchConfig::for_num_elems(1))?
        }
        device.synchronize()?;
        Ok::<_, anyhow::Error>(())
    };
    println!("warmup {:?}", kernel.flops_mem_per_thread()?);
    run()?;
    let start_time = std::time::Instant::now();
    let mut n_reps = 0;
    loop {
        run()?;
        n_reps += 1;
        let elapsed = start_time.elapsed().as_secs_f64();
        if elapsed > 2. {
            println!(
                "rows: {n_rows:4}    cols: {n_cols:4}    reps: {n_reps:4}    time-per-rep: {:.2}",
                elapsed / (n_reps as f64)
            );
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    for n_cols in [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096] {
        slow_softmax(N_ROWS, n_cols)?;
    }
    Ok(())
}
