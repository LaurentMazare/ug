use anyhow::Result;

fn eval_add() -> Result<()> {
    let kernel = ug::samples::simple_add(1024);
    println!("{kernel:?}");
    let mut w = std::io::stdout();
    ug_cuda::code_gen::gen(&mut w, "add", &kernel)?;
    Ok(())
}

fn eval_dotprod() -> Result<()> {
    let kernel = ug::samples::simple_dotprod(1024);
    println!("{kernel:?}");
    let mut buf = vec![];
    ug_cuda::code_gen::gen(&mut buf, "dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    println!("<<<< CUDA CODE >>>>\n{cuda_code}\nflops-mem: {:?}", kernel.flops_mem_per_thread()?);
    let device = ug_cuda::runtime::Device::new(0)?;
    let func = device.compile_cu(&cuda_code, "foo", "dotprod")?;
    let res = device.zeros(1)?;
    let lhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let rhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = device.slice_from_values(&lhs)?;
    let rhs = device.slice_from_values(&rhs)?;
    unsafe { func.launch3(res.slice(), lhs.slice(), rhs.slice(), 1)? };
    let res = res.to_vec()?;
    println!("res: {res:?}");
    Ok(())
}

fn main() -> Result<()> {
    eval_add()?;
    eval_dotprod()?;
    Ok(())
}
