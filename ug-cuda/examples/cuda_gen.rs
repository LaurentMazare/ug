use ug::{Device, Result, Slice};

fn eval_add() -> Result<()> {
    let kernel = ug::samples::ssa::simple_add(1024)?;
    println!("{kernel:?}");
    let mut w = std::io::stdout();
    ug_cuda::code_gen::gen(&mut w, "add", &kernel)?;
    Ok(())
}

fn eval_dotprod() -> Result<()> {
    let kernel = ug::samples::ssa::simple_dotprod(1024)?;
    println!("{kernel:?}");
    let mut buf = vec![];
    ug_cuda::code_gen::gen(&mut buf, "dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    println!("<<<< CUDA CODE >>>>\n{cuda_code}\nflops-mem: {:?}", kernel.flops_mem_per_thread()?);
    let device = ug_cuda::runtime::Device::new(0)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
    let func = device.compile_cu(&cuda_code, "dotprod")?;
    let func = ug_cuda::runtime::Func::new(&device, func, cfg);
    let mut res = device.zeros(1)?;
    let lhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let rhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let mut lhs = device.slice_from_values(&lhs)?;
    let mut rhs = device.slice_from_values(&rhs)?;
    device.run(&func, &mut [&mut res, &mut lhs, &mut rhs])?;
    let res: Vec<f32> = res.to_vec()?;
    println!("res: {res:?}");
    Ok(())
}

fn eval_lower_add() -> Result<()> {
    let kernel = ug::samples::simple_add(1024)?;
    println!("<<<< ADD LANG >>>>\n{kernel:?}");
    let kernel = kernel.lower()?;
    println!("<<<< ADD SSA >>>>\n{kernel:?}");
    let mut buf = vec![];
    ug_cuda::code_gen::gen(&mut buf, "dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    println!("<<<< CUDA CODE >>>>\n{cuda_code}\nflops-mem: {:?}", kernel.flops_mem_per_thread()?);
    let device = ug_cuda::runtime::Device::new(0)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
    let func = device.compile_cu(&cuda_code, "dotprod")?;
    let func = ug_cuda::runtime::Func::new(&device, func, cfg);
    let mut res = device.zeros(1024)?;
    let lhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let rhs = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
    let mut lhs = device.slice_from_values(&lhs)?;
    let mut rhs = device.slice_from_values(&rhs)?;
    device.run(&func, &mut [&mut lhs, &mut rhs, &mut res])?;
    let res: Vec<f32> = res.to_vec()?;
    println!("res: {res:?}");
    Ok(())
}

fn eval_softmax() -> Result<()> {
    let kernel = ug::samples::op::softmax(2, 4)?;
    println!("<<<< ADD LANG >>>>\n{kernel:?}");
    let kernel = kernel.lower(&Default::default())?;
    println!("<<<< ADD SSA >>>>\n{kernel:?}");
    let mut buf = vec![];
    ug_cuda::code_gen::gen(&mut buf, "dotprod", &kernel)?;
    let cuda_code = String::from_utf8(buf)?;
    println!("<<<< CUDA CODE >>>>\n{cuda_code}\nflops-mem: {:?}", kernel.flops_mem_per_thread()?);
    let device = ug_cuda::runtime::Device::new(0)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
    let func = device.compile_cu(&cuda_code, "dotprod")?;
    let func = ug_cuda::runtime::Func::new(&device, func, cfg);
    let mut res = device.zeros(8)?;
    let arg = vec![0., 1., 2., 3., 2., 1., 2., 1.];
    let mut arg = device.slice_from_values(&arg)?;
    device.run(&func, &mut [&mut arg, &mut res])?;
    let res: Vec<f32> = res.to_vec()?;
    println!("res: {res:?}");
    Ok(())
}

fn main() -> Result<()> {
    println!("> ADD");
    eval_add()?;
    println!("> DOTPROD");
    eval_dotprod()?;
    println!("> LOWER ADD");
    eval_lower_add()?;
    println!("> LOWER SOFTMAX");
    eval_softmax()?;
    Ok(())
}
