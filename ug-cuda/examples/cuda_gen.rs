use anyhow::Result;

fn eval_add() -> Result<()> {
    let kernel = ug::samples::simple_add(2);
    println!("{kernel:?}");
    let mut w = std::io::stdout();
    ug_cuda::code_gen::gen(&mut w, "add", &kernel)?;
    Ok(())
}

fn eval_dotprod() -> Result<()> {
    let kernel = ug::samples::simple_dotprod(2);
    println!("{kernel:?}");
    let mut w = std::io::stdout();
    ug_cuda::code_gen::gen(&mut w, "dotprod", &kernel)?;
    Ok(())
}

fn main() -> Result<()> {
    eval_add()?;
    eval_dotprod()?;
    Ok(())
}
