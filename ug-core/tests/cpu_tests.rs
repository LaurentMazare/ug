use ug::Result;

#[test]
fn run_add() -> Result<()> {
    let kernel = ug::samples::ssa::simple_add(6);
    let cpu = ug::CpuDevice;
    let mut c_code = vec![];
    ug::cpu_code_gen::gen(&mut c_code, "foo", &kernel)?;
    let _func = cpu.compile(&c_code, "foo".into())?;
    Ok(())
}
