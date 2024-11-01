use ug::{LazyBuffer as LB, Result};

#[test]
fn schedule_interpret() -> Result<()> {
    let cpu = ug::CpuDevice;
    let lhs = LB::cst(40., (5, 2), &cpu)?;
    let rhs = LB::cst(2., (5, 2), &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let items = schedule.items();
    assert_eq!(items.len(), 1);
    for item in items.iter() {
        let kernel = item.kernel()?;
        let ssa = kernel.lower(&Default::default())?;
        println!("{ssa:?}");
        let mut buffer = ug::interpreter::Buffer::F32(vec![0f32; 10]);
        let args = vec![];
        ug::interpreter::eval_ssa::<1>(&ssa, vec![&mut buffer], &args, 0)?;
        let buffer = buffer.as_f32()?;
        assert_eq!(buffer, [42., 42., 42., 42., 42., 42., 42., 42., 42., 42.]);
    }
    Ok(())
}

#[test]
fn schedule_cpu() -> Result<()> {
    use ug::{Device, Slice};

    let cpu = ug::CpuDevice;
    let lhs = LB::cst(40., (5, 2), &cpu)?;
    let rhs = LB::cst(2., (5, 2), &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let items = schedule.items();
    assert_eq!(items.len(), 1);
    let kernel = items[0].kernel()?;
    let ssa = kernel.lower(&Default::default())?;
    let func = cpu.compile(&ssa)?;
    let mut buffer = unsafe { cpu.allocate_uninit::<f32>(10)? };
    cpu.run(&func, &mut [&mut buffer])?;
    let buffer = buffer.to_vec::<f32>()?;
    assert_eq!(buffer, [42., 42., 42., 42., 42., 42., 42., 42., 42., 42.]);
    Ok(())
}
