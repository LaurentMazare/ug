use ug::{Layout, LazyBuffer as LB, Result};

#[test]
fn schedule() -> Result<()> {
    let cpu = ug::CpuDevice;
    let lhs = LB::cst(40., (5, 2), &cpu)?;
    let rhs = LB::cst(2., (5, 2), &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let items = schedule.items();
    println!("{schedule:?}");
    assert_eq!(items.len(), 1);
    for (idx, item) in items.iter().enumerate() {
        let ast = item.ast().clone();
        let arg = ug::lang::op::Arg::new(ug::lang::Type::Ptr(ug::DType::F32));
        let sto = ug::lang::op::store(arg.id(), Layout::from_shape((5, 2)), ast)?;
        let kernel = ug::lang::op::Kernel::new(format!("kernel{idx}"), vec![arg], vec![sto]);
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
