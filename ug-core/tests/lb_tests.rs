use ug::{Device, Slice};
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
        let kernel = match item {
            ug::ScheduleItem::Kernel(k) => k.kernel()?,
            _ => ug::bail!("unexpected item"),
        };
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
    let cpu = ug::CpuDevice;
    let lhs = LB::cst(40., (5, 2), &cpu)?;
    let rhs = LB::cst(2., (5, 2), &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let items = schedule.items();
    assert_eq!(items.len(), 1);
    let kernel = match &items[0] {
        ug::ScheduleItem::Kernel(k) => k.kernel()?,
        _ => ug::bail!("unexpected item"),
    };
    let ssa = kernel.lower(&Default::default())?;
    let func = cpu.compile(&ssa)?;
    let mut buffer = unsafe { cpu.allocate_uninit(ug::DType::F32, 10)? };
    cpu.run(&func, &mut [&mut buffer])?;
    let buffer = buffer.to_vec::<f32>()?;
    assert_eq!(buffer, [42., 42., 42., 42., 42., 42., 42., 42., 42., 42.]);
    Ok(())
}

#[test]
fn schedule_cpu_compile() -> Result<()> {
    let cpu = ug::CpuDevice;
    let lhs = LB::cst(40., (5, 2), &cpu)?;
    let rhs = LB::cst(2., (5, 2), &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let schedule = schedule.compile()?;
    schedule.run()?;
    let data = lb.data().lock()?;
    let data = data.as_ref().unwrap().to_vec::<f32>()?;
    assert_eq!(data, [42., 42., 42., 42., 42., 42., 42., 42., 42., 42.]);
    Ok(())
}

#[test]
fn schedule_mm() -> Result<()> {
    let cpu = ug::CpuDevice;
    let lhs = LB::cst(1., (4, 2), &cpu)?;
    let rhs = LB::cst(2., (2, 3), &cpu)?;
    let lb = lhs.matmul(rhs)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let schedule = schedule.compile()?;
    schedule.run()?;
    let data = lb.data().lock()?;
    let data = data.as_ref().unwrap().to_vec::<f32>()?;
    assert_eq!(data, [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    Ok(())
}

#[test]
fn lb_copy() -> Result<()> {
    let cpu = ug::CpuDevice;
    let shape = (2, 3);
    let buf = ug::CpuStorage::F32(vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32]);
    let lhs = LB::copy(buf, shape, &cpu)?;
    let two = LB::cst(2., shape, &cpu)?;
    let half = LB::cst(0.5, shape, &cpu)?;
    let lb = lhs.binary(ug::lang::BinaryOp::Add, two)?;
    let lb = lb.binary(ug::lang::BinaryOp::Mul, half)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let schedule = schedule.compile()?;
    schedule.run()?;
    let data = {
        let d = lb.data().lock()?;
        d.as_ref().unwrap().to_vec::<f32>()?
    };
    assert_eq!(data, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);

    {
        let mut data = lhs.data().lock().unwrap();
        let data = data.as_mut().unwrap();
        if let ug::CpuStorage::F32(vs) = data {
            vs[1] = 0.;
            vs[2] = -8.;
            vs[5] = -2.;
        }
    }

    schedule.run()?;
    let data = {
        let d = lb.data().lock()?;
        d.as_ref().unwrap().to_vec::<f32>()?
    };
    assert_eq!(data, [1.0, 1.0, -3.0, 2.5, 3.0, 0.0]);

    Ok(())
}

#[test]
fn lb_custom() -> Result<()> {
    let cpu = ug::CpuDevice;
    let add_one = |mut v: Vec<&mut ug::CpuStorage>| -> Result<()> {
        match &mut v[0] {
            ug::CpuStorage::F32(v) => {
                for elem in v.iter_mut() {
                    *elem += 1.0
                }
            }
            _ => ug::bail!("unexpected dtype"),
        }
        Ok(())
    };
    let shape = (2, 3);
    let buf = ug::CpuStorage::F32(vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32]);
    let buf_lb = LB::copy(buf, shape, &cpu)?;
    let two_lb = LB::cst(2., shape, &cpu)?;
    let lb = buf_lb.custom(std::sync::Arc::new(add_one), vec![])?;
    let lb = lb.binary(ug::lang::BinaryOp::Mul, two_lb)?;
    let schedule = ug::Schedule::create_one(&lb)?;
    let schedule = schedule.compile()?;
    schedule.run()?;
    let data = {
        let d = lb.data().lock()?;
        d.as_ref().unwrap().to_vec::<f32>()?
    };
    assert_eq!(data, [2., 4., 6., 8., 10., 12.]);

    {
        let mut data = buf_lb.data().lock().unwrap();
        let data = data.as_mut().unwrap();
        if let ug::CpuStorage::F32(vs) = data {
            vs[1] = 0.;
            vs[2] = -8.;
            vs[5] = -2.;
        }
    }

    schedule.run()?;
    let data = {
        let d = lb.data().lock()?;
        d.as_ref().unwrap().to_vec::<f32>()?
    };
    assert_eq!(data, [4., 2., -14., 10., 12., -2.]);

    Ok(())
}
