use anyhow::Result;

#[test]
fn eval_add() -> Result<()> {
    let kernel = ug::samples::ssa::simple_add(2);
    let mut a = ug::interpreter::Buffer::I32(vec![0i32, 0]);
    let mut b = ug::interpreter::Buffer::I32(vec![3i32, 4]);
    let mut c = ug::interpreter::Buffer::I32(vec![1i32, 2]);
    ug::interpreter::eval_ssa::<1>(&kernel, vec![&mut a, &mut b, &mut c], &[], 0)?;
    let a = a.as_i32()?;
    assert_eq!(a, [4, 6]);
    Ok(())
}

#[test]
fn eval_dotprod() -> Result<()> {
    let kernel = ug::samples::ssa::simple_dotprod(2);
    let mut a = ug::interpreter::Buffer::F32(vec![0f32]);
    let mut b = ug::interpreter::Buffer::F32(vec![3f32, 4.5]);
    let mut c = ug::interpreter::Buffer::F32(vec![1f32, 2.]);
    ug::interpreter::eval_ssa::<1>(&kernel, vec![&mut a, &mut b, &mut c], &[], 0)?;
    let a = a.as_f32()?;
    assert_eq!(a, [12.]);
    Ok(())
}

#[test]
fn lower_add() -> Result<()> {
    let kernel = ug::samples::simple_add(2);
    let ssa_kernel = kernel.lower()?;
    let mut c = ug::interpreter::Buffer::F32(vec![0f32, 0.]);
    let mut b = ug::interpreter::Buffer::F32(vec![3f32, 4.]);
    let mut a = ug::interpreter::Buffer::F32(vec![1f32, 2.]);
    ug::interpreter::eval_ssa::<1>(&ssa_kernel, vec![&mut a, &mut b, &mut c], &[], 0)?;
    let c = c.as_f32()?;
    assert_eq!(c, [4., 6.]);
    Ok(())
}

#[test]
fn softmax() -> Result<()> {
    let kernel = ug::samples::op::softmax(2, 4)?;
    let ssa_kernel = kernel.lower()?;
    let mut a = ug::interpreter::Buffer::F32(vec![0., 1., 2., 3., 2., 1., 2., 1.]);
    let mut b = ug::interpreter::Buffer::F32(vec![0f32; 8]);
    ug::interpreter::eval_ssa::<1>(&ssa_kernel, vec![&mut a, &mut b], &[], 0)?;
    let b = b.as_f32()?;
    assert_eq!(
        b,
        [
            0.032058604,
            0.08714432,
            0.23688284,
            0.6439143,
            0.3655293,
            0.13447072,
            0.3655293,
            0.13447072
        ]
    );
    Ok(())
}