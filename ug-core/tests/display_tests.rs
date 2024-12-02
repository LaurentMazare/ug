use ug::{CpuDevice, DType, Device, LazyBuffer as LB, Result};

fn mul_cst<D: Device>(lb: &LB<D>, v: f32) -> Result<LB<D>> {
    let t = LB::cst(v, lb.shape(), lb.device())?;
    println!(">>> {:?} {:?}", lb.dtype(), t.dtype());
    lb.binary(ug::lang::BinaryOp::Mul, t)
}

#[test]
fn display_scalar() -> Result<()> {
    let dev = &CpuDevice;
    let t = LB::copy([1234i32].as_slice(), (), dev)?;
    let s = format!("{t}");
    assert_eq!(&s, "[1234]\nTensor[[], i32]");
    let t = t.cast(DType::F32)?.unary(ug::lang::UnaryOp::Neg)?;
    let s = format!("{}", mul_cst(&t, 0.1)?);
    assert_eq!(&s, "[-123.4000]\nTensor[[], f32]");
    let s = format!("{}", mul_cst(&t, 1e-8)?);
    assert_eq!(&s, "[-1.2340e-5]\nTensor[[], f32]");
    let s = format!("{}", mul_cst(&t, 1e8)?);
    assert_eq!(&s, "[-1.2340e11]\nTensor[[], f32]");
    let s = format!("{}", mul_cst(&t, 0.0)?);
    assert_eq!(&s, "[0.]\nTensor[[], f32]");
    Ok(())
}

/*
#[test]
fn display_vector() -> Result<()> {
    let t = Tensor::new::<&[u32; 0]>(&[], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(&s, "[]\nTensor[[0], u32]");
    let t = Tensor::new(&[0.1234567, 1.0, -1.2, 4.1, f64::NAN], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(&s, "[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], f64]");
    let t = (Tensor::ones(50, DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42.]
Tensor[[50], f32]"#;
    assert_eq!(&s, expected);
    let t = (Tensor::ones(11000, DType::F32, &Cpu)? * 42.)?;
    let s = format!("{t}");
    assert_eq!(&s, "[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], f32]");
    Ok(())
}

#[test]
fn display_multi_dim() -> Result<()> {
    let t = (LB::ones((200, 100), DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[[42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 ...
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.]]
Tensor[[200, 100], f32]"#;
    assert_eq!(&s, expected);
    let t = t.reshape(&[2, 1, 1, 100, 100])?;
    let t = format!("\n{t}");
    let expected = r#"
[[[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]],
 [[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]]]
Tensor[[2, 1, 1, 100, 100], f32]"#;
    assert_eq!(&t, expected);
    Ok(())
}
*/
