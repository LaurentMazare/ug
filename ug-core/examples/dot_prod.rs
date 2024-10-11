use anyhow::Result;

fn eval_add() -> Result<()> {
    let kernel = ug::samples::simple_add(2);
    println!("{kernel:?}");
    let mut a = ug::interpreter::Buffer::I32(vec![0i32, 0]);
    let mut b = ug::interpreter::Buffer::I32(vec![3i32, 4]);
    let mut c = ug::interpreter::Buffer::I32(vec![1i32, 2]);
    ug::interpreter::eval_ssa(&kernel, vec![&mut a, &mut b, &mut c], &[])?;
    println!("a: {a:?}\nb: {b:?}\nc: {c:?}");
    Ok(())
}

fn eval_dotprod() -> Result<()> {
    let kernel = ug::samples::simple_dotprod(2);
    println!("{kernel:?}");
    let mut a = ug::interpreter::Buffer::I32(vec![0i32]);
    let mut b = ug::interpreter::Buffer::I32(vec![3i32, 4]);
    let mut c = ug::interpreter::Buffer::I32(vec![1i32, 2]);
    ug::interpreter::eval_ssa(&kernel, vec![&mut a, &mut b, &mut c], &[])?;
    println!("a: {a:?}\nb: {b:?}\nc: {c:?}");
    Ok(())
}

fn main() -> Result<()> {
    eval_add()?;
    eval_dotprod()?;
    Ok(())
}
