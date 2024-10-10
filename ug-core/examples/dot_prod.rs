use anyhow::Result;

use ug::lang::ssa::{BinaryOp, Const, Instr as I, Kernel, VarId};

fn eval_add() -> Result<()> {
    let v = VarId::new;
    let instrs = vec![
        /* 0 */ I::DefineGlobal(0),
        /* 1 */ I::DefineGlobal(1),
        /* 2 */ I::DefineGlobal(2),
        /* 3 */ I::Const(Const::I32(0)),
        /* 4 */ I::Const(Const::I32(2)),
        /* 5 */ I::Range { lo: v(3), up: v(4), end_idx: 11 },
        /* 6 */ I::Load { src: v(1), offset: v(5) },
        /* 7 */ I::Load { src: v(2), offset: v(5) },
        /* 8 */ I::Binary { op: self::BinaryOp::Add, lhs: v(6), rhs: v(7) },
        /* 9 */ I::Store { dst: v(0), offset: v(5), value: v(8) },
        /* 10 */ I::EndRange { start_idx: 5 },
    ];
    let kernel = Kernel { args: vec![], instrs };
    println!("{kernel:?}");
    let mut a = ug::interpreter::Buffer::I32(vec![0i32, 0]);
    let mut b = ug::interpreter::Buffer::I32(vec![3i32, 4]);
    let mut c = ug::interpreter::Buffer::I32(vec![1i32, 2]);
    ug::interpreter::eval_ssa(&kernel, vec![&mut a, &mut b, &mut c], &[])?;
    println!("a: {a:?}\nb: {b:?}\nc: {c:?}");
    Ok(())
}

fn eval_dotprod() -> Result<()> {
    let v = VarId::new;
    let instrs = vec![
        /* 0 */ I::DefineGlobal(0),
        /* 1 */ I::DefineGlobal(1),
        /* 2 */ I::DefineGlobal(2),
        /* 3 */ I::Const(Const::I32(0)),
        /* 4 */ I::Const(Const::I32(2)),
        /* 5 */ I::DefineAcc(Const::I32(0)),
        /* 6 */ I::Range { lo: v(3), up: v(4), end_idx: 13 },
        /* 7 */ I::Load { src: v(1), offset: v(6) },
        /* 8 */ I::Load { src: v(2), offset: v(6) },
        /* 9 */ I::Binary { op: self::BinaryOp::Mul, lhs: v(7), rhs: v(8) },
        /* 10*/ I::Binary { op: self::BinaryOp::Add, lhs: v(9), rhs: v(5) },
        /* 11*/ I::Assign { dst: v(5), src: v(10) },
        /* 12*/ I::EndRange { start_idx: 6 },
        /* 13*/ I::Store { dst: v(0), offset: v(3), value: v(5) },
    ];
    let kernel = Kernel { args: vec![], instrs };
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
