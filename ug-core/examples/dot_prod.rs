use anyhow::Result;

use ug::lang::ssa::{Const, Instr as I, Kernel, VarId};

fn main() -> Result<()> {
    let v = VarId::new;
    let instrs = vec![
        /* 0 */ I::DefineGlobal(0),
        /* 1 */ I::DefineGlobal(1),
        /* 2 */ I::DefineGlobal(2),
        /* 3 */ I::Const(Const::I32(0)),
        /* 4 */ I::Const(Const::I32(2)),
        /* 5 */ I::Range { lo: v(3), up: v(4), end_idx: 10 },
        /* 6 */ I::Load { src: v(1), offset: v(5) },
        /* 7 */ I::Load { src: v(2), offset: v(5) },
        // TODO: add v(6) and v(7)
        /* 8 */
        I::Store { dst: v(0), offset: v(5), value: v(7) },
        /* 9 */ I::EndRange { start_idx: 5 },
    ];
    let kernel = Kernel { args: vec![], instrs };
    println!("{kernel:?}");
    let mut a = ug::interpreter::Buffer::I32(vec![1i32, 2]);
    let mut b = ug::interpreter::Buffer::I32(vec![3i32, 4]);
    let mut c = ug::interpreter::Buffer::I32(vec![0i32, 0]);
    ug::interpreter::eval_ssa(&kernel, vec![&mut a, &mut b, &mut c], &[])?;
    Ok(())
}
