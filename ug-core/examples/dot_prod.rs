use anyhow::Result;

use ug::lang::ssa::{Const, Instr as I, Kernel, VarId};

fn main() -> Result<()> {
    let v = VarId::new;
    let instrs = vec![
        I::DefineGlobal(0),
        I::DefineGlobal(1),
        I::DefineGlobal(2),
        I::Const(Const::I32(0)),
        I::Const(Const::I32(1024)),
        I::Range { lo: v(3), up: v(4), end_idx: 7 },
        I::EndRange { start_idx: 5 },
    ];
    let kernel = Kernel { args: vec![], instrs };
    println!("{kernel:?}");
    ug::interpreter::eval_ssa(&kernel, vec![], &[])?;
    Ok(())
}
