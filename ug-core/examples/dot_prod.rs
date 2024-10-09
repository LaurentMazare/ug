use anyhow::Result;

use ug::lang::ssa::{Const, Instr as I, Kernel, VarId};

fn main() -> Result<()> {
    let v = VarId::new;
    let instrs = vec![
        I::Const(Const::I32(0)),
        I::Const(Const::I32(1024)),
        I::Range { lo: v(0), up: v(1), end_idx: 4 },
        I::EndRange { start_idx: 2 },
    ];
    let kernel = Kernel { args: vec![], instrs };
    println!("{kernel:?}");
    ug::interpreter::eval_ssa(&kernel, vec![], &[])?;
    Ok(())
}
