use anyhow::Result;

use ug::lang::ssa::{self, DType, Instr, Kernel, VarId};

fn main() -> Result<()> {
    let (v0, c0) = ssa::const_i32(0);
    let (v1024, c1024) = ssa::const_i32(1024);
    let range_idx = VarId::new();
    let instrs = vec![
        c0,
        c1024,
        Instr::Affect {
            var_id: range_idx,
            expr: ssa::Expr::Range { lo: v0, up: v1024, end_idx: 4 },
            dtype: DType::I32,
        },
        Instr::EndRange { start_idx: 2 },
    ];
    let kernel = Kernel { args: vec![], instrs };
    println!("{kernel:?}");
    ug::interpreter::eval_ssa(&kernel, &[])?;
    Ok(())
}
