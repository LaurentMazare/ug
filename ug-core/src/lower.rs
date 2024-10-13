use crate::lang::{self, ssa};
use anyhow::{Context, Result};

pub fn lower(kernel: &lang::Kernel) -> Result<ssa::Kernel> {
    let mut instrs = vec![];
    for (index, arg) in kernel.args.iter().enumerate() {
        let dtype = match arg.type_() {
            lang::ArgType::Ptr => ssa::DType::PtrF32, // TODO(laurent): other pointer types
            lang::ArgType::I32 => ssa::DType::I32,
        };
        instrs.push(ssa::Instr::DefineGlobal { index, dtype })
    }
    for op in kernel.ops.iter() {
        let lang::Ops::Store { dst, src } = op;
        let _ptr = dst.ptr();
        let _offset = dst.offset();
        let len = dst.len().as_const().context("non-const len")?;
        let stride = dst.stride().as_const().context("non-const stride")?;
        let _src = src;
        instrs.push(ssa::Instr::Const(ssa::Const::I32(stride as i32)));
        instrs.push(ssa::Instr::Const(ssa::Const::I32(len as i32)));
        instrs.push(ssa::Instr::Range {
            lo: ssa::VarId::new(0),
            up: ssa::VarId::new(len),
            end_idx: 42,
        });
        instrs.push(ssa::Instr::EndRange { start_idx: 42 });
    }
    Ok(ssa::Kernel { instrs })
}
