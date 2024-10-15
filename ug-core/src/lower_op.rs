#![allow(unused)]
use crate::lang::{self, ssa};
use crate::lower::{Block, Id};
use anyhow::Result;
use ssa::Instr as SsaI;

impl lang::op::Kernel {
    fn lower_b(&self) -> Result<Block> {
        let mut instrs = vec![];
        let mut per_arg = std::collections::HashMap::new();
        for (index, arg) in self.args.iter().enumerate() {
            let id = Id::new();
            let dtype = match arg.type_() {
                lang::ArgType::Ptr => ssa::DType::PtrF32, // TODO(laurent): support other pointer types
                lang::ArgType::I32 => ssa::DType::I32,
            };
            instrs.push((id, SsaI::DefineGlobal { index, dtype }));
            per_arg.insert(arg.id(), id.to_varid());
        }

        for lang::op::Store { dst, layout, value } in self.ops.iter() {
            let ptr_i = Id::new().to_varid(); // TODO
            let off_i = Id::new().to_varid(); // TODO
            let src_i = Id::new().to_varid(); // TODO
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i,
                value: src_i,
                dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
            };
            instrs.push((Id::new(), store));
        }
        Ok(Block::new(instrs))
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
