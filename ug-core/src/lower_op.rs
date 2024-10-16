use crate::lang::{self, ssa};
use crate::lower::{Block, Id};
use anyhow::Result;
use ssa::Instr as SsaI;

impl lang::op::Layout {
    fn lower(&self, range_id: Id) -> Result<(Id, Block)> {
        if !self.c_contiguous() {
            anyhow::bail!("only contiguous arrays are supported")
        }
        let offset = self.offset();
        if offset == 0 {
            Ok((range_id, Block(vec![])))
        } else {
            let dst_id = Id::new();
            let off_id = Id::new();
            let off = SsaI::Const(ssa::Const::I32(offset as i32));
            let add = SsaI::Binary {
                op: lang::BinaryOp::Add,
                lhs: range_id.to_varid(),
                rhs: off_id.to_varid(),
                dtype: ssa::DType::I32,
            };

            Ok((dst_id, Block(vec![(off_id, off), (dst_id, add)])))
        }
    }
}

impl lang::op::Ast {
    #[allow(clippy::only_used_in_recursion)]
    fn lower(
        &self,
        range_id: Id,
        per_arg: &std::collections::HashMap<lang::ArgId, ssa::VarId>,
    ) -> Result<(Id, Block)> {
        use lang::op::AstInner as A;
        let dtype = self.dtype;
        let dst_i = Id::new();
        let instrs = match self.inner.as_ref() {
            A::Load { src, layout } => {
                let ptr_i = match per_arg.get(src) {
                    None => anyhow::bail!("unknown arg {src:?}"),
                    Some(id) => *id,
                };
                let (off_i, off_b) = layout.lower(range_id)?;
                let load = SsaI::Load { src: ptr_i, dtype, offset: off_i.to_varid() };
                let mut off_b = off_b.0;
                off_b.push((dst_i, load));
                off_b
            }
            A::Unary { op, arg } => {
                let (arg_i, arg_b) = arg.lower(range_id, per_arg)?;
                let mut arg_b = arg_b.0;
                arg_b.push((dst_i, SsaI::Unary { op: *op, arg: arg_i.to_varid(), dtype }));
                arg_b
            }
            A::Reduce { op: _, arg: _, axis: _ } => {
                anyhow::bail!("TODO reduce")
            }
            A::Binary { op, lhs, rhs } => {
                let (lhs_i, lhs_b) = lhs.lower(range_id, per_arg)?;
                let (rhs_i, rhs_b) = rhs.lower(range_id, per_arg)?;
                let op =
                    SsaI::Binary { op: *op, dtype, lhs: lhs_i.to_varid(), rhs: rhs_i.to_varid() };
                [lhs_b.0.as_slice(), rhs_b.0.as_slice(), &[(dst_i, op)]].concat()
            }
        };
        Ok((dst_i, Block(instrs)))
    }
}

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
            let ptr_i = match per_arg.get(dst) {
                None => anyhow::bail!("unknown arg {dst:?}"),
                Some(id) => *id,
            };
            let len_i = Id::new();
            let num_el = layout.num_elements();
            instrs.push((len_i, SsaI::Const(ssa::Const::I32(num_el as i32))));
            let lo_i = Id::new();
            instrs.push((lo_i, SsaI::Const(ssa::Const::I32(0))));

            let range_id = Id::new();
            let range = SsaI::Range { lo: lo_i.to_varid(), up: len_i.to_varid(), end_idx: 42 };
            let start_line_idx = instrs.len();
            instrs.push((range_id, range));

            let (off_i, off_b) = layout.lower(range_id)?;
            instrs.extend_from_slice(off_b.0.as_slice());

            let (src_i, src_b) = value.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(src_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i.to_varid(),
                value: src_i.to_varid(),
                dtype: value.dtype,
            };
            instrs.push((Id::new(), store));

            let erange = ssa::Instr::EndRange { start_idx: start_line_idx };
            let end_line_idx = instrs.len();
            instrs.push((Id::new(), erange));
            if let SsaI::Range { end_idx, .. } = &mut instrs[start_line_idx].1 {
                *end_idx = end_line_idx + 1
            }
        }
        Ok(Block::new(instrs))
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
